"""Input and output adapters for supported adsorption engines."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from chemgraph.schemas.adsorption_schema import (
    AdsorptionRequest,
    ComponentUptake,
    EngineOptionValue,
)


@dataclass(frozen=True)
class EngineCapabilities:
    """Features guaranteed by one tested engine adapter."""

    min_components: int
    max_components: int
    supported_adsorbates: frozenset[str]
    supports_mole_fractions: bool
    gpus_per_task: int
    input_format: str
    output_format: str


@dataclass(frozen=True)
class StagedSimulation:
    cif_path: Path
    unit_cells: tuple[int, int, int]


_AVERAGE_RE = re.compile(
    r"Overall:\s*Average:?\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
    r"(?:\s*,?\s*(?:Error:?|\+/-)\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?))?",
    re.IGNORECASE,
)
_COMPONENT_RE = re.compile(r"\bComponent\s+(\d+)\b", re.IGNORECASE)


def _format_option(value: EngineOptionValue) -> str:
    values = value if isinstance(value, list) else [value]

    def scalar(item: bool | int | float | str) -> str:
        if isinstance(item, bool):
            return "yes" if item else "no"
        return str(item)

    return " ".join(scalar(item) for item in values)


def _calculate_cell_size(cif_path: Path, cutoff: float) -> tuple[int, int, int]:
    atoms = ase_read(cif_path)
    a, b, c = atoms.cell[:]

    def perpendicular_width(v1, v2, opposite) -> float:
        cross = np.cross(v1, v2)
        norm = np.linalg.norm(cross)
        if norm == 0:
            raise ValueError(f"Degenerate unit cell in {cif_path}")
        return float(abs(np.dot(cross, opposite)) / norm)

    widths = (
        perpendicular_width(b, c, a),
        perpendicular_width(c, a, b),
        perpendicular_width(a, b, c),
    )
    return tuple(max(1, int(math.ceil(2.0 * cutoff / width))) for width in widths)


def _framework_mass(cif_path: Path, unit_cells: tuple[int, int, int]) -> float:
    atoms = ase_read(cif_path)
    return float(sum(atoms.get_masses()) * math.prod(unit_cells))


class AdsorptionDriver(ABC):
    engine: str
    asset_package: str
    aliases: dict[str, str]
    capabilities: EngineCapabilities

    def validate_capabilities(self, request: AdsorptionRequest) -> None:
        count = len(request.components)
        if not self.capabilities.min_components <= count <= self.capabilities.max_components:
            names = ", ".join(component.name for component in request.components)
            raise ValueError(
                f"Engine {self.engine!r} supports "
                f"{self.capabilities.min_components}-"
                f"{self.capabilities.max_components} component(s); received "
                f"{count}: {names}. Use 'graspa_cuda' for mixture adsorption."
            )
        unsupported = {
            component.name
            for component in request.components
            if component.name not in self.capabilities.supported_adsorbates
        }
        if unsupported:
            raise ValueError(
                f"Engine {self.engine!r} does not support: {sorted(unsupported)}"
            )
        if count > 1 and not self.capabilities.supports_mole_fractions:
            raise ValueError(f"Engine {self.engine!r} does not support mixtures")

    def validate(self, request: AdsorptionRequest, cif_path: Path) -> None:
        self.validate_capabilities(request)
        if "_atom_site_charge" not in cif_path.read_text(
            encoding="utf-8", errors="ignore"
        ).lower():
            raise ValueError(
                "The bundled Ewald profile requires a CIF containing "
                "_atom_site_charge values"
            )

    def stage(
        self, request: AdsorptionRequest, cif_path: Path, workdir: Path
    ) -> StagedSimulation:
        package = resources.files(
            f"chemgraph.tools.files.{self.asset_package}"
        )
        for item in package.iterdir():
            if item.name.startswith("__") or not item.is_file():
                continue
            (workdir / item.name).write_bytes(item.read_bytes())

        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", cif_path.stem).strip("_")
        if not safe_stem:
            raise ValueError(f"Could not derive a safe framework name from {cif_path}")
        staged_cif = workdir / f"{safe_stem}.cif"
        staged_cif.write_bytes(cif_path.read_bytes())
        unit_cells = _calculate_cell_size(cif_path, float(request.cutoff))

        input_path = workdir / "simulation.input"
        text = input_path.read_text(encoding="utf-8")
        replacements = {
            "NCYCLE": str(request.n_cycles),
            "TEMPERATURE": str(request.temperature),
            "PRESSURE": str(request.pressure),
            "CUTOFF": str(request.cutoff),
            "CIFFILE": safe_stem,
            "UC_X": str(unit_cells[0]),
            "UC_Y": str(unit_cells[1]),
            "UC_Z": str(unit_cells[2]),
            "__ENGINE_OPTIONS__": self._render_options(request.engine_options),
            "__COMPONENTS__": self.render_components(request),
        }
        for token, value in replacements.items():
            text = text.replace(token, value)
        unresolved = [token for token in replacements if token in text]
        if unresolved:
            raise ValueError(f"Unresolved simulation template tokens: {unresolved}")
        input_path.write_text(text, encoding="utf-8")
        return StagedSimulation(staged_cif, unit_cells)

    @staticmethod
    def _render_options(options: dict[str, EngineOptionValue]) -> str:
        return "\n".join(
            f"{key} {_format_option(value)}" for key, value in options.items()
        )

    @abstractmethod
    def render_components(self, request: AdsorptionRequest) -> str:
        pass

    @abstractmethod
    def parse(
        self,
        output_path: Path,
        request: AdsorptionRequest,
        staged: StagedSimulation,
    ) -> list[ComponentUptake]:
        pass


class GraspaSyclDriver(AdsorptionDriver):
    engine = "graspa_sycl"
    asset_package = "template_graspa_sycl"
    aliases = {"CO2": "CO2", "N2": "N2", "H2O": "H2O"}
    capabilities = EngineCapabilities(
        min_components=1,
        max_components=1,
        supported_adsorbates=frozenset(aliases),
        supports_mole_fractions=False,
        gpus_per_task=1,
        input_format="graspa-sycl-text",
        output_format="graspa-sycl-text",
    )

    def render_components(self, request: AdsorptionRequest) -> str:
        component = request.components[0]
        fugacity = component.fugacity_coefficient
        if fugacity in (None, "PR-EOS"):
            fugacity = 1.0
        lines = [
            f"Component 0 MoleculeName             {self.aliases[component.name]}",
            f"            IdealGasRosenbluthWeight {component.ideal_gas_rosenbluth_weight}",
            f"            FugacityCoefficient      {fugacity}",
            f"            TranslationProbability   {component.translation_probability}",
            f"            RotationProbability      {component.rotation_probability}",
            f"            SwapProbability          {component.swap_probability}",
            f"            CreateNumberOfMolecules  {component.create_number_of_molecules}",
        ]
        lines.extend(
            f"            {key} {_format_option(value)}"
            for key, value in component.engine_options.items()
        )
        return "\n".join(lines)

    def parse(
        self,
        output_path: Path,
        request: AdsorptionRequest,
        staged: StagedSimulation,
    ) -> list[ComponentUptake]:
        candidates: list[tuple[str, float, float | None]] = []
        context = ""
        for raw in output_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _AVERAGE_RE.search(raw)
            if match:
                candidates.append(
                    (
                        context,
                        float(match.group(1)),
                        float(match.group(2)) if match.group(2) else None,
                    )
                )
            elif raw.strip():
                context = raw.strip()
        if not candidates:
            raise ValueError("No 'Overall: Average' value found in SYCL output")
        molecule_values = [
            candidate
            for candidate in candidates
            if "molecule" in candidate[0].lower()
        ]
        _, count, count_error = (molecule_values or candidates)[-1]
        mass = _framework_mass(staged.cif_path, staged.unit_cells)
        scale = 1000.0 / mass
        component = request.components[0]
        return [
            ComponentUptake(
                name=component.name,
                feed_mole_fraction=float(component.mole_fraction),
                uptake=count * scale,
                uncertainty=count_error * scale if count_error is not None else None,
            )
        ]


class GraspaCudaDriver(AdsorptionDriver):
    engine = "graspa_cuda"
    asset_package = "template_graspa_cuda"
    aliases = {"CO2": "CO2", "N2": "N2", "H2O": "TIP4P"}
    capabilities = EngineCapabilities(
        min_components=1,
        max_components=3,
        supported_adsorbates=frozenset(aliases),
        supports_mole_fractions=True,
        gpus_per_task=1,
        input_format="graspa-cuda-text",
        output_format="graspa-cuda-text",
    )

    def render_components(self, request: AdsorptionRequest) -> str:
        mixture = len(request.components) > 1
        blocks = []
        for index, component in enumerate(request.components):
            fugacity = component.fugacity_coefficient or "PR-EOS"
            lines = [
                f"Component {index} MoleculeName              {self.aliases[component.name]}",
                f"             IdealGasRosenbluthWeight {component.ideal_gas_rosenbluth_weight}",
                f"             FugacityCoefficient      {fugacity}",
                f"             MolFraction               {component.mole_fraction}",
                f"             TranslationProbability    {component.translation_probability}",
                f"             RotationProbability       {component.rotation_probability}",
                f"             ReinsertionProbability    {component.reinsertion_probability}",
            ]
            identity = component.identity_change_probability
            if mixture and identity is None:
                identity = 1.0
            if identity is not None:
                lines.append(f"             IdentityChangeProbability {identity}")
            lines.extend(
                [
                    f"             SwapProbability           {component.swap_probability}",
                    "             CreateNumberOfMolecules  "
                    f"{component.create_number_of_molecules}",
                ]
            )
            lines.extend(
                f"             {key} {_format_option(value)}"
                for key, value in component.engine_options.items()
            )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def parse(
        self,
        output_path: Path,
        request: AdsorptionRequest,
        staged: StagedSimulation,
    ) -> list[ComponentUptake]:
        del staged
        current_component = 0 if len(request.components) == 1 else None
        context = ""
        values: dict[int, tuple[float, float | None]] = {}
        fallback_values: dict[int, tuple[float, float | None]] = {}
        for raw in output_path.read_text(encoding="utf-8", errors="replace").splitlines():
            component_match = _COMPONENT_RE.search(raw)
            if component_match:
                current_component = int(component_match.group(1))
                context = raw.strip()
                continue
            average_match = _AVERAGE_RE.search(raw)
            metric = f"{context} {raw}".lower()
            if average_match and current_component is not None and "mol/kg" in metric:
                parsed = (
                    float(average_match.group(1)),
                    float(average_match.group(2)) if average_match.group(2) else None,
                )
                fallback_values[current_component] = parsed
                if "absolute" in metric:
                    values[current_component] = parsed
            elif raw.strip():
                context = raw.strip()

        values = {**fallback_values, **values}
        missing = [index for index in range(len(request.components)) if index not in values]
        if missing:
            raise ValueError(
                "Missing CUDA mol/kg statistics for component index(es): "
                + ", ".join(str(index) for index in missing)
            )
        return [
            ComponentUptake(
                name=component.name,
                feed_mole_fraction=float(component.mole_fraction),
                uptake=values[index][0],
                uncertainty=values[index][1],
            )
            for index, component in enumerate(request.components)
        ]


_DRIVERS: dict[str, AdsorptionDriver] = {
    driver.engine: driver for driver in (GraspaSyclDriver(), GraspaCudaDriver())
}


def get_adsorption_driver(engine: str) -> AdsorptionDriver:
    """Return a registered driver or raise an actionable error."""

    try:
        return _DRIVERS[engine]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported adsorption engine {engine!r}; "
            f"choose one of {sorted(_DRIVERS)}"
        ) from exc
