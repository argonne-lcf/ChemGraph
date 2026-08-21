"""Schemas for runtime-selectable machine-learned interatomic potentials."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemgraph.schemas.atomsdata import AtomsData


class _StrictModel(BaseModel):
    """Base model that rejects misspelled runtime and model options."""

    model_config = ConfigDict(extra="forbid")


class ASEMLIPRuntimeConfig(_StrictModel):
    """Execute MLIP calls through ASE calculators and optimizers."""

    type: Literal["ase"] = "ase"
    device: str = Field(default="cpu", description="Calculator device.")
    dtype: Literal["float32", "float64"] = "float64"
    optimizer: Literal["bfgs", "lbfgs", "gpmin", "fire", "mdmin"] = "bfgs"


class NVAlchemiRuntimeConfig(_StrictModel):
    """Execute MLIP calls through NVIDIA ALCHEMI Toolkit."""

    type: Literal["nvalchemi"] = "nvalchemi"
    device: str = Field(default="cuda", description="PyTorch device.")
    dtype: Literal["float32", "float64"] = "float32"
    optimizer: Literal["fire"] = "fire"
    dt: float = Field(default=0.1, gt=0.0)
    compile_model: bool = False
    enable_cueq: bool = False


MLIPRuntimeConfig = Annotated[
    Union[ASEMLIPRuntimeConfig, NVAlchemiRuntimeConfig],
    Field(discriminator="type"),
]


class MACEModelConfig(_StrictModel):
    """MACE checkpoint usable by the ASE and NVIDIA ALCHEMI runtimes."""

    provider: Literal["mace"] = "mace"
    checkpoint: str = Field(
        description="Named MACE checkpoint or path to a checkpoint file."
    )
    calculator_type: Literal["mace_mp", "mace_off", "mace_anicc"] = "mace_mp"
    dispersion: bool = False
    damping: str = "bj"
    dispersion_xc: str = "pbe"
    dispersion_cutoff: float = 21.167088422553647


class UMAModelConfig(_StrictModel):
    """UMA model configuration for the ASE runtime."""

    provider: Literal["uma"] = "uma"
    checkpoint: str = Field(description="Registered UMA model name.")
    task_name: Literal["omol", "omat", "oc20", "odac", "omc"] = "omol"
    inference_settings: Literal["default", "turbo"] = "default"
    charge: int = 0
    multiplicity: int = Field(default=1, ge=1)


class AIMNet2ModelConfig(_StrictModel):
    """AIMNet2 model configuration for the ASE runtime."""

    provider: Literal["aimnet2"] = "aimnet2"
    checkpoint: str = Field(description="AIMNet2 model alias or checkpoint path.")


class RootstockModelConfig(_StrictModel):
    """Rootstock-managed MLIP checkpoint exposed as an ASE calculator."""

    provider: Literal["rootstock"] = "rootstock"
    checkpoint: str
    cluster: str | None = None
    root: str | None = None
    cache_root: str | None = None
    setup_kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float = Field(default=600.0, gt=0.0)
    weights: str | None = None

    @model_validator(mode="after")
    def _validate_location(self) -> "RootstockModelConfig":
        if self.cluster is not None and self.root is not None:
            raise ValueError("Rootstock model cannot specify both cluster and root.")
        return self


MLIPModelConfig = Annotated[
    Union[MACEModelConfig, UMAModelConfig, AIMNet2ModelConfig, RootstockModelConfig],
    Field(discriminator="provider"),
]


class _MLIPCalculationConfig(_StrictModel):
    """Options shared by single-structure and batch MLIP requests."""

    driver: Literal["energy", "opt"] = "energy"
    runtime: MLIPRuntimeConfig = Field(default_factory=ASEMLIPRuntimeConfig)
    model: MLIPModelConfig
    fmax: float = Field(default=0.01, gt=0.0)
    steps: int = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def _validate_runtime_model_pair(self) -> "_MLIPCalculationConfig":
        if self.model.provider == "rootstock" and self.runtime.type != "ase":
            raise ValueError("Rootstock models require runtime.type='ase'.")
        if self.runtime.type == "nvalchemi" and self.model.provider != "mace":
            raise ValueError(
                "NVIDIA ALCHEMI runtime supports only provider='mace' in v1."
            )
        return self


class MLIPInputSchema(_MLIPCalculationConfig):
    """Input for a single MLIP energy or fixed-cell optimization calculation."""

    input_structure_file: str
    output_results_file: str = "output.json"

    @model_validator(mode="after")
    def _validate_output_extension(self) -> "MLIPInputSchema":
        if not self.output_results_file.lower().endswith(".json"):
            raise ValueError("output_results_file must end with '.json'.")
        return self


class MLIPBatchInputSchema(_MLIPCalculationConfig):
    """Input for a deterministic batch of MLIP calculations."""

    input_structure_files: list[str] | None = None
    input_structure_directory: str | None = None
    output_results_directory: str = "mlip_results"
    manifest_file: str = "batch_manifest.json"
    batch_size: int = Field(default=16, ge=1)
    max_atoms: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_input_source(self) -> "MLIPBatchInputSchema":
        sources = [
            bool(self.input_structure_files),
            self.input_structure_directory is not None,
        ]
        if sum(sources) != 1:
            raise ValueError(
                "Specify exactly one of input_structure_files or "
                "input_structure_directory."
            )
        if not self.manifest_file.lower().endswith(".json"):
            raise ValueError("manifest_file must end with '.json'.")
        return self


class MLIPOutputSchema(_StrictModel):
    """Runtime-neutral result compatible with the existing ASE result envelope."""

    schema_version: int = 1
    input_structure_file: str
    converged: bool = False
    final_structure: AtomsData | None = None
    simulation_input: dict[str, Any]
    single_point_energy: float | None = None
    energy_unit: str = "eV"
    forces: list[list[float]] | None = None
    force_unit: str = "eV/angstrom"
    stress: list[list[float]] | None = None
    stress_unit: str = "eV/angstrom^3"
    runtime_info: dict[str, Any]
    model_info: dict[str, Any]
    success: bool = False
    error: str = ""
    wall_time: float | None = None


class MLIPBatchItemSchema(_StrictModel):
    """One ordered item in an MLIP batch manifest."""

    index: int
    input_structure_file: str
    status: Literal["success", "failure"]
    result_file: str
    error: str = ""


class MLIPBatchManifestSchema(_StrictModel):
    """Persistent summary for an MLIP batch calculation."""

    schema_version: int = 1
    status: Literal["completed", "partial", "failure"]
    runtime_info: dict[str, Any]
    model_info: dict[str, Any]
    total: int
    succeeded: int
    failed: int
    wall_time: float
    items: list[MLIPBatchItemSchema]
