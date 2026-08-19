"""Engine-neutral execution for adsorption simulations."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from itertools import combinations
from pathlib import Path

from chemgraph.schemas.adsorption_schema import (
    AdsorptionRequest,
    AdsorptionResult,
    AdsorptionSelectivity,
    ComponentUptake,
)
from chemgraph.tools.adsorption_config import (
    AdsorptionRuntimeConfig,
    load_adsorption_runtime,
)
from chemgraph.tools.adsorption_drivers import get_adsorption_driver


def _resolve_executable(executable: str) -> str:
    expanded = str(Path(executable).expanduser())
    if os.path.sep in expanded:
        path = Path(expanded)
        if not path.is_file():
            raise FileNotFoundError(f"Adsorption executable does not exist: {path}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Adsorption executable is not executable: {path}")
        return str(path.resolve())
    resolved = shutil.which(expanded)
    if resolved is None:
        raise FileNotFoundError(
            f"Adsorption executable {executable!r} was not found on PATH"
        )
    return resolved


def _selectivities(
    components: list[ComponentUptake],
) -> list[AdsorptionSelectivity]:
    results = []
    for numerator, denominator in combinations(components, 2):
        if denominator.uptake == 0:
            results.append(
                AdsorptionSelectivity(
                    numerator=numerator.name,
                    denominator=denominator.name,
                    value=None,
                    message="Selectivity is undefined because denominator uptake is zero",
                )
            )
            continue
        value = (
            numerator.uptake
            * denominator.feed_mole_fraction
            / (denominator.uptake * numerator.feed_mole_fraction)
        )
        results.append(
            AdsorptionSelectivity(
                numerator=numerator.name,
                denominator=denominator.name,
                value=value,
            )
        )
    return results


def _result(
    *,
    status: str,
    runtime: AdsorptionRuntimeConfig,
    request: AdsorptionRequest,
    cif_path: Path,
    workdir: Path,
    stdout_path: Path,
    stderr_path: Path,
    components: list[ComponentUptake] | None = None,
    return_code: int | None = None,
    wall_time: float | None = None,
    message: str | None = None,
) -> dict:
    component_results = components or []
    model = AdsorptionResult(
        status=status,
        engine=runtime.engine,
        temperature=float(request.temperature),
        pressure=float(request.pressure),
        components=component_results,
        selectivities=_selectivities(component_results),
        cif_path=str(cif_path),
        working_directory=str(workdir),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        return_code=return_code,
        wall_time_seconds=wall_time,
        message=message,
    )
    return model.model_dump(mode="json")


def run_adsorption_core(
    params: AdsorptionRequest | dict,
    *,
    runtime: AdsorptionRuntimeConfig | dict | None = None,
    config_path: str | None = None,
) -> dict:
    """Stage, execute, and parse one adsorption simulation."""

    from chemgraph.tools.ase_core import _resolve_existing_path, _resolve_path

    request = (
        params
        if isinstance(params, AdsorptionRequest)
        else AdsorptionRequest.model_validate(params)
    )
    if runtime is None:
        runtime_config = load_adsorption_runtime(config_path)
    elif isinstance(runtime, AdsorptionRuntimeConfig):
        runtime_config = runtime
    else:
        runtime_config = AdsorptionRuntimeConfig.model_validate(runtime)

    cif_path = Path(_resolve_existing_path(request.input_structure_file)).resolve()
    if not cif_path.is_file():
        raise FileNotFoundError(f"CIF file does not exist: {cif_path}")

    output_base = Path(_resolve_path(request.output_result_file)).expanduser().resolve()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    composition = "-".join(component.name for component in request.components)
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", cif_path.stem)
    run_name = (
        f"{safe_stem}--{composition}-T{float(request.temperature):g}-"
        f"P{float(request.pressure):g}-{uuid.uuid4().hex[:8]}"
    )
    workdir = output_base.parent / run_name
    workdir.mkdir(parents=True, exist_ok=False)
    stdout_path = workdir / output_base.name
    stderr_path = workdir / "raspa.err"

    driver = get_adsorption_driver(runtime_config.engine)
    driver.validate(request, cif_path)
    staged = driver.stage(request, cif_path, workdir)
    stdout_path.touch()
    stderr_path.touch()

    try:
        executable = _resolve_executable(runtime_config.executable)
    except Exception as exc:
        return _result(
            status="failure",
            runtime=runtime_config,
            request=request,
            cif_path=cif_path,
            workdir=workdir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            message=str(exc),
        )

    environment = os.environ.copy()
    environment.update(runtime_config.environment)
    started = time.monotonic()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                [executable],
                cwd=workdir,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                timeout=float(runtime_config.timeout_seconds),
                check=False,
                text=True,
            )
    except subprocess.TimeoutExpired:
        wall_time = time.monotonic() - started
        return _result(
            status="failure",
            runtime=runtime_config,
            request=request,
            cif_path=cif_path,
            workdir=workdir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            wall_time=wall_time,
            message=(
                f"Adsorption simulation exceeded "
                f"{runtime_config.timeout_seconds:g} seconds"
            ),
        )

    wall_time = time.monotonic() - started
    if completed.returncode != 0:
        return _result(
            status="failure",
            runtime=runtime_config,
            request=request,
            cif_path=cif_path,
            workdir=workdir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            return_code=completed.returncode,
            wall_time=wall_time,
            message=f"Adsorption executable exited with code {completed.returncode}",
        )

    try:
        components = driver.parse(stdout_path, request, staged)
    except Exception as exc:
        return _result(
            status="failure",
            runtime=runtime_config,
            request=request,
            cif_path=cif_path,
            workdir=workdir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            return_code=completed.returncode,
            wall_time=wall_time,
            message=f"Failed to parse {runtime_config.engine} output: {exc}",
        )

    return _result(
        status="success",
        runtime=runtime_config,
        request=request,
        cif_path=cif_path,
        workdir=workdir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        components=components,
        return_code=completed.returncode,
        wall_time=wall_time,
    )
