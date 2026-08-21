"""Plain-Python core for runtime-selectable MLIP calculations.

This module owns all calculation, batching, lifecycle, and serialization
logic. Framework wrappers in ``run_mlip_tools.py`` and ``run_mlip_mcp.py``
only decorate and delegate to :func:`run_mlip_core` or
:func:`run_mlip_batch_core`.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from chemgraph.schemas.mlip_input import (
    AIMNet2ModelConfig,
    ASEMLIPRuntimeConfig,
    MACEModelConfig,
    MLIPBatchInputSchema,
    MLIPBatchItemSchema,
    MLIPBatchManifestSchema,
    MLIPInputSchema,
    MLIPOutputSchema,
    NVAlchemiRuntimeConfig,
    RootstockModelConfig,
    UMAModelConfig,
)
from chemgraph.tools.ase_core import (
    _resolve_existing_path,
    _resolve_path,
    atoms_to_atomsdata,
)


def _resolved_input_file(path: str) -> str:
    resolved = _resolve_existing_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Input structure file {path} does not exist.")
    return resolved


def _resolved_input_directory(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate
    log_candidate = Path(_resolve_path(path))
    if log_candidate.is_dir():
        return log_candidate
    raise FileNotFoundError(f"Input structure directory {path} does not exist.")


def _read_atoms(path: str):
    from ase.io import read

    try:
        return read(path)
    except Exception as exc:
        raise ValueError(f"Cannot read {path} using ASE: {exc}") from exc


def _write_result(result: MLIPOutputSchema, output_file: str) -> str:
    resolved = _resolve_path(output_file)
    output_path = Path(resolved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return str(output_path.resolve())


def _failure_result(
    params: MLIPInputSchema,
    input_file: str,
    error: Exception | str,
    wall_time: float,
) -> MLIPOutputSchema:
    return MLIPOutputSchema(
        input_structure_file=input_file,
        simulation_input=params.model_dump(mode="json"),
        runtime_info=params.runtime.model_dump(mode="json"),
        model_info=params.model.model_dump(mode="json"),
        success=False,
        error=str(error),
        wall_time=wall_time,
    )


def _success_result(
    params: MLIPInputSchema,
    input_file: str,
    atoms,
    energy: float,
    forces: Any,
    stress: Any,
    converged: bool,
    wall_time: float,
) -> MLIPOutputSchema:
    force_values = None if forces is None else forces.tolist()
    stress_values = None if stress is None else stress.tolist()
    return MLIPOutputSchema(
        input_structure_file=input_file,
        converged=converged,
        final_structure=atoms_to_atomsdata(atoms),
        simulation_input=params.model_dump(mode="json"),
        single_point_energy=float(energy),
        forces=force_values,
        stress=stress_values,
        runtime_info=params.runtime.model_dump(mode="json"),
        model_info=params.model.model_dump(mode="json"),
        success=True,
        wall_time=wall_time,
    )


@contextmanager
def _ase_calculator_context(
    params: MLIPInputSchema,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Create one ASE calculator and keep it alive for the surrounding batch."""
    runtime = params.runtime
    if not isinstance(runtime, ASEMLIPRuntimeConfig):
        raise TypeError("ASE calculator requested for a non-ASE runtime.")

    model = params.model
    if isinstance(model, RootstockModelConfig):
        try:
            from rootstock import RootstockCalculator
        except ImportError as exc:
            raise ImportError(
                "Rootstock provider requires the optional dependency. "
                "Install ChemGraph with the 'rootstock' extra."
            ) from exc

        kwargs = {
            "checkpoint": model.checkpoint,
            "cluster": model.cluster,
            "root": model.root,
            "cache_root": model.cache_root,
            "device": runtime.device,
            "setup_kwargs": model.setup_kwargs,
            "timeout": model.timeout,
            "weights": model.weights,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        with RootstockCalculator(**kwargs) as calculator:
            yield calculator, {}
        return

    if isinstance(model, MACEModelConfig):
        try:
            from chemgraph.schemas.calculators.mace_calc import (
                MaceCalc,
                mace_loading_lock,
            )
        except ImportError as exc:
            raise ImportError(
                "MACE provider requires mace-torch to be installed."
            ) from exc

        config = MaceCalc(
            calculator_type=model.calculator_type,
            model=model.checkpoint,
            device=runtime.device,
            default_dtype=runtime.dtype,
            dispersion=model.dispersion,
            damping=model.damping,
            dispersion_xc=model.dispersion_xc,
            dispersion_cutoff=model.dispersion_cutoff,
        )
        with mace_loading_lock():
            calculator = config.get_calculator()
        yield calculator, {}
        return

    if isinstance(model, UMAModelConfig):
        try:
            from chemgraph.schemas.calculators.fairchem_calc import FAIRChemCalc
        except ImportError as exc:
            raise ImportError(
                "UMA provider requires fairchem-core to be installed."
            ) from exc

        config = FAIRChemCalc(
            model_name=model.checkpoint,
            task_name=model.task_name,
            inference_settings=model.inference_settings,
            device=runtime.device,
            charge=model.charge,
            multiplicity=model.multiplicity,
        )
        yield config.get_calculator(), config.get_atoms_properties()
        return

    if isinstance(model, AIMNet2ModelConfig):
        try:
            from chemgraph.schemas.calculators.aimnet2_calc import AIMNET2Calc
        except ImportError as exc:
            raise ImportError(
                "AIMNet2 provider requires aimnet2calc to be installed."
            ) from exc

        config = AIMNET2Calc(model=model.checkpoint)
        yield config.get_calculator(), {}
        return

    raise ValueError(f"Unsupported ASE MLIP provider: {model.provider}")


def _optional_stress(atoms):
    if not atoms.pbc.any() or atoms.cell.rank != 3:
        return None
    try:
        return atoms.get_stress(voigt=False)
    except Exception:
        return None


def _run_ase_calculation(
    params: MLIPInputSchema,
    input_file: str,
    atoms,
    calculator: Any,
    atoms_info: dict[str, Any],
    start_time: float,
) -> MLIPOutputSchema:
    from ase.optimize import BFGS, FIRE, GPMin, LBFGS, MDMin

    runtime = params.runtime
    if not isinstance(runtime, ASEMLIPRuntimeConfig):
        raise TypeError("ASE calculation requested for a non-ASE runtime.")

    atoms.info.update(atoms_info)
    atoms.calc = calculator
    converged = True
    if params.driver == "opt" and len(atoms) > 1:
        optimizer_classes = {
            "bfgs": BFGS,
            "lbfgs": LBFGS,
            "gpmin": GPMin,
            "fire": FIRE,
            "mdmin": MDMin,
        }
        optimizer = optimizer_classes[runtime.optimizer](atoms, logfile=None)
        converged = bool(optimizer.run(fmax=params.fmax, steps=params.steps))

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    stress = _optional_stress(atoms)
    return _success_result(
        params,
        input_file,
        atoms,
        energy,
        forces,
        stress,
        converged,
        time.time() - start_time,
    )


def _load_nvalchemi_model(params: MLIPInputSchema):
    runtime = params.runtime
    model_config = params.model
    if not isinstance(runtime, NVAlchemiRuntimeConfig) or not isinstance(
        model_config, MACEModelConfig
    ):
        raise TypeError("Invalid NVIDIA ALCHEMI runtime/model configuration.")

    try:
        import torch
        from nvalchemi.models.mace import MACEWrapper
    except ImportError as exc:
        raise ImportError(
            "NVIDIA ALCHEMI runtime requires ChemGraph's 'nvalchemi_mace' "
            "extra plus the CUDA-specific ALCHEMI dependencies."
        ) from exc

    dtype = getattr(torch, runtime.dtype)
    wrapped = MACEWrapper.from_checkpoint(
        model_config.checkpoint,
        device=torch.device(runtime.device),
        dtype=dtype,
        enable_cueq=runtime.enable_cueq,
        compile_model=runtime.compile_model,
    )
    wrapped.eval()
    return wrapped


def _atoms_to_nvalchemi_data(atoms, runtime: NVAlchemiRuntimeConfig):
    try:
        import torch
        from nvalchemi.data import AtomicData
    except ImportError as exc:
        raise ImportError(
            "NVIDIA ALCHEMI runtime requires ChemGraph's 'nvalchemi_mace' "
            "extra plus the CUDA-specific ALCHEMI dependencies."
        ) from exc

    dtype = getattr(torch, runtime.dtype)
    data = AtomicData.from_atoms(
        atoms,
        device=torch.device(runtime.device),
        dtype=dtype,
    )
    data.forces = torch.zeros(data.num_nodes, 3, device=data.device, dtype=dtype)
    data.energy = torch.zeros(1, 1, device=data.device, dtype=dtype)
    data.velocities = torch.zeros(
        data.num_nodes, 3, device=data.device, dtype=dtype
    )
    return data


def _nvalchemi_result(
    params: MLIPInputSchema,
    input_file: str,
    original_atoms,
    data,
    converged: bool,
    start_time: float,
) -> MLIPOutputSchema:
    final_atoms = original_atoms.copy()
    final_atoms.positions = data.positions.detach().cpu().numpy()
    if data.cell is not None:
        final_atoms.cell = data.cell.squeeze(0).detach().cpu().numpy()
    if data.pbc is not None:
        final_atoms.pbc = data.pbc.squeeze(0).detach().cpu().numpy()

    energy = data.energy.detach().cpu().reshape(-1)[0].item()
    forces = data.forces.detach().cpu().numpy() if data.forces is not None else None
    stress = None
    if data.stress is not None:
        stress = data.stress.detach().cpu().reshape(-1, 3, 3)[0].numpy()
    return _success_result(
        params,
        input_file,
        final_atoms,
        energy,
        forces,
        stress,
        converged,
        time.time() - start_time,
    )


def _run_nvalchemi_chunk(
    model: Any,
    entries: Sequence[tuple[MLIPInputSchema, str, Any, float]],
) -> list[MLIPOutputSchema]:
    try:
        from nvalchemi.data import Batch
        from nvalchemi.dynamics import BaseDynamics, ConvergenceHook, FIRE
    except ImportError as exc:
        raise ImportError(
            "NVIDIA ALCHEMI runtime requires ChemGraph's 'nvalchemi_mace' "
            "extra plus the CUDA-specific ALCHEMI dependencies."
        ) from exc

    params0 = entries[0][0]
    runtime = params0.runtime
    if not isinstance(runtime, NVAlchemiRuntimeConfig):
        raise TypeError("NVIDIA ALCHEMI calculation received a non-ALCHEMI runtime.")

    data_list = [
        _atoms_to_nvalchemi_data(atoms, runtime) for _, _, atoms, _ in entries
    ]
    batch = Batch.from_data_list(data_list)
    hooks = model.make_neighbor_hooks()
    convergence = None
    if params0.driver == "energy":
        dynamics = BaseDynamics(model=model, hooks=hooks, n_steps=1)
    else:
        convergence = ConvergenceHook.from_fmax(params0.fmax)
        dynamics = FIRE(
            model=model,
            dt=runtime.dt,
            hooks=hooks,
            convergence_hook=convergence,
            n_steps=params0.steps,
        )

    with dynamics:
        batch = dynamics.run(batch)

    converged_indices = set(range(len(entries)))
    if convergence is not None:
        indices = convergence.evaluate(batch)
        converged_indices = (
            set() if indices is None else set(indices.detach().cpu().tolist())
        )

    final_data = batch.to_data_list()
    return [
        _nvalchemi_result(
            params,
            input_file,
            atoms,
            data,
            index in converged_indices,
            start_time,
        )
        for index, ((params, input_file, atoms, start_time), data) in enumerate(
            zip(entries, final_data)
        )
    ]


def _chunks_by_capacity(
    entries: Sequence[tuple[MLIPInputSchema, str, Any, float]],
    batch_size: int,
    max_atoms: int | None,
) -> Iterator[list[tuple[MLIPInputSchema, str, Any, float]]]:
    chunk: list[tuple[MLIPInputSchema, str, Any, float]] = []
    atom_count = 0
    for entry in entries:
        n_atoms = len(entry[2])
        exceeds_atoms = max_atoms is not None and chunk and atom_count + n_atoms > max_atoms
        if len(chunk) >= batch_size or exceeds_atoms:
            yield chunk
            chunk = []
            atom_count = 0
        chunk.append(entry)
        atom_count += n_atoms
    if chunk:
        yield chunk


def _execute_single_requests(
    params_list: Sequence[MLIPInputSchema],
    batch_size: int = 16,
    max_atoms: int | None = None,
) -> list[MLIPOutputSchema]:
    results: list[MLIPOutputSchema | None] = [None] * len(params_list)
    prepared: list[tuple[int, MLIPInputSchema, str, Any, float]] = []

    for index, params in enumerate(params_list):
        started = time.time()
        input_file = params.input_structure_file
        try:
            input_file = _resolved_input_file(input_file)
            atoms = _read_atoms(input_file)
            prepared.append((index, params, input_file, atoms, started))
        except Exception as exc:
            results[index] = _failure_result(
                params, input_file, exc, time.time() - started
            )

    if not prepared:
        return [result for result in results if result is not None]

    first_params = prepared[0][1]
    if first_params.runtime.type == "ase":
        try:
            with _ase_calculator_context(first_params) as (calculator, atoms_info):
                for index, params, input_file, atoms, started in prepared:
                    try:
                        results[index] = _run_ase_calculation(
                            params,
                            input_file,
                            atoms,
                            calculator,
                            atoms_info,
                            started,
                        )
                    except Exception as exc:
                        results[index] = _failure_result(
                            params, input_file, exc, time.time() - started
                        )
        except Exception as exc:
            for index, params, input_file, _, started in prepared:
                results[index] = _failure_result(
                    params, input_file, exc, time.time() - started
                )
    else:
        try:
            model = _load_nvalchemi_model(first_params)
            nvalchemi_entries = [
                (params, input_file, atoms, started)
                for _, params, input_file, atoms, started in prepared
            ]
            offset = 0
            for chunk in _chunks_by_capacity(
                nvalchemi_entries, batch_size, max_atoms
            ):
                try:
                    chunk_results = _run_nvalchemi_chunk(model, chunk)
                    if len(chunk_results) != len(chunk):
                        raise RuntimeError(
                            "NVIDIA ALCHEMI returned a different number of "
                            "results than input structures."
                        )
                except Exception as exc:
                    chunk_results = [
                        _failure_result(
                            params, input_file, exc, time.time() - started
                        )
                        for params, input_file, _, started in chunk
                    ]
                for chunk_index, result in enumerate(chunk_results):
                    original_index = prepared[offset + chunk_index][0]
                    results[original_index] = result
                offset += len(chunk)
        except Exception as exc:
            for index, params, input_file, _, started in prepared:
                results[index] = _failure_result(
                    params, input_file, exc, time.time() - started
                )

    return [result for result in results if result is not None]


def _tool_result(result: MLIPOutputSchema, result_file: str) -> dict[str, Any]:
    if result.success:
        return {
            "status": "success",
            "message": f"MLIP calculation completed. Results saved to {result_file}",
            "output_results_file": result_file,
            "single_point_energy": result.single_point_energy,
            "unit": result.energy_unit,
            "converged": result.converged,
        }
    return {
        "status": "failure",
        "error_type": "MLIPCalculationError",
        "message": result.error,
        "output_results_file": result_file,
    }


def run_mlip_core(params: MLIPInputSchema) -> dict[str, Any]:
    """Run one MLIP calculation and persist a runtime-neutral JSON result."""
    if not isinstance(params, MLIPInputSchema):
        params = MLIPInputSchema.model_validate(params)
    result = _execute_single_requests([params], batch_size=1)[0]
    try:
        result_file = _write_result(result, params.output_results_file)
    except Exception as exc:
        return {
            "status": "failure",
            "error_type": type(exc).__name__,
            "message": f"Could not write MLIP result: {exc}",
        }
    return _tool_result(result, result_file)


def _batch_input_files(params: MLIPBatchInputSchema) -> list[str]:
    if params.input_structure_files is not None:
        return list(params.input_structure_files)
    directory = _resolved_input_directory(params.input_structure_directory or "")
    return [str(path) for path in sorted(directory.iterdir()) if path.is_file()]


def _single_params_for_batch_item(
    batch_params: MLIPBatchInputSchema,
    input_file: str,
    output_file: str,
) -> MLIPInputSchema:
    return MLIPInputSchema(
        input_structure_file=input_file,
        output_results_file=output_file,
        driver=batch_params.driver,
        runtime=batch_params.runtime,
        model=batch_params.model,
        fmax=batch_params.fmax,
        steps=batch_params.steps,
    )


def run_mlip_batch_core(params: MLIPBatchInputSchema) -> dict[str, Any]:
    """Run an ordered MLIP batch and write per-item results plus a manifest."""
    if not isinstance(params, MLIPBatchInputSchema):
        params = MLIPBatchInputSchema.model_validate(params)

    started = time.time()
    output_directory = Path(_resolve_path(params.output_results_directory))
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(params.manifest_file)
    if not manifest_path.is_absolute():
        manifest_path = output_directory / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        input_files = _batch_input_files(params)
    except Exception as exc:
        return {
            "status": "failure",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    single_params = [
        _single_params_for_batch_item(
            params,
            input_file,
            str(output_directory / f"{index:05d}_{Path(input_file).stem}.json"),
        )
        for index, input_file in enumerate(input_files)
    ]
    results = _execute_single_requests(
        single_params,
        batch_size=params.batch_size,
        max_atoms=params.max_atoms,
    )

    items: list[MLIPBatchItemSchema] = []
    for index, (single, result) in enumerate(zip(single_params, results)):
        try:
            result_file = _write_result(result, single.output_results_file)
        except Exception as exc:
            result_file = str(Path(single.output_results_file).resolve())
            result = _failure_result(
                single,
                result.input_structure_file,
                f"Could not write MLIP result: {exc}",
                result.wall_time or 0.0,
            )
        items.append(
            MLIPBatchItemSchema(
                index=index,
                input_structure_file=result.input_structure_file,
                status="success" if result.success else "failure",
                result_file=result_file,
                error=result.error,
            )
        )

    succeeded = sum(item.status == "success" for item in items)
    failed = len(items) - succeeded
    if items and failed == 0:
        status = "completed"
    elif succeeded:
        status = "partial"
    else:
        status = "failure"

    manifest = MLIPBatchManifestSchema(
        status=status,
        runtime_info=params.runtime.model_dump(mode="json"),
        model_info=params.model.model_dump(mode="json"),
        total=len(items),
        succeeded=succeeded,
        failed=failed,
        wall_time=time.time() - started,
        items=items,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    manifest_file = str(manifest_path.resolve())
    return {
        "status": status,
        "manifest_file": manifest_file,
        "total": len(items),
        "succeeded": succeeded,
        "failed": failed,
        "message": f"MLIP batch {status}. Manifest saved to {manifest_file}",
    }
