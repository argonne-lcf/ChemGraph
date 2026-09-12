"""Schemas for calculator-selectable machine-learned interatomic potentials."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemgraph.schemas.atomsdata import AtomsData


class _StrictModel(BaseModel):
    """Base model that rejects misspelled calculator and model options."""

    model_config = ConfigDict(extra="forbid")


_ASEOptimizer = Literal["bfgs", "lbfgs", "gpmin", "fire", "mdmin"]


class ASECalculatorConfig(_StrictModel):
    """Evaluate a model through an ASE calculator and optimizer."""

    backend: Literal["ase"] = "ase"
    device: str | None = Field(
        default=None,
        description="Calculator device override when the selected model supports it.",
    )
    dtype: Literal["float32", "float64"] | None = Field(
        default=None,
        description="Calculator dtype override when the selected model supports it.",
    )
    optimizer: _ASEOptimizer = "bfgs"


class NVAlchemiCalculatorConfig(_StrictModel):
    """Evaluate a MACE model through NVIDIA ALCHEMI Toolkit."""

    backend: Literal["nvalchemi"] = "nvalchemi"
    device: str = Field(default="cuda", description="PyTorch device.")
    dtype: Literal["float32", "float64"] = "float32"
    optimizer: Literal["fire"] = "fire"
    dt: float = Field(default=0.1, gt=0.0)
    compile_model: bool = False
    enable_cueq: bool = False


class RootstockCalculatorConfig(_StrictModel):
    """Evaluate a hosted model through a Rootstock ASE calculator."""

    backend: Literal["rootstock"] = "rootstock"
    cluster: str | None = None
    root: str | None = None
    cache_root: str | None = None
    device: str = Field(default="cuda", description="Rootstock worker device.")
    optimizer: _ASEOptimizer = "bfgs"
    setup_kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float = Field(default=600.0, gt=0.0)
    weights: str | None = None

    @model_validator(mode="after")
    def _validate_location(self) -> "RootstockCalculatorConfig":
        if self.cluster is not None and self.root is not None:
            raise ValueError("Rootstock calculator cannot specify both cluster and root.")
        return self


MLIPCalculatorConfig = Annotated[
    Union[
        ASECalculatorConfig,
        NVAlchemiCalculatorConfig,
        RootstockCalculatorConfig,
    ],
    Field(discriminator="backend"),
]


class MaceDispersionConfig(_StrictModel):
    """D3 dispersion correction supported by the ASE MACE adapter."""

    damping: str = "bj"
    xc: str = "pbe"
    cutoff: float = 21.167088422553647


class MACEModelConfig(_StrictModel):
    """Scientific identity and optional loader settings for a MACE model."""

    family: Literal["mace"] = "mace"
    checkpoint: str = Field(
        description="Named MACE checkpoint or path to a checkpoint file."
    )
    calculator_type: Literal["mace_mp", "mace_off", "mace_anicc"] | None = None
    dispersion: MaceDispersionConfig | None = None


class UMAModelConfig(_StrictModel):
    """Scientific identity and optional loader settings for a UMA model."""

    family: Literal["uma"] = "uma"
    checkpoint: str = Field(description="Registered UMA model name.")
    task_name: Literal["omol", "omat", "oc20", "odac", "omc"] | None = None
    inference_settings: Literal["default", "turbo"] | None = None
    charge: int = 0
    multiplicity: int = Field(default=1, ge=1)


class AIMNet2ModelConfig(_StrictModel):
    """Scientific identity for an AIMNet2 model."""

    family: Literal["aimnet2"] = "aimnet2"
    checkpoint: str = Field(description="AIMNet2 model alias or checkpoint path.")


MLIPModelConfig = Annotated[
    Union[MACEModelConfig, UMAModelConfig, AIMNet2ModelConfig],
    Field(discriminator="family"),
]


class _MLIPCalculationConfig(_StrictModel):
    """Options shared by single-structure and batch MLIP requests."""

    driver: Literal["energy", "opt"] = "energy"
    model: MLIPModelConfig
    calculator: MLIPCalculatorConfig = Field(default_factory=ASECalculatorConfig)
    fmax: float = Field(default=0.01, gt=0.0)
    steps: int = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def _validate_calculator_model_pair(self) -> "_MLIPCalculationConfig":
        model = self.model
        calculator = self.calculator

        if isinstance(calculator, ASECalculatorConfig):
            if isinstance(model, UMAModelConfig) and calculator.dtype is not None:
                raise ValueError("ASE UMA calculations do not support a dtype override.")
            if isinstance(model, AIMNet2ModelConfig) and (
                calculator.device is not None or calculator.dtype is not None
            ):
                raise ValueError(
                    "ASE AIMNet2 calculations do not support device or dtype overrides."
                )
            return self

        if isinstance(calculator, NVAlchemiCalculatorConfig):
            if not isinstance(model, MACEModelConfig):
                raise ValueError(
                    "NVIDIA ALCHEMI supports only model.family='mace' in v1."
                )
            if model.calculator_type not in (None, "mace_mp"):
                raise ValueError(
                    "NVIDIA ALCHEMI supports only MACE-MP checkpoints in v1."
                )
            if model.dispersion is not None:
                raise ValueError(
                    "NVIDIA ALCHEMI does not support MACE dispersion settings in v1."
                )
            return self

        if isinstance(model, MACEModelConfig) and (
            model.calculator_type is not None or model.dispersion is not None
        ):
            raise ValueError(
                "Rootstock resolves the MACE implementation from its canonical "
                "checkpoint; omit calculator_type and dispersion."
            )
        if isinstance(model, UMAModelConfig) and (
            model.task_name is not None or model.inference_settings is not None
        ):
            raise ValueError(
                "Rootstock UMA loader settings must be supplied through "
                "calculator.setup_kwargs."
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
    """Backend-neutral result compatible with the existing ASE result envelope."""

    schema_version: int = 1
    input_structure_file: str
    converged: bool = False
    final_structure: AtomsData | None = None
    simulation_input: dict[str, Any]
    potential_energy: float | None = None
    single_point_energy: float | None = None
    energy_unit: str = "eV"
    forces: list[list[float]] | None = None
    force_unit: str = "eV/angstrom"
    stress: list[list[float]] | None = None
    stress_unit: str = "eV/angstrom^3"
    calculator_info: dict[str, Any]
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
    calculator_info: dict[str, Any]
    model_info: dict[str, Any]
    total: int
    succeeded: int
    failed: int
    wall_time: float
    items: list[MLIPBatchItemSchema]
