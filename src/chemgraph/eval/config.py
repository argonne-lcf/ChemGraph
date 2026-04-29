"""Configuration models for ChemGraph evaluation benchmarks."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from pydantic import BaseModel, Field, field_validator, model_validator

from chemgraph.eval.datasets import default_dataset_path
from chemgraph.utils.config_utils import (
    flatten_config,
    get_argo_user_from_flat_config,
    get_base_url_for_model_from_flat_config,
)


class BenchmarkConfig(BaseModel):
    """Configuration for a multi-model evaluation benchmark run.

    Evaluation is performed using an **LLM-as-judge** strategy: a
    separate judge LLM grades the agent's tool-call sequence and final
    answer against the ground-truth result using binary scoring
    (1 = correct, 0 = wrong).

    Parameters
    ----------
    models : list[str]
        List of LLM model names to evaluate.
    workflow_types : list[str]
        Workflow types to test each model against.  Common choices are
        ``"mock_agent"`` (tool-call accuracy only, no execution) and
        ``"single_agent"`` (end-to-end with tool execution).
    dataset : str
        Path to a ground-truth JSON file.  Defaults to the bundled
        ``data/ground_truth.json`` shipped with the package.  Accepts
        both the *list* format and the *dict* format.
    output_dir : str
        Directory where per-model results, aggregate reports and raw
        tool-call logs are written.
    structured_output : bool
        Whether to enable structured output on the ``ChemGraph`` agent.
    recursion_limit : int
        Maximum number of LangGraph recursion steps per query.
    judge_model : str
        LLM model name to use as the judge.  Must be different from the
        models under test to avoid self-evaluation bias.
    tags : list[str]
        Optional free-form tags attached to the run metadata (e.g.
        ``["nightly", "ci"]``).
    max_queries : int
        Maximum number of queries to evaluate from the dataset.
        0 means evaluate all queries (no limit).
    resume : bool
        When ``True``, load per-query checkpoint files from the output
        directory and skip already-completed ``(model, workflow, query)``
        combinations.  Useful for resuming after a crash.
    config_file : str, optional
        Path to a TOML configuration file (e.g. ``config.toml``).
    """

    models: List[str] = Field(
        ...,
        min_length=1,
        description="LLM model names to benchmark.",
    )
    workflow_types: List[str] = Field(
        default=["single_agent"],
        description="Workflow graph types to evaluate.",
    )
    dataset: str = Field(
        default_factory=default_dataset_path,
        description=(
            "Path to ground-truth JSON file. "
            "Defaults to the bundled dataset shipped with the package."
        ),
    )
    output_dir: str = Field(
        default="eval_results",
        description="Output directory for results.",
    )
    structured_output: bool = Field(
        default=True,
        description="Enable structured output on ChemGraph agent.",
    )
    recursion_limit: int = Field(
        default=50,
        description="Max LangGraph recursion steps per query.",
    )
    judge_model: Optional[str] = Field(
        default=None,
        description=(
            "LLM model name for the judge. Required when judge_type "
            "is 'llm' or 'both'; ignored for 'structured'."
        ),
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Optional tags for the benchmark run.",
    )
    max_queries: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum number of queries to evaluate from the dataset. "
            "0 means evaluate all queries (no limit)."
        ),
    )
    judge_type: str = Field(
        default="llm",
        description=(
            "Judge strategy to use: 'llm' (LLM-as-judge only), "
            "'structured' (deterministic structured-output comparison "
            "only), or 'both' (run both judges side by side)."
        ),
    )
    resume: bool = Field(
        default=False,
        description=(
            "Resume from per-query checkpoint files, skipping "
            "already-completed (model, workflow, query) combinations. "
            "Checkpoints are always written regardless of this flag."
        ),
    )
    config_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to a TOML configuration file (e.g. config.toml). "
            "When provided, model base_url and argo_user are resolved "
            "from the [api.*] sections, matching the main CLI behaviour. "
            "Eval profiles are also loaded from [eval.profiles.*]."
        ),
    )

    # Internal cache for the flattened config -- not part of the public schema.
    _flat_config: Dict[str, Any] = {}
    # Cache the raw (non-flattened) config for profile access.
    _raw_config: Dict[str, Any] = {}

    @field_validator("dataset")
    @classmethod
    def dataset_must_exist(cls, v: str) -> str:
        p = Path(v)
        if not p.exists():
            raise ValueError(f"Dataset file does not exist: {v}")
        if p.suffix != ".json":
            raise ValueError(f"Dataset must be a .json file, got: {p.suffix}")
        return str(p.resolve())

    @model_validator(mode="after")
    def load_config_file(self):
        """Load and cache the flattened TOML config when *config_file* is set."""
        if self.config_file:
            p = Path(self.config_file)
            if not p.exists():
                raise ValueError(f"Config file does not exist: {self.config_file}")
            with open(p) as fh:
                raw = toml.load(fh)
            self._flat_config = flatten_config(raw)
            self._raw_config = raw
        return self

    @model_validator(mode="after")
    def validate_judge_model_required(self):
        """Ensure *judge_model* is set when the LLM judge is requested."""
        if self.judge_type in ("llm", "both") and not self.judge_model:
            raise ValueError(
                f"judge_model is required when judge_type is "
                f"'{self.judge_type}'. Provide --judge-model or set "
                f"judge_type to 'structured' to skip the LLM judge."
            )
        return self

    @field_validator("judge_type")
    @classmethod
    def validate_judge_type(cls, v: str) -> str:
        valid = {"llm", "structured", "both"}
        if v not in valid:
            raise ValueError(f"Unknown judge_type: {v!r}. Valid: {sorted(valid)}")
        return v

    @field_validator("workflow_types")
    @classmethod
    def validate_workflow_types(cls, v: List[str]) -> List[str]:
        valid = {
            "single_agent",
            "multi_agent",
            "single_agent_mcp",
        }
        for wf in v:
            if wf not in valid:
                raise ValueError(
                    f"Unknown workflow type: {wf!r}. Valid: {sorted(valid)}"
                )
        return v

    # ------------------------------------------------------------------
    # Helpers for per-model config resolution
    # ------------------------------------------------------------------

    def get_base_url(self, model_name: str) -> Optional[str]:
        """Resolve the provider base URL for *model_name* from the config file.

        Returns ``None`` when no config file was provided (the provider
        loaders will fall back to their defaults / environment variables).
        """
        if not self._flat_config:
            return None
        return get_base_url_for_model_from_flat_config(model_name, self._flat_config)

    def get_argo_user(self) -> Optional[str]:
        """Resolve the Argo user from the config file, if present."""
        if not self._flat_config:
            return None
        return get_argo_user_from_flat_config(self._flat_config)

    # ------------------------------------------------------------------
    # Profile-based construction
    # ------------------------------------------------------------------

    @classmethod
    def from_profile(
        cls,
        profile_name: str,
        models: List[str],
        config_file: str,
        **overrides,
    ) -> "BenchmarkConfig":
        """Create a ``BenchmarkConfig`` from a named profile in ``config.toml``.

        Profile values are read from ``[eval.profiles.<name>]``.  Any
        keyword arguments in *overrides* take precedence over the profile
        values, allowing CLI flags to selectively override profile
        defaults.

        Parameters
        ----------
        profile_name : str
            Name of the profile (e.g. ``"quick"``, ``"standard"``).
        models : list[str]
            LLM model names (always required, not part of profiles).
        config_file : str
            Path to the TOML config file containing ``[eval.profiles.*]``.
        **overrides
            Any ``BenchmarkConfig`` fields to override.  ``None`` values
            are ignored so that unset CLI flags don't clobber profile
            defaults.

        Returns
        -------
        BenchmarkConfig

        Raises
        ------
        ValueError
            If the profile name is not found in the config file.
        """
        p = Path(config_file)
        if not p.exists():
            raise ValueError(f"Config file does not exist: {config_file}")
        with open(p) as fh:
            raw = toml.load(fh)

        profiles = raw.get("eval", {}).get("profiles", {})
        if profile_name not in profiles:
            available = sorted(profiles.keys()) if profiles else []
            raise ValueError(
                f"Unknown eval profile: {profile_name!r}. "
                f"Available profiles: {available}"
            )

        prof = dict(profiles[profile_name])

        # Map profile keys to BenchmarkConfig fields.
        kwargs: Dict[str, Any] = {
            "models": models,
            "config_file": config_file,
        }

        # Direct mappings (profile key == config field)
        _direct = [
            "dataset",
            "workflow_types",
            "recursion_limit",
            "structured_output",
            "judge_model",
            "judge_type",
            "max_queries",
            "resume",
        ]
        for key in _direct:
            if key in prof:
                kwargs[key] = prof[key]

        # Apply overrides (skip None values so unset CLI flags don't
        # clobber profile defaults).
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value

        return cls(**kwargs)

    @staticmethod
    def list_profiles(config_file: str) -> List[str]:
        """Return the names of all eval profiles defined in *config_file*.

        Parameters
        ----------
        config_file : str
            Path to a TOML config file.

        Returns
        -------
        list[str]
            Sorted list of profile names, e.g. ``["quick", "standard"]``.
        """
        p = Path(config_file)
        if not p.exists():
            return []
        with open(p) as fh:
            raw = toml.load(fh)
        return sorted(raw.get("eval", {}).get("profiles", {}).keys())
