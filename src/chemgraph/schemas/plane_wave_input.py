"""Calculator-pinned input schemas for the plane-wave DFT tools.

``run_ase`` accepts any installed calculator through the ``CalculatorUnion`` on
:class:`~chemgraph.schemas.ase_input.ASEInputSchema`, so its whole tool schema
(every calculator's fields) is serialized into every LLM request. The dedicated
``run_qe`` / ``run_vasp`` tools instead each pin a single engine, so the model
sees only that engine's parameters. These subclasses reuse every common ASE
input field (``input_structure_file``, ``driver``, ``optimizer``, ``fmax``,
``steps`` ...) from :class:`ASEInputSchema` and override only ``calculator`` to
the concrete plane-wave type.

The parent's ``_validate_calculator_type`` coerces + availability-gates the
calculator against ``available_calculator_classes`` (engines detected at import
time). That gate is right for ``run_ase`` (the union only offers installed
engines) but wrong here: these schemas are also constructed in hermetic tests
and by ``tool_call_eval`` on machines with no pw.x / VASP binary. So each
subclass replaces the validator with a narrow coercion that instantiates its own
calculator type WITHOUT the availability probe. Registration of the *tools*
themselves still gates on availability (see ``graphs/single_agent.py``), so an
engine that is not installed is never offered to the model on a real run.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.calculators.espresso_calc import EspressoCalc
from chemgraph.schemas.calculators.vasp_calc import VaspCalc


def _coerce_pinned_calculator(data: Any, calc_cls: type, canonical_type: str) -> Any:
    """Coerce ``data['calculator']`` into ``calc_cls`` without the availability gate.

    Mirrors the accept-dict-or-instance behavior of
    :func:`chemgraph.schemas.ase_input._coerce_calculator_payload`, but pins the
    single calculator type and skips the installed-engine check so the schema is
    constructible in hermetic tests / eval on a host lacking the binary.

    Parameters
    ----------
    data : Any
        Raw payload before Pydantic validation (dict when it carries a
        ``calculator`` key; passed through unchanged otherwise).
    calc_cls : type
        The concrete calculator model to instantiate (``EspressoCalc`` /
        ``VaspCalc``).
    canonical_type : str
        The canonical ``calculator_type`` string forced onto the result so it
        routes correctly in ``ase_core`` dispatch even if the caller passed an
        alias (``qe``, ``pw``, ...).

    Returns
    -------
    Any
        The payload with ``calculator`` set to a ``calc_cls`` instance.
    """
    if not isinstance(data, dict):
        return data

    calc = data.get("calculator")
    if calc is None:
        data["calculator"] = calc_cls()
    elif isinstance(calc, dict):
        init_args = {k: v for k, v in calc.items() if k != "calculator_type"}
        data["calculator"] = calc_cls(**init_args)
    elif not isinstance(calc, calc_cls):
        # A pre-built instance of the wrong type is a caller error worth surfacing.
        raise ValueError(
            f"{calc_cls.__name__} tool received a "
            f"{type(calc).__name__} calculator; expected {calc_cls.__name__}."
        )
    # Normalize the type tag so ase_core dispatch always sees the canonical name.
    data["calculator"].calculator_type = canonical_type
    return data


class QEInputSchema(ASEInputSchema):
    """ASE input pinned to the Quantum ESPRESSO (pw.x) calculator.

    Identical to :class:`ASEInputSchema` for every non-calculator field; the
    ``calculator`` is fixed to :class:`EspressoCalc` so ``run_qe`` exposes only
    QE parameters to the model.
    """

    calculator: EspressoCalc = Field(
        default_factory=EspressoCalc,
        description=(
            "Quantum ESPRESSO (pw.x) plane-wave DFT configuration: plane-wave "
            "cutoffs, k-point mesh, pseudopotentials, smearing, etc."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_calculator_type(cls, data: Any):
        return _coerce_pinned_calculator(data, EspressoCalc, "espresso")


class VaspInputSchema(ASEInputSchema):
    """ASE input pinned to the VASP calculator.

    Identical to :class:`ASEInputSchema` for every non-calculator field; the
    ``calculator`` is fixed to :class:`VaspCalc` so ``run_vasp`` exposes only
    VASP parameters to the model.
    """

    calculator: VaspCalc = Field(
        default_factory=VaspCalc,
        description=(
            "VASP plane-wave DFT configuration: ENCUT, k-point mesh, spin, "
            "smearing, INCAR overrides, etc."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_calculator_type(cls, data: Any):
        return _coerce_pinned_calculator(data, VaspCalc, "vasp")
