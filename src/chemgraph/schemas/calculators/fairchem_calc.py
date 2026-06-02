from pydantic import BaseModel, Field, model_validator

from typing import Any, Optional, Dict
import torch
import logging

try:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
except ImportError:
    logging.warning("fairchem is not installed. .")
    FAIRChemCalculator = None
    pretrained_mlip = None


class FAIRChemCalc(BaseModel):
    """FAIRChem calculator configuration for ASE integration.

    Parameters
    ----------
    task_name : str, optional
        Task name (omol', 'omat', 'oc20', 'odac', or 'omc) for the prediction head.
        Must match available tasks in the model.
    seed : int, optional
        Seed for model reproducibility. Default is 42.
    multiplicity : int, optional
        Spin multiplicity (2S+1) of the system. Default is 1 (singlet).
        UMA/OMOL reads this from ``atoms.info["spin"]``; the schema field is named
        ``multiplicity`` for consistency with other calculators (TBLite, ORCA).
        The deprecated alias ``spin=`` is still accepted as input.
    charge : int, optional
        System charge. Default is 0.
    model_name: str
        Inference model name. Default is uma-s-1p1.
    device : str, optional
        Device to run inference on. Default is 'cuda' if available, otherwise 'cpu'.

    """

    calculator_type: str = Field(
        default="FAIRChem", description="Calculator identifier. Must be 'FAIRChem'."
    )
    task_name: Optional[str] = Field(
        default=None,
        description="Prediction task. Options are 'omol', 'omat', 'oc20', 'odac', or 'omc",
    )
    seed: int = Field(default=42, description="Random seed for inference reproducibility.")
    multiplicity: Optional[int] = Field(
        default=1,
        description=(
            "Spin multiplicity (2S+1) of the system. Default 1 (singlet). "
            "Passed to UMA via atoms.info['spin']."
        ),
        ge=1,
    )
    charge: Optional[int] = Field(default=0, description="Total system charge.")
    model_name: str = Field(
        default="uma-s-1p1", description="Model names. Options are 'uma-s-1p1' and 'uma-m-1'"
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Computation device to use, either 'cpu' or 'cuda'.",
    )
    inference_settings: str = Field(
        default="default", description="Settings for inference. Can be 'default' or 'turbo'"
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_spin_alias(cls, data: Any) -> Any:
        """Accept deprecated ``spin`` input as ``multiplicity``.

        Parameters
        ----------
        data : Any
            Raw calculator payload before Pydantic validation.

        Returns
        -------
        Any
            Payload with ``spin`` converted when applicable.
        """
        if isinstance(data, dict) and "spin" in data and "multiplicity" not in data:
            logging.warning(
                "FAIRChemCalc: field 'spin' is deprecated; use 'multiplicity' instead."
            )
            data["multiplicity"] = data.pop("spin")
        return data

    def get_calculator(self) -> Any:
        """Return a configured FAIRChemCalculator.

        Parameters
        ----------
        predict_unit : MLIPPredictUnit
            Pre-loaded MLIP model.

        Returns
        -------
        FAIRChemCalculator
            ASE-compatible calculator instance.
        """

        if pretrained_mlip is None or FAIRChemCalculator is None:
            raise ImportError("fairchem is not installed.")

        predict_unit = pretrained_mlip.get_predict_unit(
            model_name=self.model_name,
            inference_settings=self.inference_settings,
            device=self.device,
        )
        return FAIRChemCalculator(
            predict_unit=predict_unit,
            task_name=self.task_name,
            seed=self.seed,
        )

    def get_atoms_properties(self) -> Dict[str, Optional[int]]:
        """Return atom-level info keys to inject into atoms.info.

        UMA/OMOL reads spin multiplicity from ``atoms.info["spin"]``; we keep
        that key name here even though our schema field is ``multiplicity``.
        """
        return {
            "spin": self.multiplicity,
            "charge": self.charge,
        }

    def get_multiplicity(self) -> Optional[int]:
        """Return spin multiplicity (2S+1) for thermochemistry."""
        return self.multiplicity
