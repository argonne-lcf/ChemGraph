from pydantic import BaseModel, Field

from typing import Optional, Dict, Any
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
    spin : int, optional
        Spin multiplicity. Default is 1.
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
    spin: Optional[int] = Field(default=1, description="Total spin multiplicity of the system.")
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
        """Return atom-level info keys to inject into atoms.info."""
        return {
            "spin": self.spin,
            "charge": self.charge,
        }
