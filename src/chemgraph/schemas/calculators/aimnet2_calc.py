"""AIMNET2 foundation models parameters for ChemGraph
"""

from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel, Field


class AIMNET2Calc(BaseModel):
    """AIMNET2  calculator configuration.

    This class defines the configuration parameters for AIMNET2 machine learning models
    used in molecular simulations. It supports different calculator types including
    aimnet2 or a Path.

    Parameters
    ----------
    calculator_type : str, optional
        Type of calculator to use. Options: 'aimnet2' (default)
    model : str or Path, optional
        Name or path to the model file. If None, uses default model for selected calculator type.
    device : str, optional
    """

    calculator_type: str = Field(
        default="aimnet2",
        description="Type of calculator. Options: 'aimnet2' (default) ",
    )
    model: Optional[Union[str, Path]] = Field(
        default='aimnet2',
        description="Path to the model. If None, it will use the default model for the selected calculator type. "
        "Options: 'aimnet2' ",
    )

    def get_calculator(self):
        """Get the appropriate AIMNET2Calculator instance based on the selected calculator type.

        Returns
        -------
        AIMNET2Calc
            An instance of the appropriate AIMNET2 calculator

        Raises
        ------
        ValueError
            If an invalid calculator_type is specified
        """
        from aimnet2calc import AIMNet2ASE

        # Allow loading slice and AIMNET2 objects for compatibility with older model files

        # Force torch to disable weights_only loading (allows full pickle loads) for AIMNET2 models

        if self.calculator_type == "aimnet2":
            return AIMNet2ASE(self.model)
        else:
            raise ValueError(
                "Invalid calculator_type. Choose 'aimnet2' or path."
            )
