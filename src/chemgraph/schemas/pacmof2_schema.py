# PACMOF2 predicts partial atomic charges for MOFs using ML (scikit-learn).
# It is a standalone CIF-in/CIF-out engine (not an ASE calculator), so the
# schemas mirror the gRASPA schemas but drop the temperature/pressure axis.
from typing import Union

from pydantic import BaseModel, Field


class pacmof2_input_schema(BaseModel):
    input_structure_file: str = Field(
        description="Path to the input CIF file to assign partial charges to."
    )
    identifier: str = Field(
        default="_pacmof",
        description=(
            "Suffix for the output CIF filename, written as "
            "'{stem}{identifier}.cif' next to the input CIF."
        ),
    )
    adjust_charge_method: str = Field(
        default="mean",
        description=(
            "How residual charge is redistributed so the framework meets the "
            "target net charge. Either 'mean' or 'magnitude'."
        ),
    )
    net_charge: Union[int, float, dict] = Field(
        default=0,
        description=(
            "Target net charge of the framework. Use 0 for neutral MOFs, an "
            "int/float for a charged framework, or a dict mapping per-site "
            "charges for ionic MOFs."
        ),
    )


class pacmof2_input_schema_ensemble(BaseModel):
    input_structures: Union[str, list[str]] = Field(
        default="",
        description=(
            "Path to a directory of CIF files OR a specific list of file paths. "
            "Required unless remote_structure_directory is provided."
        ),
    )
    remote_structure_directory: str | None = Field(
        default=None,
        description=(
            "Path to pre-staged CIF files on the remote HPC filesystem. "
            "When provided, workers read structures directly from this path. "
            "Use the transfer_files tool to stage files first."
        ),
    )
    identifier: str = Field(
        default="_pacmof",
        description="Suffix for each output CIF filename ('{stem}{identifier}.cif').",
    )
    adjust_charge_method: str = Field(
        default="mean",
        description="Charge adjustment method applied to every structure ('mean' or 'magnitude').",
    )
    net_charge: Union[int, float, dict] = Field(
        default=0,
        description="Target net charge applied to every structure in the batch.",
    )
