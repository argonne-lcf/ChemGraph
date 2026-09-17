"""Reject direct-URL dependencies before uploading ChemGraph distributions."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile

from packaging.requirements import Requirement

GRASPA_ASSETS = (
    "simulation.input", "H2O.def", "force_field.def",
    "force_field_mixing_rules.def", "pseudo_atoms.def",
)


def check_distribution(path: Path) -> None:
    """Check built metadata, including extras, and the sdist add-on files."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            contents = archive.namelist()
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError(f"{path}: expected exactly one wheel METADATA file")
            metadata = archive.read(names[0])
            package_prefix = "chemgraph/"
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            names = archive.getnames()
            roots = [n for n in names if n.count("/") == 1 and n.endswith("/PKG-INFO")]
            if len(roots) != 1:
                raise ValueError(f"{path}: expected exactly one root PKG-INFO file")
            root = roots[0].split("/")[0]
            contents = names
            package_prefix = f"{root}/src/chemgraph/"
            for filename in ("mace-polar.txt", "ocsr-models.txt"):
                if f"{root}/requirements/{filename}" not in names:
                    raise ValueError(f"{path}: missing requirements/{filename}")
            with archive.extractfile(roots[0]) as source:
                metadata = source.read()
    else:
        raise ValueError(f"Unsupported distribution: {path}")

    requirements = BytesParser().parsebytes(metadata).get_all("Requires-Dist", [])
    for raw in requirements:
        if Requirement(raw).url is not None:
            raise ValueError(f"{path}: direct-URL dependency is not publishable: {raw}")
    for asset in GRASPA_ASSETS:
        name = f"{package_prefix}tools/files/template_graspa_sycl/{asset}"
        if name not in contents:
            raise ValueError(f"{path}: missing gRASPA asset {asset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distributions", nargs="+", type=Path)
    for distribution in parser.parse_args().distributions:
        check_distribution(distribution)
        print(f"Metadata checked: {distribution}")
