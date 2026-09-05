"""Reject direct-URL dependencies before uploading ChemGraph distributions."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile

from packaging.requirements import Requirement


def check_distribution(path: Path) -> None:
    """Check built metadata, including extras, and the sdist add-on files."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError(f"{path}: expected exactly one wheel METADATA file")
            metadata = archive.read(names[0])
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            names = archive.getnames()
            roots = [n for n in names if n.count("/") == 1 and n.endswith("/PKG-INFO")]
            if len(roots) != 1:
                raise ValueError(f"{path}: expected exactly one root PKG-INFO file")
            root = roots[0].split("/")[0]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distributions", nargs="+", type=Path)
    for distribution in parser.parse_args().distributions:
        check_distribution(distribution)
        print(f"Metadata checked: {distribution}")
