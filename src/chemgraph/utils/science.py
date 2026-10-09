"""Scientific comparability and convergence checks for conformer collections."""

import math


def compare_conformers(results):
    if not results:
        raise ValueError("At least one conformer result is required")
    reference = results[0]
    for item in results:
        if (
            item.get("units") != "eV"
            or not item.get("composition")
            or not item.get("settings")
        ):
            raise ValueError("Composition, settings and eV units are required")
        if (
            item["composition"] != reference["composition"]
            or item["settings"] != reference["settings"]
        ):
            raise ValueError(
                "Only consistent composition and scientific settings may be compared"
            )
        if not math.isfinite(item["potential_energy"]):
            raise ValueError("All energies must be finite")
        if not isinstance(item.get("converged"), bool):
            raise ValueError(
                "Convergence must be recorded separately from execution success"
            )
    valid = [item["potential_energy"] for item in results if item["converged"]]
    minimum = min(valid) if valid else None
    return [
        {
            **item,
            "relative_energy_eV": (
                item["potential_energy"] - minimum
                if item["converged"] and minimum is not None
                else None
            ),
            "validation": "converged" if item["converged"] else "nonconverged",
        }
        for item in results
    ]
