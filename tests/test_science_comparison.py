import pytest


def test_comparison_accepts_any_nonempty_conformer_collection():
    from chemgraph.utils.science import compare_conformers

    row = {
        "composition": "H2",
        "settings": {"model": "test"},
        "units": "eV",
        "potential_energy": 1.0,
        "converged": True,
    }
    assert compare_conformers([row])[0]["relative_energy_eV"] == 0
    assert len(compare_conformers([row] * 6)) == 6
    with pytest.raises(ValueError, match="At least one"):
        compare_conformers([])


@pytest.mark.parametrize(
    "key,value",
    [
        ("composition", "He"),
        ("settings", {"model": "another"}),
        ("units", "Hartree"),
        ("potential_energy", float("nan")),
        ("converged", None),
    ],
)
def test_comparison_rejects_inconsistent_or_unvalidated_results(key, value):
    from chemgraph.utils.science import compare_conformers

    row = {
        "composition": "H2",
        "settings": {"model": "test"},
        "units": "eV",
        "potential_energy": 1.0,
        "converged": True,
    }
    with pytest.raises(ValueError):
        compare_conformers([row, {**row, key: value}])


def test_nonconverged_energies_are_not_ranked():
    from chemgraph.utils.science import compare_conformers

    row = {
        "composition": "H2",
        "settings": {"model": "test"},
        "units": "eV",
        "potential_energy": 1.0,
        "converged": True,
    }
    report = compare_conformers(
        [{**row, "converged": False, "potential_energy": -100}, row]
    )
    assert report[0]["relative_energy_eV"] is None
    assert report[1]["relative_energy_eV"] == 0
