"""ASE adapter for validated, bounded structured ORCA evaluations."""

from pathlib import Path
import shlex

import numpy as np
from ase.calculators.genericfileio import GenericFileIOCalculator
from ase.calculators.orca import OrcaProfile, OrcaTemplate
from ase.io.orca import read_energy, read_orca_engrad
from ase.units import Hartree

from chemgraph.parsers.orca import validate_native
from chemgraph.utils.orca import identify_orca, resolve_orca, run_bounded


def generated_input(settings, driver):
    task = "EnGrad" if driver == "opt" else "SP"
    simple = f"RKS {settings.functional} {settings.basis} TightSCF NoAutoStart {task}"
    blocks = "%scf ConvForced 1 end\n%pal nprocs 1 end"
    if driver == "electronic":
        blocks += "\n%output Print[P_OrbEn] 2 end"
    return simple, blocks


class StructuredOrcaTemplate(OrcaTemplate):
    def __init__(self, config, driver, executable, version, budget):
        super().__init__()
        self.config, self.driver = config, driver
        self.executable, self.version, self.budget = executable, version, budget
        self.evaluations = 0
        self.evidence = None
        self.active_directory = None

    def write_input(self, profile, directory, atoms, parameters, properties):
        self.budget.remaining()
        if atoms.pbc.any() or not np.isfinite(atoms.positions).all():
            raise ValueError("ORCA evaluation requires finite nonperiodic coordinates")
        self.evaluations += 1
        # Every evaluation gets a fresh directory: old .out/.engrad/.gbw bytes
        # can never satisfy a subsequent force evaluation.
        self.active_directory = directory / f"evaluation-{self.evaluations:04d}"
        self.active_directory.mkdir(exist_ok=False)
        self.atoms = atoms.copy()
        super().write_input(
            profile, self.active_directory, atoms, parameters, properties
        )

    def execute(self, directory, profile):
        active = self.active_directory
        code = run_bounded(
            [self.executable, self.inputname],
            active,
            active / self.outputname,
            active / self.errorname,
            self.budget.remaining(),
        )
        if code:
            raise RuntimeError(
                f"ORCA exited with status {code}; see {active / self.errorname}"
            )

    def read_results(self, directory):
        active = self.active_directory
        text = (active / self.outputname).read_text()
        evidence = validate_native(
            text,
            charge=self.config.charge,
            multiplicity=1,
            symbols=self.atoms.get_chemical_symbols(),
            positions=self.atoms.positions,
            allow_gradient=self.driver == "opt",
        )
        with (active / self.outputname).open() as fd:
            energy = read_energy(fd.readlines())
        if (
            energy is None
            or not np.isfinite(energy)
            or abs(energy - evidence['energy']) > 1e-7
        ):
            raise ValueError("ASE and native total energies disagree")
        results = {"energy": energy, "free_energy": energy}
        if self.driver == "opt":
            path = active / "orca.engrad"
            values = [
                line.split()
                for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')
            ]
            n = len(self.atoms)
            if (
                len(values) != 2 + 4 * n
                or int(values[0][0]) != n
                or not np.isfinite(float(values[1][0]))
                or abs(float(values[1][0]) * Hartree - energy) > 1e-6
            ):
                raise ValueError(
                    "Gradient atom count/energy disagrees with this invocation"
                )
            # ORCA's documented Bohr conversion differs slightly from ASE's.
            # Use native units for the geometry cross-check; ASE reads forces.
            geometry = np.array(values[2 + 3 * n :], float)
            if (
                geometry.shape != (n, 4)
                or not np.array_equal(geometry[:, 0], self.atoms.numbers)
                or not np.allclose(
                    geometry[:, 1:] * 0.5291772083,
                    self.atoms.positions,
                    atol=1e-7,
                    rtol=0,
                )
            ):
                raise ValueError(
                    "Gradient geometry/order disagrees with this invocation"
                )
            forces = read_orca_engrad(path)
            if forces.shape != (n, 3) or not np.isfinite(forces).all():
                raise ValueError("Missing/nonfinite gradient")
            results['forces'] = forces
        self.evidence = evidence
        return results


def structured_calculator(config, driver, directory, budget):
    executable = resolve_orca()
    directory = Path(directory)
    version = identify_orca(executable, directory, budget)
    simple, blocks = generated_input(config.dft_settings, driver)
    template = StructuredOrcaTemplate(config, driver, executable, version, budget)
    return GenericFileIOCalculator(
        template=template,
        profile=OrcaProfile(command=shlex.quote(executable)),
        directory=directory,
        parameters=dict(
            charge=config.charge, mult=1, orcasimpleinput=simple, orcablocks=blocks
        ),
    )
