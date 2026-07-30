# Keywords and parameters obtained from
# https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html
# VASP parameters for ChemGraph. Periodic plane-wave DFT; same subprocess model
# as NWChem/ORCA (an external executable ASE launches), so the ChemGraph
# wall-clock cap applies at optimizer-step / displacement boundaries while an
# ASE optimizer drives the geometry. A single internal VASP relax (NSW>0) stays
# outside the cap, since it runs to completion inside one subprocess.

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from chemgraph.schemas.calculators._plane_wave import (
    is_nonperiodic,
    mask_kmesh_by_pbc,
)


class VaspCalc(BaseModel):
    """VASP periodic plane-wave DFT calculator configuration.

    VASP is an external executable that ASE launches as a subprocess (like
    NWChem and ORCA). It requires a licensed ``vasp_std``/``vasp_gam`` binary on
    ``PATH`` (or via ``ASE_VASP_COMMAND``) and a pseudopotential library pointed
    to by ``VASP_PP_PATH``. This machine has neither, so the class is written to
    be constructed and unit-tested (kwargs -> ASE ``Vasp``) without a real run.

    Parameters
    ----------
    calculator_type : str, optional
        Calculator type. Currently supports only 'vasp', by default 'vasp'.
    xc : str, optional
        Exchange-correlation functional, by default 'PBE'.
    encut : float, optional
        Plane-wave kinetic-energy cutoff in eV, by default 520.0.
    kpts : list of int, optional
        Monkhorst-Pack k-point mesh, by default [3, 3, 3]. Mutually exclusive
        with ``kspacing`` (set one; if both are given ``kspacing`` wins in VASP).
    kspacing : float, optional
        k-point spacing in 2*pi/Angstrom, VASP's own KSPACING convention (ASE
        writes this value straight to the INCAR KSPACING tag, so the same number
        means a denser mesh here than QE's Angstrom^-1 kspacing). An alternative
        to ``kpts``, by default None.
    ispin : int, optional
        Spin polarization: 1 = non-spin-polarized, 2 = spin-polarized, by
        default 1.
    nsw : int, optional
        Number of ionic steps for VASP's own internal relaxation. Leave None (or
        0) to let an ASE optimizer drive the geometry instead -- required for the
        ChemGraph wall-clock cap to see ionic-step boundaries, by default None.
    ediff : float, optional
        Electronic SCF convergence threshold in eV, by default None (VASP
        default).
    ediffg : float, optional
        Ionic relaxation convergence threshold; eV for energy or, if negative,
        eV/Angstrom for forces, by default None.
    prec : str, optional
        VASP precision mode ('Normal', 'Accurate', 'Single', ...), by default
        'Accurate'.
    setups : str or dict, optional
        Pseudopotential setup selection (e.g. 'recommended' or a per-element
        map), by default None.
    charge : int, optional
        Net charge of the cell (VASP ``nelect`` is derived from this by ASE), by
        default None.
    directory : str, optional
        Working directory for VASP I/O, by default '.'.
    gamma : bool, optional
        Use a Gamma-centered k-mesh; the default (False) uses Monkhorst-Pack.
        Forced True for a non-periodic molecule (single Gamma point).
    vacuum : float, optional
        Padding in Angstrom added around a non-periodic molecule that has no
        cell, so plane-wave VASP has a finite box to run in. Applied by the
        ChemGraph runner (``ase_core``) via ``atoms.center(vacuum=...)`` before
        the run; ignored for structures that already carry a cell, by default
        6.0.
    input_data : dict, optional
        Extra raw INCAR tags merged into the ASE ``Vasp`` kwargs (advanced
        escape hatch, the VASP analogue of ``EspressoCalc.input_data``): e.g.
        ``{'ismear': 0, 'sigma': 0.05, 'algo': 'Fast', 'lreal': 'Auto'}``. Keys
        are lowercased before merging: ASE routes a canonical lowercase name to
        its typed parameter slot, but a raw uppercase key lands in a stray slot
        and, on a conflict with a convenience field, writes a *duplicate* INCAR
        tag (e.g. both ``ISMEAR = 0`` and ``ISMEAR = 1``). Lowercasing collapses
        that so ``{'ISMEAR': 0}`` cleanly overrides. Merged **last**, so these
        win over the convenience fields (and even the molecule k-mesh / molecule
        smearing) on conflict. Unlike QE, VASP keys are case-normalized here;
        ``EspressoCalc.input_data`` passes its namelist keys through untouched.
        ASE does not validate these keys, so an unrecognized tag (a typo such as
        ``ismaer``) is accepted silently and never reaches INCAR; check spelling
        against the VASP tag list. By default None.
    """

    calculator_type: str = Field(
        default="vasp",
        description="Calculator type. Currently supports only 'vasp'.",
    )
    xc: str = Field(
        default="PBE", description="Exchange-correlation functional."
    )
    encut: float = Field(
        default=520.0, description="Plane-wave kinetic-energy cutoff in eV."
    )
    kpts: Optional[List[int]] = Field(
        default=[3, 3, 3],
        min_length=3,
        max_length=3,
        description="Monkhorst-Pack k-point mesh (exactly 3 axes). Mutually exclusive with kspacing.",
    )
    kspacing: Optional[float] = Field(
        default=None,
        description=(
            "k-point spacing in 2*pi/Angstrom (VASP KSPACING convention, denser "
            "than QE's Angstrom^-1 kspacing for the same value); alternative to kpts."
        ),
    )
    ispin: int = Field(
        default=1,
        description="Spin polarization: 1 = non-spin-polarized, 2 = spin-polarized.",
        ge=1,
        le=2,
    )
    nsw: Optional[int] = Field(
        default=None,
        description=(
            "Ionic steps for VASP's internal relaxation. Leave None/0 to let an "
            "ASE optimizer drive the geometry (needed for the wall-clock cap)."
        ),
    )
    ediff: Optional[float] = Field(
        default=None, description="Electronic SCF convergence threshold in eV."
    )
    ediffg: Optional[float] = Field(
        default=None,
        description=(
            "Ionic convergence threshold; eV for energy, or eV/Angstrom (forces) "
            "if negative."
        ),
    )
    prec: str = Field(
        default="Accurate",
        description="VASP precision mode ('Normal', 'Accurate', 'Single', ...).",
    )
    setups: Optional[Union[str, dict]] = Field(
        default=None,
        description="Pseudopotential setup selection (e.g. 'recommended' or a per-element map).",
    )
    charge: Optional[int] = Field(
        default=None, description="Net charge of the cell."
    )
    directory: str = Field(
        default=".", description="Working directory for VASP calculations."
    )
    gamma: bool = Field(
        default=False,
        description="Use a Gamma-centered k-mesh (forced True for a non-periodic molecule).",
    )
    vacuum: float = Field(
        default=6.0,
        description=(
            "Padding (Angstrom) added around a cell-less non-periodic molecule "
            "so plane-wave VASP has a finite box; applied by ase_core."
        ),
    )
    input_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Extra raw INCAR tags (keys lowercased) merged last into the ASE "
            "Vasp kwargs (advanced escape hatch mirroring EspressoCalc.input_data; "
            "overrides convenience fields on conflict). ASE does not validate "
            "these keys, so a misspelled tag is accepted silently and dropped."
        ),
    )

    def get_calculator(self, atoms=None):
        """Get an ASE-compatible VASP calculator instance.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            The structure the calculator will run on. When provided and fully
            non-periodic (``atoms.pbc.any()`` is False), the k-mesh is pinned to
            a single Gamma point (``kpts=[1,1,1]``, ``gamma=True``), any
            ``kspacing`` is dropped -- a real VASP run still needs a KPOINTS
            file (unlike QE's gamma shortcut), and an INCAR ``KSPACING`` would
            silently override that single Gamma point -- and ``ismear=0`` with a
            small ``sigma`` is defaulted so an isolated molecule does not inherit
            VASP's metallic default (``ISMEAR=1``, which gives spurious partial
            occupancies for discrete molecular levels). For a partially periodic
            structure (slab ``[T,T,F]`` / wire ``[T,F,F]``) the configured mesh
            is masked per axis so the vacuum directions collapse to a single
            k-point. When None (construction / unit-test time), the configured
            mesh is kept unchanged for backward compatibility.

        Returns
        -------
        Vasp
            An ASE-compatible VASP calculator instance.

        Raises
        ------
        ValueError
            If an invalid calculator_type is specified.
        """
        if self.calculator_type != "vasp":
            raise ValueError(
                "Invalid calculator_type. The only valid option is 'vasp'."
            )

        from ase.calculators.vasp import Vasp

        # Only pass keys the user actually set, so ASE/VASP defaults stand in for
        # the rest. kpts is dropped when kspacing is given (VASP uses kspacing).
        kwargs: dict = dict(
            xc=self.xc,
            encut=self.encut,
            ispin=self.ispin,
            prec=self.prec,
            directory=self.directory,
        )
        nonperiodic = is_nonperiodic(atoms)
        if nonperiodic:
            # A single Gamma point for the isolated molecule. Pin kpts (real VASP
            # needs a KPOINTS file -- None writes none) and force gamma; do NOT
            # pass kspacing (an INCAR KSPACING silently overrides KPOINTS).
            kwargs["kpts"] = [1, 1, 1]
            kwargs["gamma"] = True
            # A discrete molecule has sharp, well-separated levels, so use the
            # Gaussian smearing that suits an insulator/molecule. Without this
            # VASP falls back to its metallic default (ISMEAR=1, Methfessel-
            # Paxton), which produces spurious partial/negative occupancies for
            # a molecule. Overridable via input_data (merged last, below).
            kwargs["ismear"] = 0
            kwargs["sigma"] = 0.03
        else:
            # Periodic (or atoms=None at construction time). Keep the configured
            # mesh, but for a periodic structure mask it per axis so a slab/wire
            # uses a single k-point in its vacuum direction(s); atoms=None keeps
            # the mesh verbatim for backward compatibility.
            if self.kspacing is not None:
                kwargs["kspacing"] = self.kspacing
            elif self.kpts is not None:
                if atoms is None:
                    kwargs["kpts"] = self.kpts
                else:
                    kwargs["kpts"] = list(
                        mask_kmesh_by_pbc(self.kpts, atoms.pbc)
                    )
            if self.gamma:
                kwargs["gamma"] = self.gamma
        if self.nsw is not None:
            kwargs["nsw"] = self.nsw
        if self.ediff is not None:
            kwargs["ediff"] = self.ediff
        if self.ediffg is not None:
            kwargs["ediffg"] = self.ediffg
        if self.setups is not None:
            kwargs["setups"] = self.setups
        if self.charge is not None:
            kwargs["charge"] = self.charge

        # Advanced escape hatch: raw INCAR tags win over the convenience fields
        # above (and over the molecule k-mesh / smearing) on any key collision.
        # Keys are lowercased because ASE routes a canonical lowercase name to
        # its typed parameter slot; a raw uppercase key lands in a stray slot and
        # (on a collision with a convenience field we set above) writes a
        # DUPLICATE INCAR tag -- e.g. both "ISMEAR = 0" and "ISMEAR = 1".
        # Lowercasing collapses the two so the escape-hatch value cleanly wins.
        if self.input_data:
            kwargs.update(
                {k.lower(): v for k, v in self.input_data.items()}
            )

        return Vasp(**kwargs)

    def get_multiplicity(self) -> Optional[int]:
        """Return spin multiplicity for thermochemistry.

        VASP expresses spin through ``ispin``/``nupdown``, so it reports no
        2S+1 molecular multiplicity.
        """
        return None
