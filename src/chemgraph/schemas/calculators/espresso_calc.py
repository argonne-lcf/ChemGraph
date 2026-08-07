# Keywords and parameters obtained from
# https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html
# Quantum ESPRESSO (pw.x) parameters for ChemGraph. Periodic plane-wave DFT;
# same subprocess model as NWChem/ORCA/VASP (an external executable ASE
# launches), so the ChemGraph wall-clock cap applies at optimizer-step /
# displacement boundaries while an ASE optimizer drives the geometry. A single
# internal pw.x 'relax'/'vc-relax' run stays outside the cap, since it runs to
# completion inside one subprocess.

import shlex
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from chemgraph.schemas.calculators._plane_wave import mask_kmesh_by_pbc

# Tokens that mark the start of ASE's own input/redirection handling. A legacy
# ASE_ESPRESSO_COMMAND may bake in the full pw.x command line
# ("mpirun -np 4 pw.x -in PREFIX.pwi > PREFIX.pwo"), but the modern
# EspressoProfile only wants the executable (+ any launcher/runtime flags) and
# appends "-in <file>" itself. Anything from the first of these tokens onward is
# ASE's job, so it must be stripped to avoid a duplicated "-in", a literal ">"
# token, and the wrong input file on a real run.
_ESPRESSO_CMD_STOP_TOKENS = frozenset(
    {"-in", "-inp", "-input", "-i", "<", ">", ">>", "|", "&>", "1>", "2>"}
)


def _sanitize_espresso_command(raw: str) -> str:
    """Reduce a (possibly legacy) pw.x command line to just its launch prefix.

    Keeps the executable and any launcher/runtime flags (e.g. ``mpirun -np 4``,
    ``srun``, ``-npool 4``) but drops ASE-owned input/redirection tokens and the
    input-file operand, so the result is safe to hand to
    ``EspressoProfile(command=...)`` (which appends ``-in <file>`` itself).

    Parameters
    ----------
    raw : str
        The raw command string, typically from ``$ASE_ESPRESSO_COMMAND``.

    Returns
    -------
    str
        The sanitized launch prefix. Falls back to ``raw`` (stripped) when the
        string cannot be parsed or nothing would remain.
    """
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return raw.strip()

    kept: List[str] = []
    for tok in tokens:
        if tok in _ESPRESSO_CMD_STOP_TOKENS or tok.endswith((".pwi", ".pwo")):
            break
        kept.append(tok)

    if not kept:
        return raw.strip()
    return shlex.join(kept)


class EspressoCalc(BaseModel):
    """Quantum ESPRESSO (pw.x) periodic plane-wave DFT calculator configuration.

    Quantum ESPRESSO is an external executable that ASE launches as a subprocess
    (like NWChem, ORCA, and VASP). It requires ``pw.x`` on ``PATH`` (or via
    ``ASE_ESPRESSO_COMMAND``) and a pseudopotential directory (``ESPRESSO_PSEUDO``
    or the ``pseudo_dir`` field). This machine has neither, so the class is
    written to be constructed and unit-tested (kwargs -> ASE ``Espresso``)
    without a real run.

    Parameters
    ----------
    calculator_type : str, optional
        Calculator type. Currently supports only 'espresso', by default
        'espresso'. Aliases 'qe', 'pwscf', 'quantum-espresso' resolve to this.
    pseudopotentials : dict, optional
        Per-element pseudopotential filename map, e.g. {'Si': 'Si.UPF'}. Required
        for a real run; may be omitted when only constructing the calculator, by
        default None.
    pseudo_dir : str, optional
        Directory holding the .UPF pseudopotential files. Falls back to the
        ESPRESSO_PSEUDO environment variable when None, by default None.
    ecutwfc : float, optional
        Plane-wave wavefunction cutoff in Ry, by default 50.0.
    ecutrho : float, optional
        Charge-density cutoff in Ry; defaults to QE's 4*ecutwfc when None, by
        default None.
    kpts : list of int, optional
        Monkhorst-Pack k-point mesh, by default [4, 4, 4]. Mutually exclusive
        with ``kspacing``.
    kspacing : float, optional
        k-point spacing in 1/Angstrom; an alternative to ``kpts``, by default
        None.
    xc : str, optional
        Exchange-correlation functional written to the pw.x ``input_dft``
        keyword, by default 'PBE'.
    smearing : str, optional
        Occupation smearing type for metals (e.g. 'gaussian', 'mp', 'mv'); None
        uses fixed occupations, by default None.
    degauss : float, optional
        Smearing width in Ry (used with ``smearing``), by default None.
    nspin : int, optional
        1 = non-spin-polarized, 2 = spin-polarized, by default 1.
    tot_charge : float, optional
        Net charge of the cell, by default None.
    input_data : dict, optional
        Extra raw pw.x namelist parameters merged into the input (advanced;
        overrides the convenience fields above where they overlap), by default
        None.
    directory : str, optional
        Working directory for pw.x I/O, by default '.'.
    vacuum : float, optional
        Padding in Angstrom added around a non-periodic molecule that has no
        cell, so plane-wave pw.x has a finite box to run in. Applied by the
        ChemGraph runner (``ase_core``) via ``atoms.center(vacuum=...)`` before
        the run; ignored for structures that already carry a cell, by default
        6.0. For a *charged* isolated molecule also set
        ``input_data={'assume_isolated': 'mt'}`` (Martyna-Tuckerman) so the
        periodic images do not interact spuriously.
    """

    model_config = {"arbitrary_types_allowed": True}

    calculator_type: str = Field(
        default="espresso",
        description="Calculator type. Supports 'espresso' (aliases: qe, pwscf, quantum-espresso).",
    )
    pseudopotentials: Optional[Dict[str, str]] = Field(
        default=None,
        description="Per-element pseudopotential filename map, e.g. {'Si': 'Si.UPF'}.",
    )
    pseudo_dir: Optional[str] = Field(
        default=None,
        description="Directory of .UPF files; falls back to $ESPRESSO_PSEUDO when None.",
    )
    ecutwfc: float = Field(
        default=50.0, description="Plane-wave wavefunction cutoff in Ry."
    )
    ecutrho: Optional[float] = Field(
        default=None,
        description="Charge-density cutoff in Ry; QE default is 4*ecutwfc when None.",
    )
    kpts: Optional[List[int]] = Field(
        default=[4, 4, 4],
        min_length=3,
        max_length=3,
        description="Monkhorst-Pack k-point mesh (exactly 3 axes). Mutually exclusive with kspacing.",
    )
    kspacing: Optional[float] = Field(
        default=None,
        description="k-point spacing in 1/Angstrom; alternative to kpts.",
    )
    xc: str = Field(
        default="PBE",
        description="Exchange-correlation functional (pw.x input_dft keyword).",
    )
    smearing: Optional[str] = Field(
        default=None,
        description="Occupation smearing for metals ('gaussian', 'mp', 'mv'); None = fixed.",
    )
    degauss: Optional[float] = Field(
        default=None, description="Smearing width in Ry (used with smearing)."
    )
    nspin: int = Field(
        default=1,
        description="1 = non-spin-polarized, 2 = spin-polarized.",
        ge=1,
        le=2,
    )
    tot_charge: Optional[float] = Field(
        default=None, description="Net charge of the cell."
    )
    input_data: Optional[Dict[str, Union[str, int, float, bool, dict]]] = Field(
        default=None,
        description="Extra raw pw.x namelist parameters merged into the input (advanced).",
    )
    directory: str = Field(
        default=".", description="Working directory for Quantum ESPRESSO calculations."
    )
    vacuum: float = Field(
        default=6.0,
        description=(
            "Padding (Angstrom) added around a cell-less non-periodic molecule "
            "so plane-wave pw.x has a finite box; applied by ase_core."
        ),
    )

    def _build_input_data(self) -> dict:
        """Assemble the pw.x ``input_data`` namelist from the convenience fields.

        Returns
        -------
        dict
            A flat pw.x parameter dict; ASE routes keys into their namelists.
            Any keys in ``self.input_data`` override the convenience defaults.
        """
        data: dict = {
            "ecutwfc": self.ecutwfc,
            "input_dft": self.xc,
            "nspin": self.nspin,
            # ChemGraph drives geometry with an ASE optimizer, so pw.x is used as
            # a force engine and must print forces after every SCF. ASE 3.29's
            # espresso writer does NOT translate properties=['forces'] into the
            # pw.x 'tprnfor' control flag, so without this the first optimizer
            # step raises PropertyNotImplementedError('forces not present'). Set
            # it explicitly; overridable via the input_data field below.
            "tprnfor": True,
        }
        if self.ecutrho is not None:
            data["ecutrho"] = self.ecutrho
        if self.smearing is not None:
            data["occupations"] = "smearing"
            data["smearing"] = self.smearing
            if self.degauss is not None:
                data["degauss"] = self.degauss
        if self.tot_charge is not None:
            data["tot_charge"] = self.tot_charge
        if self.input_data:
            data.update(self.input_data)
        return data

    def get_calculator(self, atoms=None):
        """Get an ASE-compatible Quantum ESPRESSO calculator instance.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            The structure the calculator will run on. When provided and fully
            non-periodic (``atoms.pbc.any()`` is False), the k-point mesh is
            dropped so ASE writes ``K_POINTS gamma`` -- a Monkhorst-Pack mesh is
            meaningless for an isolated molecule. For a partially periodic
            structure (slab ``[T,T,F]`` / wire ``[T,F,F]``) the configured mesh
            is masked per axis so the non-periodic (vacuum) directions collapse
            to a single k-point. When None (construction / unit-test time), the
            configured ``kpts``/``kspacing`` is kept unchanged for backward
            compatibility.

        Returns
        -------
        Espresso
            An ASE-compatible Espresso calculator instance.

        Raises
        ------
        ValueError
            If an invalid calculator_type is specified.
        """
        if self.calculator_type != "espresso":
            raise ValueError(
                "Invalid calculator_type. The only valid option is 'espresso'."
            )

        import os

        from ase.calculators.espresso import Espresso, EspressoProfile

        pseudo_dir = self.pseudo_dir or os.environ.get("ESPRESSO_PSEUDO")
        # EspressoProfile.command must be the executable (or an mpi wrapper
        # around it), e.g. "pw.x" or "mpirun -np 4 pw.x". ASE appends
        # "-in <input>" itself, so the legacy full-command-line form of
        # ASE_ESPRESSO_COMMAND (with "-in PREFIX.pwi > PREFIX.pwo") is sanitized
        # down to just the launch prefix -- otherwise a real run gets a
        # duplicated "-in", a literal ">" token, and the wrong input file.
        profile = EspressoProfile(
            command=_sanitize_espresso_command(
                os.environ.get("ASE_ESPRESSO_COMMAND", "pw.x")
            ),
            pseudo_dir=pseudo_dir or ".",
        )

        kwargs: dict = dict(
            profile=profile,
            input_data=self._build_input_data(),
            pseudopotentials=self.pseudopotentials or {},
            directory=self.directory,
        )
        # k-mesh follows atoms.pbc:
        #   * fully non-periodic molecule -> pass neither kpts nor kspacing, so
        #     ASE emits "K_POINTS gamma" (a Monkhorst-Pack mesh is meaningless);
        #   * slab/wire -> mask the configured mesh per axis so vacuum
        #     directions collapse to a single k-point (a mesh there is wasted
        #     and unphysical);
        #   * fully periodic solid -> keep the configured mesh verbatim.
        # kspacing is left to ASE, whose own mesh generation is already
        # pbc-aware. atoms=None (construction time) keeps the configured mesh so
        # existing no-arg callers are unaffected.
        if atoms is None:
            if self.kspacing is not None:
                kwargs["kspacing"] = self.kspacing
            elif self.kpts is not None:
                kwargs["kpts"] = tuple(self.kpts)
        elif atoms.pbc.any():
            if self.kspacing is not None:
                kwargs["kspacing"] = self.kspacing
            elif self.kpts is not None:
                kwargs["kpts"] = mask_kmesh_by_pbc(self.kpts, atoms.pbc)

        return Espresso(**kwargs)

    def get_multiplicity(self) -> Optional[int]:
        """Return spin multiplicity for thermochemistry.

        Quantum ESPRESSO expresses spin through ``nspin``/``tot_magnetization``,
        so it reports no 2S+1 molecular multiplicity.
        """
        return None
