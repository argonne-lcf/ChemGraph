# Shared helpers for the subprocess plane-wave DFT calculators (Quantum
# ESPRESSO and VASP). Both launch an external executable via ASE and need the
# same periodicity-aware k-mesh handling, so the logic lives here once instead
# of being duplicated in each calculator schema.


def is_nonperiodic(atoms) -> bool:
    """Return True when ``atoms`` is a fully non-periodic (isolated) structure.

    An isolated molecule has ``pbc`` False on every axis, so it takes the
    Gamma-only / molecule branch in both plane-wave schemas. ``atoms=None``
    (construction / unit-test time, before any structure is read) is treated as
    *not* known-nonperiodic so callers fall back to their configured mesh.

    Parameters
    ----------
    atoms : ase.Atoms or None
        The structure, or None when no structure is available yet.

    Returns
    -------
    bool
        True only when ``atoms`` exists and no axis is periodic.
    """
    return atoms is not None and not atoms.pbc.any()


def mask_kmesh_by_pbc(kpts, pbc) -> tuple:
    """Collapse the k-mesh to a single point on every non-periodic axis.

    A Monkhorst-Pack mesh is only meaningful along a periodic direction; a slab
    (``pbc=[T,T,F]``) or wire (``pbc=[T,F,F]``) must use a single k-point in its
    vacuum direction(s). This keeps the user's requested subdivisions on the
    periodic axes and forces ``1`` on the rest -- unlike ASE's
    ``kptdensity2monkhorstpack`` it does NOT recompute the mesh from a density,
    so an explicit ``kpts`` is honored exactly as configured.

    Parameters
    ----------
    kpts : sequence of int
        The configured 3-axis Monkhorst-Pack subdivisions.
    pbc : sequence of bool
        ``atoms.pbc`` -- periodicity per axis.

    Returns
    -------
    tuple of int
        The per-axis masked mesh (``1`` on non-periodic axes).
    """
    return tuple(int(k) if p else 1 for k, p in zip(kpts, pbc))
