# Prepare an input structure

Use or prepare a coordinate file for the requested calculation. Start from the
supplied structure or molecular identity; preserve the user's geometry and
scientific choices.

## Choose the input route

- **Existing file:** pass the supplied path directly when the selected calculation
  accepts its format and can access it. Use `file_to_atomsdata` only when the task
  requires inspection, conversion, missing information from the contents, or
  diagnosis of a read error. Preserve the supplied file unless a change is requested.
- **Explicit coordinates:** confirm their units and save them without generating
  a replacement geometry. XYZ is suitable for an isolated molecule; periodic
  inputs also need their cell and periodicity preserved.
- **SMILES:** use `smiles_to_coordinate_file` with the supplied SMILES and an
  absolute host `output_file` path. Check `ok`, `path` and `natoms`, then use
  the returned file. This generates a molecular
  starting geometry with explicit hydrogens and RDKit/UFF preparation; it does
  not establish convergence under the requested ASE calculator.
- **Molecule name:** use `molecule_name_to_smiles` when identity needs lookup.
  It uses PubChem and requires network access. For a specified stereoisomer,
  set `include_stereochemistry=True`; the default omits stereochemistry.

For script-based preparation when native tools are unavailable, reuse
`chemgraph.tools.cheminformatics_core.smiles_to_coordinate_file_core` with the
SMILES and absolute output path. Report missing dependencies.

## Coordinate sources

- Reuse user-provided structure files directly. Preserve their coordinates
  unless the user requests a transformation.
- When the user supplies explicit coordinates, write those values faithfully.
  Do not fill in missing atoms or positions by guessing.
- When coordinates need generation, use ChemGraph's structure-generation tools.
  Use the file returned by a successful tool call; do not manually compose
  coordinates from a molecule's name, formula, or SMILES.
- Pass the resulting file path to downstream calculations. Avoid retyping
  coordinates into scripts or reconstructing files from conversation text.
- If the input is incomplete or generation fails, report the missing information
  or failure. Do not fabricate a replacement structure.

## Hand off the artifact

Report the artifact path, any checks performed and unresolved choices; include
composition when already known or requested. For batch input construction,
continue with [ASE calculations](ase-calculations.md).

Carry the requested charge and multiplicity into the calculation settings;
XYZ coordinates alone do not record them.
