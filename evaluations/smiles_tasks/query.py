def get_query(
    smiles: str,
    query_name: str = "atomsdata",  # options: atomsdata, opt, vib
    method: str = "mace_mp",
) -> str:
    """Get query for a SMILES-related task for CompChemAgent

    Args:
        smiles (str): SMILES string.
        query_name (str, optional): Type of query. Defaults to "atomsdata". Options: "atomsdata", "opt", "vib", "opt_method" and "vib_method".
        method (str, optional): The method/level of theory for CompChemAgent to run simulation. Defaults to "mace_mp".

    Returns:
        str: formatted query.
    """
    query_dict = {
        "atomsdata": f"What is the coordinate of {smiles}?",
        "opt_method": f"Perform geometry optimization for {smiles} using {method}",
        "vib_method": f"Run vibrational frequency calculation for {smiles} using {method}",
    }

    return query_dict.get(query_name, "Query not found")  # Returns the query or a default message
