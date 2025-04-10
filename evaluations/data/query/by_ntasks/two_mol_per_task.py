def get_query(
    mol1_str: str,
    mol2_str: str,
    query_index: int = None,
    query_name: str = None,
    method: str = "mace_mp",
    temperature: float = 298,
    pressure: float = 1.01325,
) -> str:
    """Get query for an evaluation task for CompChemAgent.

    Args:
        mol1_str (str): molecule name, SMILES string or file path.
        mol2_str (str): molecule name, SMILES string or file path.
        query_index (int, optional): Index of the query type.
        query_name (str, optional): Name of the query type (e.g., "atomsdata", "opt_method").
        method (str, optional): Method/level of theory. Defaults to "mace_mp".
        temperature (float, optional): Temperature for thermodynamic calculations. Defaults to 298.
        pressure (float, optional): Pressure for thermodynamic calculations. Defaults to 1.01325.

    Returns:
        str: Formatted query string.
    """

    query_dict = {
        "smile_to_coord": {
            "query": lambda: f"Provide the XYZ coordinates corresponding to these SMILES strings: {mol1_str} and {mol2_str}",
            "description": "Get atomic coordinates from two SMILES strings",
            "n_tasks": 2,
        },
        "name_to_coord": {
            "query": lambda: f"Provide the XYZ coordinates corresponding to these molecules: {mol1_str} and {mol2_str}",
            "description": "Get atomic coordinates from a molecule name",
            "n_tasks": 4,
        },
        "smile_to_opt_coord": {
            "query": lambda: f"Perform geometry optimization for these two SMILES strings {mol1_str} and {mol2_str} using {method}",
            "description": "Get optimized atomic coordinates from a SMILES string",
            "n_tasks": 4,
        },
        "smiles_to_vib": {
            "query": lambda: f"Run vibrational frequency calculation for these two SMILES strings {mol1_str} and {mol2_str} using {method}",
            "description": "Calculate vibrational frequencies using the specified method",
            "n_tasks": 4,
        },
        "name_to_opt_coord": {
            "query": lambda: f"Perform geometry optimization for these two molecule {mol1_str} and {mol2_str} using {method}",
            "description": "Optimize molecular geometry using the specified method",
            "n_tasks": 6,
        },
        "name_to_vib": {
            "query": lambda: f"Run vibrational frequency calculation for these two molecule molecule {mol1_str} and {mol2_str} using {method}",
            "description": "Calculate vibrational frequencies using the specified method",
            "n_tasks": 6,
        },
        "name_to_enthalpy": {
            "query": lambda: f"Calculate the sum of enthalpy of these two molecules {mol1_str} and {mol2_str} using {method}",
            "description": "Calculate enthalpy using the specified method",
            "n_tasks": 6,
        },
        "name_to_opt_file": {
            "query": lambda: f"Perform geometry optimization for two molecules {mol1_str} and {mol2_str} using {method}. Save the optimized coordinates in two XYZ files.",
            "description": "Optimize molecular geometry using the specified method and save the coordinate in a XYZ file",
            "n_tasks": 8,
        },
    }

    # Support lookup by index or name
    query_keys = list(query_dict.keys())

    if query_index is not None:
        if 0 <= query_index < len(query_keys):
            key = query_keys[query_index]
            return query_dict[key]["query"]()
        else:
            return "Query index out of range"

    elif query_name is not None:
        if query_name in query_dict:
            return query_dict[query_name]["query"]()
        else:
            return f"Query name '{query_name}' not found"

    else:
        return "Either query_index or query_name must be provided"
