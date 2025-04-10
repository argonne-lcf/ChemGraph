def get_query(
    name: str,
    query_index: int = None,
    query_name: str = None,
    method: str = "mace_mp",
    temperature: float = 298,
    pressure: float = 1.01325,
) -> str:
    """Get query for a SMILES-related task for CompChemAgent.

    Args:
        name (str): molecule name.
        query_index (int, optional): Index of the query type.
        query_name (str, optional): Name of the query type (e.g., "atomsdata", "opt_method").
        method (str, optional): Method/level of theory. Defaults to "mace_mp".
        temperature (float, optional): Temperature for thermodynamic calculations. Defaults to 298.
        pressure (float, optional): Pressure for thermodynamic calculations. Defaults to 1.01325.

    Returns:
        str: Formatted query string.
    """

    query_dict = {
        "coordinate": {
            "query": lambda: f"Provide the XYZ coordinates corresponding to this molecule: {name}",
            "description": "Get atomic coordinates of the molecule",
        },
        "opt_method": {
            "query": lambda: f"Perform geometry optimization for a molecule {name} using {method}",
            "description": "Optimize molecular geometry using the specified method",
        },
        "vib_method": {
            "query": lambda: f"Run vibrational frequency calculation for a molecule {name} using {method}",
            "description": "Calculate vibrational frequencies using the specified method",
        },
        "enthalpy_method": {
            "query": lambda: f"Calculate the enthalpy of a molecule {name} using {method}",
            "description": "Calculate enthalpy using the specified method",
        },
        "entropy_method": {
            "query": lambda: f"Calculate the entropy of a molecule {name} using {method}",
            "description": "Calculate enthalpy using the specified method",
        },
        "gibbs_method_temperature_pressure": {
            "query": lambda: f"Calculate the Gibbs free energy of a molecule {name} at T={temperature}K and P={pressure}bar using {method}",
            "description": "Calculate Gibbs free energy using the specified method and input temperature and pressure",
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
