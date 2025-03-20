def get_query(
    reaction: dict,
    query_name: str = "enthalpy",
    temperature: float = 298,
    pressure: float = 101325,
    prop: str = "enthalpy",
    method: str = "mace_mp",
) -> str:
    """Get quer

    Returns:
        _type_: _description_
    """
    reactants_str = " + ".join([f"{r['coefficient']} {r['name']}" for r in reaction["reactants"]])
    products_str = " + ".join([f"{p['coefficient']} {p['name']}" for p in reaction["products"]])

    reaction_equation = f"{reactants_str} -> {products_str}"
    query_dict = {
        "enthalpy": f"What is the reaction enthalpy of {reaction_equation}?",
        "enthalpy_method": f"Calculate the reaction enthalpy of {reaction_equation} using {method}?",
        "gibbs_free_energy": f"What is the Gibbs free energy of reaction for {reaction_equation}?",
        "gibbs_free_energy_method": f"What is the Gibbs free energy of reaction for {reaction_equation} using {method}?",
        "gibbs_free_energy_method_temperature": f"What is the Gibbs free energy of reaction for {reaction_equation} using {method} at {temperature}K?",
    }

    return query_dict.get(query_name, "Query not found")  # Returns the query or a default message
