import os
from langchain_core.tools import tool
import pandas as pd

@tool
def get_files_in_directories(directories="/Users/celsloaner/work/llm_agent_chemistry/src/databases/CoREMOF2024DB_public/CR", extensions=".cif"):
    
    """
    Retrieve all file paths from the specified directories and their subdirectories, 
    with an optional filter for file extensions.

    Parameters:
    -----------
    directories : list of str
        A list of directory paths to search for files. Each directory should be a valid path.
    
    extensions : list of str, optional
        A list of file extensions to filter the results (e.g., ['.txt', '.csv']).
        If `None` (default), all files will be included regardless of extension.

    Returns:
    --------
    list of str
        A list containing the full paths to all matching files found in the specified directories.
    """

    file_paths = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if extensions is None or file.endswith(tuple(extensions)):
                    file_paths.append(os.path.join(root, file))
    return file_paths

@tool
def search_file_by_keyword(files, keyword):
    """
    Search for files containing the keyword (case-insensitive) in their file names.

    Parameters:
        files (list): List of file paths.
        keyword (str): The keyword to search for.
    Returns:
        list: Matching file paths.
    """
    keyword = keyword.lower()
    return [file for file in files if keyword in os.path.basename(file).lower()]

@tool
def extract_coreid_and_refcode(material_name, 
                               csv_file="/Users/celsloaner/work/llm_agent_chemistry/src/databases/CoREMOF2024DB_public/CR/ASR_data_20241125_internal.csv", 
                               ):
    """
    Reads a CSV file and extracts the 'coreid' and 'refcode' corresponding to a material name.
    """
    try:
        df = pd.read_csv(csv_file)
        match = df[df['refcode'].str.contains(material_name, case=False, na=False)]
        if not match.empty:
            return match[['coreid', 'refcode']].iloc[0]
        else:
            return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
