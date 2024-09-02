import pandas as pd
import json
from pathlib import Path


def mapper_rename_columns_by_prefix(df: pd.DataFrame, prefix_old: str, prefix_new: str) -> dict:
    """
    Filters column names starting with a given prefix and replaces that prefix
    with a new one.

    Args:
    df (pd.DataFrame): The input DataFrame.
    prefix_old (str): The old prefix to be replaced.
    prefix_new (str): The new prefix to replace the old one.

    Returns:
    dict: A dictionary with old column names as keys and new column names as values.

    Examples:
    >>> import pandas as pd
    >>> data = {'BL_1 col1': [1, 2], 'BL_1 col2': [3, 4], 'other_col': [5, 6]}
    >>> df = pd.DataFrame(data)
    >>> mapper_rename_columns_by_prefix(df, prefix_old="BL_1 ", prefix_new="BL1 ")
    {'BL_1 col1': 'BL1 col1', 'BL_1 col2': 'BL1 col2'}
    """
    cols_old = list(df.filter(regex="^" + prefix_old).columns)
    cols_new = [prefix_new + c[len(prefix_old):]  for c in cols_old]
    return {old: new for old, new in zip(cols_old, cols_new)}


def rename_columns_by_prefix(df: pd.DataFrame, prefix_old: str, prefix_new: str, verbose=False) -> dict:
    mapper = mapper_rename_columns_by_prefix(df, prefix_old ,prefix_new)
    if verbose: 
        print("Renaming columns:", mapper)
    return df.rename(columns=mapper)


def create_renaming_dict_raw(df, dir, 
                             out_filename = "renaming_dict_raw.json"):
    renaming_dict_raw = dict()
    for c_old in df.columns:
        renaming_dict_raw[c_old] = [c_old, c_old]
    # Save renaming dict raw to a JSON file
    with open(dir / out_filename, 'w') as file:
        json.dump(renaming_dict_raw, file)


def read_renaming_dict(file_name):
    # read back in
    with open(file_name, 'r') as file:
        renaming_dict = json.load(file)

    # create mapper:
    mapper_old2new = {k: renaming_dict[k][0] for k in renaming_dict.keys()}
    mapper_new2old = {mapper_old2new[k]: k for k in mapper_old2new.keys()}
    return mapper_old2new, mapper_new2old



