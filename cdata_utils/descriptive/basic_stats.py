"""
some descriptive statistics for table 1. 
E.g. mean, class frequency, ... 
"""

import pandas as pd
import numpy as np
import doctest
import typing
from typing import List
from enum import Enum
from pathlib import Path 
import collections
import yaml 
from abc import ABC, abstractmethod

from cdata_utils.utils import (
    int_throw,
    masks_are_all_mutually_exclusive, 
    union_of_masks_is_complete,
    isdate_masks
)



def ordinal_numbers_frequency(d: pd.DataFrame, min=1, max="auto") -> np.array:
    """
    >>> dat = pd.DataFrame(data=np.array([1, 1, 2, 1, 0, 0]), columns=["div = 0, male=1, female=2"])
    >>> ordinal_numbers_frequency(dat["div = 0, male=1, female=2"], min=0, max=2)
    array([2, 3, 1])
    """
    dd = d.describe()
    r = []
    if max=="auto":
        max = int_throw(dd["max"])
    for i in range(min, max+1):
        r.append((d==i).sum())
    return np.array(r)


def ordinal_numbers_frequency_str(d: pd.DataFrame, min=1, max="auto") -> str:
    """
    >>> dat = pd.DataFrame(data=np.array([1, 1, 2, 1]), columns=["male=1, female=2"])
    >>> ordinal_numbers_frequency_str(dat["male=1, female=2"])
    '(3, 1)'
    """
    r = ordinal_numbers_frequency(d, min=min, max=max)
    sr = ", ".join([str(x) for x in r])
    return f"({sr})"

def ordinal_numbers_frequency_ratio_str(d: pd.DataFrame, min=1, max="auto", percent=False) -> str:
    """
    >>> dat = pd.DataFrame(data=np.array([1, 1, 2, 1]), columns=["male=1, female=2"])
    >>> ordinal_numbers_frequency_ratio_str(dat["male=1, female=2"])
    '(0.75, 0.25)'
    """
    r = ordinal_numbers_frequency(d, min=min, max=max)
    r = r / r.sum()
    if percent:
        r *= 100
        r = np.round(r, decimals=0)
        sr = ", ".join([f"{x:.0f}%" for x in r])
    else:
        sr = ", ".join([f"{x:.2f}" for x in r])
    return f"({sr})"




class BasicStats():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
    
class BasicStatsMedian_IQR(BasicStats):
    def __init__(self):
        super().__init__()
        self.name = "MEDIAN_IQR"
    def __repr__(self):
        return f"median (IQR)"
    
class BasicStatsMean_SD(BasicStats):
    def __init__(self):
        super().__init__()
        self.name = "MEAN_SD"
    def __repr__(self):
        return f"mean ± SD"

class BasicStatsN_PROCENT(BasicStats):
    def __init__(self):
        super().__init__()
        self.name = "N_PROCENT"
    def __repr__(self):
        return "n (%)"



def n_subscript_repr(max: int, min=1, n="n") -> str:
    """
    >>> n_subscript_repr(max=4, min=0, n="n")
    '(n₀, n₁, n₂, n₃, n₄)'
    >>> n_subscript_repr(max=2, n="n")
    '(n₁, n₂)'
    """

    # Define a list to store the formatted strings
    formatted_numbers = []
    
    # Unicode subscript characters for numbers 0-9
    subscript_digits = [
        "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"
    ]
    
    def int_to_subscript(number):
        prefix = ""
        if number < 0:
            prefix = "₋"
            number *= -1
        subscript_index = "".join(subscript_digits[int(digit)] for digit in str(number))
        return prefix + subscript_index


    for i in range(min, max + 1):
        # Get the subscript representation for the index
        subscript_index = int_to_subscript(i)
        
        # Append the formatted string to the list
        formatted_numbers.append(f"{n}{subscript_index}")

    # Join the formatted strings with commas
    result = ", ".join(formatted_numbers)
    
    return "(" + result + ")"
   

class BasicStatsOrdinal(BasicStats):
    def __init__(self, min: int, max: int):
        super().__init__()
        self.name = "ORDINAL"
        self.min = min
        self.max = max

    def __repr__(self):
        return n_subscript_repr(max = self.max, min = self.min)


class BasicStatsOrdinalFractions(BasicStats):
    def __init__(self, min: int, max: int):
        super().__init__()
        self.name = "FRACTIONS"
        self.min = min
        self.max = max

    def __repr__(self):
        return n_subscript_repr(max = self.max, min = self.min, n="f")

class BasicStatsOrdinalFractionsP(BasicStats):
    def __init__(self, min: int, max: int):
        super().__init__()
        self.name = "FRACTIONS"
        self.min = min
        self.max = max
    def __repr__(self):
        return n_subscript_repr(max = self.max, min = self.min, n="p")


MEDIAN_IQR = BasicStatsMedian_IQR()
MEAN_SD = BasicStatsMean_SD()
N_PROCENT = BasicStatsN_PROCENT()
ORDINAL_0_to_1=BasicStatsOrdinal(min=0, max=1)
ORDINAL_1_to_2=BasicStatsOrdinal(min=1, max=2)
ORDINAL_1_to_3=BasicStatsOrdinal(min=1, max=3)

ORDINAL_F_0_to_1=BasicStatsOrdinalFractions(min=0, max=1)
ORDINAL_F_1_to_2=BasicStatsOrdinalFractions(min=1, max=2)
ORDINAL_F_1_to_3=BasicStatsOrdinalFractions(min=1, max=3)

ORDINAL_FP_0_to_1=BasicStatsOrdinalFractions(min=0, max=1)
ORDINAL_FP_1_to_2=BasicStatsOrdinalFractions(min=1, max=2)
ORDINAL_FP_1_to_3=BasicStatsOrdinalFractions(min=1, max=3)


def basic_stat_description(d: pd.DataFrame, basic_stat: BasicStats) -> str:
    """
    Return a string representing describing some basic statisitcs like median, ... of the data

    >>> dat = pd.DataFrame(data=np.array([1, 1, 2, 1]), columns=["sex (male=1, female=2)"])
    >>> basic_stat_description(dat["sex (male=1, female=2)"], BasicStatsOrdinal(min=1, max=2))
    '(3, 1)'
    >>> dat = pd.DataFrame(data=np.array([0.1, 0.2, 0.3]), columns=["alb"])
    >>> basic_stat_description(dat["alb"], MEDIAN_IQR)
    '0.20 (0.15 - 0.25)'
    >>> dat = pd.DataFrame(data=np.array([0.1, 0.2, 0.3]), columns=["alb"])
    >>> basic_stat_description(dat["alb"], MEAN_SD)
    '0.20 ± 0.10'
    >>> dat = pd.DataFrame(data=np.array([1, 1, 2, 1]), columns=["sex (male=1, female=2)"])
    >>> basic_stat_description(dat["sex (male=1, female=2)"]==1,  N_PROCENT)
    '3 (75.00)'
    >>> dat = pd.DataFrame(data=np.array([1, 1, 1, 1]), columns=["sex (male=0, female=1, div=2)"])
    >>> basic_stat_description(dat["sex (male=0, female=1, div=2)"], BasicStatsOrdinal(min=0, max=2))
    '(0, 4, 0)'
    """

    if isinstance(basic_stat, BasicStatsMedian_IQR):
        dd = d.describe()
        m = dd["50%"]
        iqr_l = dd["25%"]
        iqr_u = dd["75%"]
        desc_str = f"{m:.2f} ({iqr_l:.2f} - {iqr_u:.2f})"
        return desc_str
    elif isinstance(basic_stat, BasicStatsMean_SD):
        dd = d.describe()
        m = dd["mean"]
        std = dd["std"]
        desc_str = f"{m:.2f} ± {std:.2f}"
        return desc_str
    elif isinstance(basic_stat, BasicStatsN_PROCENT):
        dd = d.describe()
        N = dd["count"]
        if dd["top"]:
            n = dd["freq"]
        else:
            n = N - dd["freq"]
        n = int(n)
        r = n/N * 100
        desc_str = f"{n} ({r:.2f})"
        return desc_str
    elif isinstance(basic_stat, BasicStatsOrdinal):
        desc_str = ordinal_numbers_frequency_str(d, max = basic_stat.max, min = basic_stat.min)
        return  desc_str   
    elif isinstance(basic_stat, BasicStatsOrdinalFractions):
        desc_str = ordinal_numbers_frequency_ratio_str(d, max = basic_stat.max, min = basic_stat.min)
        return  desc_str   
    elif isinstance(basic_stat, BasicStatsOrdinalFractionsP):
        desc_str = ordinal_numbers_frequency_ratio_str(d, max = basic_stat.max, min = basic_stat.min, percent=True)
        return  desc_str   


    else:
        raise Exception("Unknown basic_stat given!")
    

def arrange_descriptive_data_in_array(df: pd.DataFrame, masks: list, key_stat_pairs: list, verbose=False) -> pd.DataFrame:
    desc_data = []
    for (column_name, mask) in masks:
        if verbose: 
            print(column_name)
        dd = []
        for (key, stat) in key_stat_pairs:
            if verbose:
                print("  ", key)

            v = basic_stat_description(df[mask][key], stat)
            dd.append(v)
        desc_data.append(dd)


    desc_data = np.array(desc_data).transpose()
    param_desc = [f"{key}" for (key, _) in key_stat_pairs]
    stats_desc_string = [f"{stat}" for (_, stat) in key_stat_pairs]
    stats_desc_string = np.array(stats_desc_string)
    ns = []
    for (mname, m) in masks:
        min = df[m].describe().iloc[0].min()
        max = df[m].describe().iloc[0].max()
        ns.append(int_throw(max))
        if max != min: 
            raise Exception(f"for mask {mname} the number of elements are different for the different quantities: min={min} != max={max}")

    columns = [f"{c} (n={n})" for ((c, _), n) in zip(masks, ns)]
    return desc_data, columns, param_desc, stats_desc_string    


def generate_descriptive_table(df: pd.DataFrame, masks: list, key_stat_pairs: list, verbose=False) -> pd.DataFrame:
    """
    Generate descriptive statistics of a type specified in `key_stat_pairs` for `df`. 

    The masks correspond to the colums of the output table. 
    One caviate: The column-name of `df` is used as a index of the output. 
    Thus it is not advised to generate several diverent statitics for the same quantity. 


    >>> test_df = pd.DataFrame(data=np.array([[1,  1,  1,  2,  2 ],
    ...                                       [15, 20, 25, 60, 70]]).transpose(),
    ...            columns=["Sex (1=male, 2=female)", "age [y]"])
    >>> mask_female = test_df["Sex (1=male, 2=female)"] == 2
    >>> mask_any = mask_female + ~mask_female
    >>> masks = [("female", mask_female), 
    ...          ("any", mask_any)]
    >>> key_stat_pairs = [
    ...    ("Sex (1=male, 2=female)", BasicStatsOrdinal(min=1, max=2)),
    ...    ("age [y]",                MEAN_SD)]
    >>> generate_descriptive_table(test_df, masks, key_stat_pairs)
                             quantity        female            any
    Sex (1=male, 2=female)   (n₁, n₂)        (0, 2)         (3, 2)
    age [y]                 mean ± SD  65.00 ± 7.07  38.00 ± 25.15
    """

    desc_data, columns, index, stats_desc_string = arrange_descriptive_data_in_array(df, masks, key_stat_pairs, verbose=verbose)


    n_groups = len(masks); n_features = len(key_stat_pairs)
    a = np.empty((n_features,n_groups+1), dtype='<U100')
    a[:, 1:] = desc_data[:,:]
    a[:,0] = stats_desc_string[:]
    columns.insert(0, "quantity")

    desc_df = pd.DataFrame(data=a,
                index=index,
                columns=columns)
    return desc_df





def generate_descriptive_table_no_index(df: pd.DataFrame, masks: list, key_stat_pairs: list, verbose=False) -> pd.DataFrame:
    """
    Generate descriptive statistics of a type specified in `key_stat_pairs` for `df`. 

    >>> test_df = pd.DataFrame(data=np.array([[1,  1,  1,  2,  2 ],
    ...                                       [15, 20, 25, 60, 70]]).transpose(),
    ...            columns=["Sex (1=male, 2=female)", "age [y]"])
    >>> mask_female = test_df["Sex (1=male, 2=female)"] == 2
    >>> mask_any = mask_female + ~mask_female
    >>> masks = [("female", mask_female), 
    ...          ("any", mask_any)]
    >>> key_stat_pairs = [
    ...    ("Sex (1=male, 2=female)", BasicStatsOrdinal(min=1, max=2)),
    ...    ("age [y]",                MEAN_SD)]
    >>> generate_descriptive_table_no_index(test_df, masks, key_stat_pairs)
                    parameter  statistic        female            any
    0  Sex (1=male, 2=female)   (n₁, n₂)        (0, 2)         (3, 2)
    1                 age [y]  mean ± SD  65.00 ± 7.07  38.00 ± 25.15
    """

    desc_data, columns, index, stats_desc_string = arrange_descriptive_data_in_array(df, masks, key_stat_pairs, verbose=verbose)



    n_groups = len(masks); n_features = len(key_stat_pairs)
    a = np.empty((n_features,n_groups+2), dtype='<U100')
    a[:, 2:] = desc_data[:,:]
    a[:,1] = stats_desc_string[:]
    a[:,0] = index[:]
    columns.insert(0, "parameter")
    columns.insert(1, "statistic")

    desc_df = pd.DataFrame(data=a,
                columns=columns)
    return desc_df




def get_occurencies(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for option in list(set(df[column])):
        rows.append({"occurancies": sum(df[column] == option)})
    df_occ = pd.DataFrame(rows, index=list(set(df[column])))
    return df_occ


def get_distinct_values_summary(df, number_of_displayed_values=10):
    rows = []
    for c in df.columns:
        num = len(set(df[c]))
        if num < number_of_displayed_values:
            v = set(df[c])
        else:
            v = "..."
        d = {
            "column": c,
            "number of distinct values": num,
            "distinct values": v
        }
        rows.append(d)
    return pd.DataFrame(rows)



def make_ordinal_desciption_single(df, c):
    mask = ~df[c].isnull()
    desc = pd.NA
    uv = np.unique(df[mask][c])
    all_integers = np.array([isinstance(x, (int, np.integer)) for x in uv]).all()
    if all_integers and len(uv)<12:
        stat = BasicStatsOrdinal(min=uv.min(),max=uv.max())
        desc = f"{stat} = {basic_stat_description(df[mask][c], stat)}"
    return desc

def make_ordinal_desciption_single_seperate(df, c):
    mask = ~df[c].isnull()
    descA = pd.NA
    descB = pd.NA
    try:
        uv = np.unique(df[mask][c])
        all_integers = np.array([isinstance(x, (int, np.integer)) for x in uv]).all()
        if all_integers and len(uv)<12:
            stat = BasicStatsOrdinal(min=uv.min(),max=uv.max())
            descA = stat
            descB = basic_stat_description(df[mask][c], stat)
    except TypeError:
        pass
    return descA, descB

def make_ordinal_desciption(df):
    rows = []
    for c in df.columns:
        a, b = make_ordinal_desciption_single_seperate(df, c)
        d = {"column": c, 
             #"ordinal description": make_ordinal_desciption_single(df, c)
             "ordinal_stat": a,
             "ordinal_stat_value": b
             }
        rows.append(d)
    return pd.DataFrame(rows)

def convert_bool_bool_to_int(df):
    df = df.convert_dtypes()
    bool_cols = df.select_dtypes(include=['bool']).columns
    for c in bool_cols:
        mNA = df[c].isnull()
        m0 = df[c] == 0
        m1 = df[c] == 1
        assert (set(df[c]).issubset(set([False, True, pd.NA])))
        # assert set(df[c]) == set([0, 1, pd.NA]), f"{set(df[c])}"
        # assert union_of_masks_is_complete([mNA, m0, m1]), f"column = {c}: Unexpected values occured!"
        
        df[c] = np.select([m0, m1], [0, 1], default=pd.NA)
    #df[bool_cols] = df[bool_cols].astype('int64')
    return df


def describe(df1):
    """ Extension to `df.describe()` with ordinal value ranges, ...    """
    #df1 = convert_bool_bool_to_int(df1)
    p1 = df1.describe()
    p2 = get_distinct_values_summary(df1, number_of_displayed_values=12).set_index("column").transpose()
    p3 = make_ordinal_desciption(df1).set_index("column").transpose()
    result = pd.concat([p1, p2, p3], axis=0)
    return result