import pandas as pd
import numpy as np




def int_throw(i) -> int:
    """
    >>> int_throw(1.1)
    Traceback (most recent call last):
        ...
    ValueError: argument must be exact integer but is 1.1
    >>> int_throw(1.0)
    1
    """
    ii = int(i)
    if ii != i:
        raise ValueError(f"argument must be exact integer but is {i}")
    return ii


def  masks_are_mutually_exclusive(m1: pd.core.series.Series, m2: pd.core.series.Series):
    """
    Check if masks are mutually exclusive
    T,T -> F
    all other cases are ok

    >>> m1 = pd.DataFrame(data=np.array([True,  False]), columns=["m"])
    >>> m2 = pd.DataFrame(data=np.array([False, True]),  columns=["m"])
    >>> masks_are_mutually_exclusive(m1["m"], m2["m"])
    True
    >>> m1 = pd.DataFrame(data=np.array([True,  False]), columns=["m"])
    >>> m2 = pd.DataFrame(data=np.array([False, False]), columns=["m"])
    >>> masks_are_mutually_exclusive(m1["m"], m2["m"])
    True
    >>> m1 = pd.DataFrame(data=np.array([True,  True]), columns=["m"])
    >>> m2 = pd.DataFrame(data=np.array([False, True]), columns=["m"])
    >>> masks_are_mutually_exclusive(m1["m"], m2["m"])
    False
    """
    return np.logical_or( ~m1, ~m2).to_numpy().all()


def masks_are_all_mutually_exclusive(masks: list):
    for i in range(len(masks)):
        for j in range(i):
            if not masks_are_mutually_exclusive(masks[i], masks[j]):
                return False
            
    return True

def union_of_masks_is_complete(masks: list):
    """
    >>> m1 = pd.DataFrame(data=np.array([True,  False]), columns=["m"])
    >>> m2 = pd.DataFrame(data=np.array([False, True]), columns=["m"])
    >>> union_of_masks_is_complete([m1["m"], m2["m"]])
    True
    >>> m1 = pd.DataFrame(data=np.array([True,  False, False]), columns=["m"])
    >>> m2 = pd.DataFrame(data=np.array([False, True,  False]), columns=["m"])
    >>> union_of_masks_is_complete([m1["m"], m2["m"]])
    False
    >>> m1 = pd.DataFrame(data=np.array([True,  False, False]), columns=["m"])
    >>> m2 = pd.DataFrame(data=np.array([False, True,  False]), columns=["m"])
    >>> union_of_masks_is_complete([m1["m"], m2["m"], m1["m"]])
    False

    """
    m0 = masks[0]
    for m in masks[1:]:
        m0 = np.logical_or(m0, m)
    return m0.to_numpy().all()

# the problem with this function is that when "0" is in the column...
#
# def isdate_masks(df, column_time: str):
#     m_null = df[column_time].isnull()
#     m_date = ~df[column_time].apply(lambda x: pd.to_datetime(x, errors='coerce')).isnull()
#     m_other = ~m_date & ~m_null
#     assert union_of_masks_is_complete([m_date, m_null, m_other])
#     assert masks_are_all_mutually_exclusive([m_date, m_null, m_other])
    
#     return {"date": m_date, 
#             "zero": m_null, 
#             "other": m_other}
    
def isdate_masks(df, column_time: str):
    m_null = df[column_time].isnull()
    m_date = ~df[column_time].apply(lambda x: pd.to_datetime(x, errors='coerce')).isnull()
    # has 0 as entry -> gives valid date without error, but wrong
    m_0 = df[column_time]==0
    m_date = m_date & ~m_0  # 
    # add entry 0 as pd.NA
    m_null = m_null | m_0  # in old data I removed to 0 by hand ... (no longer neccesary)
    m_other = ~m_date & ~m_null
    assert union_of_masks_is_complete([m_date, m_null, m_other])
    assert masks_are_all_mutually_exclusive([m_date, m_null, m_other])
    
    return {"date": m_date, 
            "zero": m_null, 
            "other": m_other}




def integerized(df, cols2ignore="status|event"):
    df_ = df.copy()
    l = df_.filter(regex=f"^((?!{cols2ignore}).)*$").columns
    for c in l:
        df_[c] = df_[c].apply(lambda x: int(x))
    return df_