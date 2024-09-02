import pandas as pd
import json
from pathlib import Path

from cdata_utils.preprocess.read_and_clean_tabular import (
     rename_columns_by_prefix,
     read_renaming_dict
)

from cdata_utils.utils import (
    isdate_masks
)

import numpy as np

import sksurv
import sksurv.preprocessing

import lifelines


from cdata_utils.descriptive.basic_stats import (
    ordinal_numbers_frequency_str, 
    ordinal_numbers_frequency_ratio_str, 
    basic_stat_description,
    BasicStats,
    MEDIAN_IQR,
    MEAN_SD,
    N_PROCENT,
    ORDINAL_0_to_1,
    ORDINAL_1_to_2,
    ORDINAL_1_to_3,
    BasicStatsOrdinal,
    BasicStatsOrdinalFractions,
    ORDINAL_F_0_to_1,
    ORDINAL_F_1_to_2,
    ORDINAL_F_1_to_3,
    generate_descriptive_table, 
    generate_descriptive_table_no_index
)

from cdata_utils.utils import int_throw

def read_and_clean_PSVD_data__BL_consensus(data_path: Path | str, 
                                           verbose=False, 
                                           return_rename_dicts = False,
                                           file_name = "data_PSVD_unified_1.xlsx" #"data_PSVD_orig.xlsx"
                                           ):
    data_origin_path = data_path / file_name
    dfo = pd.read_excel(data_origin_path)
    
    df = dfo.copy()
    # Drop patient names and DOB
    df = df.drop(columns=["Name", "Prename", "DOB"])

    # df = df.drop(columns=["ID"])  # for data 

    # rename some other inconsistend columns: 
    mapper = {
          "PV intra- or extrahepatic <50%≥ (<=0, ≥=1)": 
       "BL_PV intra- or extrahepatic <50%≥ (<=0, ≥=1)", 
          "Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean": 
       "BL_Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean"
    }
    df = df.rename(columns=mapper)


    # rename inconsistent column names: 
    df = rename_columns_by_prefix(df, prefix_old="BL_1 ", prefix_new="BL1 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_2 ", prefix_new="BL2 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_1", prefix_new="BL1", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL 1_", prefix_new="BL1 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_2", prefix_new="BL2", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_", prefix_new="BL ", verbose=verbose)


    # the table has the following structure: 
    # id, ..., 
    # BL_a, ..., BL_z, Sarc.,    
    # FU CT Date, FU_a, ..., FU_z, 
    # 'HISTOLOGY', ...,   
    # 'CLINICAL ', ...,
    # perhaps other not important stuff 
    # '1. Decompensation date',	'Last Visit', 'Death', 'BL CT Date'

    
    # for now we only focus on BL_a, ..., 
    # all other parameters are filtered
    l = list(df.columns)
    il_1 = l.index("FU CT Date") 
    il_2 = l.index("1. Decompensation date")
    cols2drop = l[il_1 : il_2]
    df = df.drop(columns=cols2drop)


    # # Drop unnamed columns
    df = df[df.columns.drop(list(df.filter(regex='Unnamed:')))]

    # # Drop all follow ups for now 
    # df = df[df.columns.drop(list(df.filter(regex='FU')))]

    # # Drop first and second BL (just keep consensus)
    df = df[df.columns.drop(list(df.filter(regex='BL1')))]
    df = df[df.columns.drop(list(df.filter(regex='BL2')))]


    c = "BL CT Date"
    assert (df[c] == df[f"{c}.1"]).all(), "not a duplicate"
    df = df.drop(columns=[c])
    df = df.rename(columns={f"{c}.1": c})

    # cosmetics: 
    mapper = {"Last Visit.1": "Last Visit" , "Death.1": "Death"}
    df = df.rename(columns=mapper)

    # fix some inconsistencies: 
    c = "BL Intrahepatic portal abnormalities consensus" 
    df[c] = df[c].apply(lambda x: str(x))
    df[c] = df[c].apply(lambda x: x.replace(".", ";"))

    c = "BL Splanchnic thrombosis consensus"
    df[c] = df[c].apply(lambda x: str(x))

    c = 'BL PV intra- or extrahepatic <50%≥ (<=0, ≥=1)'
    df[c] = df[c].fillna(0) 

    # rename mapper (cosmetics) 
    json_dir = data_path / "json" 
    rename_dict_filename = json_dir / "renaming_dict.json"
    mapper_old2new, mapper_new2old = read_renaming_dict(rename_dict_filename)
      
    if return_rename_dicts:
      return df, mapper_old2new, mapper_new2old
    else:
      return df


# for new data some column names have changed (see email data 2.0 from Kathi 24.6.24)
def read_and_clean_PSVD_data__BL_consensus_NEW(data_path: Path | str, 
                                           verbose=False, 
                                           return_rename_dicts = False,
                                           file_name = "data_PSVD_unified_3.xlsx"
                                           ):
    data_origin_path = data_path / file_name
    dfo = pd.read_excel(data_origin_path)
    
    df = dfo.copy()
    # Drop patient names and DOB
    df = df.drop(columns=["Name", "Prename", "DOB"])

    # df = df.drop(columns=["ID"])  # for data 

    # # rename some other inconsistend columns: 
    # mapper = {
    #       "PV intra- or extrahepatic <50%≥ (<=0, ≥=1)": 
    #    "BL_PV intra- or extrahepatic <50%≥ (<=0, ≥=1)", 
    #       "Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean": 
    #    "BL_Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean"
    # }
    # df = df.rename(columns=mapper)
    assert "Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean" not in df.columns
    assert "PV intra- or extrahepatic <50%≥ (<=0, ≥=1)" not in df.columns



    # rename inconsistent column names: 
    df = rename_columns_by_prefix(df, prefix_old="BL_1 ", prefix_new="BL1 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_2 ", prefix_new="BL2 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_1", prefix_new="BL1", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL 1_", prefix_new="BL1 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_2", prefix_new="BL2", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_", prefix_new="BL ", verbose=verbose)


    # the table has the following structure: 
    # id, ..., 
    # BL_a, ..., BL_z, Sarc.,    
    # FU CT Date, FU_a, ..., FU_z, 
    # 'HISTOLOGY', ...,   
    # 'CLINICAL ', ...,
    # perhaps other not important stuff 
    # '1. Decompensation date',	'Last Visit', 'Death', 'BL CT Date'

    # for now we only focus on BL_a, ..., 
    # all other parameters are filtered
    l = list(df.columns)
    il_1 = l.index("FU CT Date") 
    il_2 = l.index("1. Decompensation date")
    cols2drop = l[il_1 : il_2]
    df = df.drop(columns=cols2drop)


  

    # # Drop unnamed columns
    df = df[df.columns.drop(list(df.filter(regex='Unnamed:')))]

    # # Drop all follow ups for now 
    # df = df[df.columns.drop(list(df.filter(regex='FU')))]

    # # Drop first and second BL (just keep consensus)
    df = df[df.columns.drop(list(df.filter(regex='BL1')))]
    df = df[df.columns.drop(list(df.filter(regex='BL2')))]

    

    c = "BL CT Date"
    assert (df[c] == df[f"{c}.1"]).all(), "not a duplicate"
    df = df.drop(columns=[c])
    df = df.rename(columns={f"{c}.1": c})

    # cosmetics: 
    mapper = {"Last Visit.1": "Last Visit" , "Death.1": "Death"}
    df = df.rename(columns=mapper)

    # fix some inconsistencies: 
    c = "BL Intrahepatic portal abnormalities consensus" 
    df[c] = df[c].apply(lambda x: str(x))
    df[c] = df[c].apply(lambda x: x.replace(".", ";"))

    c = "BL Splanchnic thrombosis consensus"
    df[c] = df[c].apply(lambda x: str(x))

    # drop some other columns: 
    c2drop =  ['Biopsy date.1',
              'Days between biopsy and death',
              'nths between biopsy and death',
              #'Days between BL CT and death',
              #'Months between BL CT and death',
              'BL Location consensus (none=0, intrahep/LPV/RPV=1, extrahepatic=2, SMV/SV=3, combination any PVT=4, combination any PVT+SMV/SV=5)',  # encoded in a different way
              'Death variable',
              'Days between BL CT and death',
              'Months between BL CT and death'
              ]
    df = df.drop(columns=c2drop)

    return df
    
    # c = 'BL PV intra- or extrahepatic <50%≥ (<=0, ≥=1)'
    # df[c] = df[c].fillna(0) 

    # rename mapper (cosmetics) 
    json_dir = data_path / "json" 
    rename_dict_filename = json_dir / "renaming_dict.json"
    mapper_old2new, mapper_new2old = read_renaming_dict(rename_dict_filename)
      
    if return_rename_dicts:
      return df, mapper_old2new, mapper_new2old
    else:
      return df






def categorize_PSVD_data(df_old_names, drop_modfied_colums=False):
    df = df_old_names.copy()
    
    
    # Keept as one variable:  (could be changed)
    #    BL1_Ascites (0=none, 1=little, 2=moderate, 3=severe) 
    #    BL PV overall extent (no PVT=0, <50%=1, ≥50%=2)  
    #    BL segment IV MW (-1 = atrophy, 0 = normal,  1 = hypertrophy)
    #    BL segment 1 consensus (-1 = atrophy, 0 = normal,  1 = hypertrophy)
    
    c = "BL Splanchnic thrombosis consensus"; c1 = c
    # these are strings like: {'0', '1', '1;2', '2'}
    for i in range(1,3):
        cb = f"{c} binary cat. {i}"
        df[cb] = df[c].apply(lambda x: str(i) in x)
        
        
    c = "BL Intrahepatic portal abnormalities consensus"; c2=c
    # these are strings like: 
    # '0'       -> 0000 
    # '1',      -> 1000
    # '1;2;3;4',-> 1111
    for i in range(1,5):
        cb = f"{c} binary cat. {i}"
        df[cb] = df[c].apply(lambda x: str(i) in x)
        
        
    c = "BL Location consensus"; c3=c
    # one patient has no enty (NaN)
    # replace with 0 (up for discussion) 
    # UPDATE no longer necessary
    # df[c] = df[c].fillna(0)    
    # just differentiate between has thrombosos and has no
    cb = f"{c} binary cat."
    df[cb] = df[c].apply(lambda x: x != 0)
    
    if drop_modfied_colums:
        df = df.drop(columns=[c1, c2, c3])
        
    return df

# # probably not neccessary ... try single variable first
# def categorize(df_input, category_columns = ["BL segment 1 consensus", "BL segment IV MW"]) -> pd.DataFrame:
#     df = df_input.copy()
#     # OneHotenc. of cat.: 
#     for c in category_columns:
#         df[c] = df[c].astype("category")
#     if len(category_columns) > 0:            
#         df  = sksurv.preprocessing.OneHotEncoder().fit_transform(df)
#     return df


def categorize(df_input, category_columns=["BL segment 1 consensus", "BL segment IV MW"], drop_first=False) -> pd.DataFrame:
    df = df_input.copy()
    for c in category_columns:
        df[c] = df[c].astype("category")
    df = pd.get_dummies(df, columns=category_columns, drop_first=drop_first)
    return df


def replace_(x, value_origin=2, value_new=-1):
   if x == value_origin:
      return value_new
   else:
      return x


def reorder_some_categorical_values(df_old_names):
   df = df_old_names.copy()
   
   # BL1_segment IV (0=normal, 1=atrophy, 2=hypertrophy)
   # -> (-1 = atrophy, 0 = normal,  1 = hypertrophy)
   c = "BL segment IV MW"
   df[c] = df[c].apply(lambda x: replace_(x, value_origin=1, value_new = -1))
   df[c] = df[c].apply(lambda x: replace_(x, value_origin=2, value_new =  1))
   
   # BL1_segment 1 (0=normal, 1=atrophy, 2=hypertrophy)
   # -> (-1 = atrophy, 0 = normal,  1 = hypertrophy)
   c = "BL segment 1 consensus"
   df[c] = df[c].apply(lambda x: replace_(x, value_origin=1, value_new = -1))
   df[c] = df[c].apply(lambda x: replace_(x, value_origin=2, value_new =  1))
   
   # BL 1_Splanchnic thrombosis (0=no, 1=yes, 2=mural calcifications)
   # todo: understand
   # Soweit ich mich erinnern kann haben wir über diesen Wert gesagt dass 
   # "mural calcifications" zwischen "no" und "yes" ist. 
   # Aber was bedeutet dann "1; 2"?

   return df


# exclude lost patients
def exclude_patients(df, verbose=False):
    df_ = df.copy()
    c = "1. Decompensation date"
    m = df_["1. Decompensation date"] == "Lost"
    if verbose:
        print(f"There are {sum(m)} patients with 'Lost' in {c}")
        print(f"Droped these patients size: ({len(df_)} -> {len(df_) - sum(m)})")
    df_ = df_[~m]
    return df_



# Descriptive stat: 
# define descriptive statistic
key_stat_pairs = [
    ("Sex (1=male, 2=female)", BasicStatsOrdinal(min=1, max=2)),
    ("BL Height (m)", MEAN_SD),
    # ('BL Spleen size mean (cm)', MEAN_SD),  # some have spleenectomy (exclude for combined model, but include for stat !!)
    ('BL Ascites mean', BasicStatsOrdinal(min=0, max=3)), 
    ('BL SPSS consensus', ORDINAL_0_to_1),
    ('BL LSPSS consensus',ORDINAL_0_to_1),
    ('BL PV intra- or extrahepatic <50%≥ (<=0, ≥=1)', ORDINAL_0_to_1),
    ('BL Intrahepatic portal vein abnormalities (y/n)', ORDINAL_0_to_1),
    ('BL Liver morphology consensus', ORDINAL_0_to_1),
    ('BL segment 1 consensus', BasicStatsOrdinal(min=-1, max=1)), 
    ("BL segment IV MW", BasicStatsOrdinal(min=-1, max=1)),
    ('BL Atrophy/hypertrophy complex consensus', ORDINAL_0_to_1),
    #('BL FNH-like lesions consensus',   Nan?
    ('BL intrahepatic shunts consensus', ORDINAL_0_to_1), 
    ('BL TPMT mean', MEDIAN_IQR),
    ('BL Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean', ORDINAL_0_to_1),
    #'1. Decompensation date', 
    #'Last Visit', 'Death', 'BL CT Date',
    ('BL Splanchnic thrombosis consensus binary cat. 1', ORDINAL_0_to_1),
    ('BL Splanchnic thrombosis consensus binary cat. 2', ORDINAL_0_to_1),
    ('BL Intrahepatic portal abnormalities consensus binary cat. 1', ORDINAL_0_to_1),
    ('BL Intrahepatic portal abnormalities consensus binary cat. 2', ORDINAL_0_to_1),
    ('BL Intrahepatic portal abnormalities consensus binary cat. 3', ORDINAL_0_to_1),
    ('BL Intrahepatic portal abnormalities consensus binary cat. 4', ORDINAL_0_to_1),
    ('BL Location consensus binary cat.', ORDINAL_0_to_1),
    # 
    # ("Sex (1=male, 2=female)", BasicStatsOrdinalFractions(min=1, max=2)),
    # ('BL Ascites mean', BasicStatsOrdinalFractions(min=0, max=3)), 
    # ('BL SPSS consensus', ORDINAL_F_0_to_1),
    # ('BL LSPSS consensus',ORDINAL_F_0_to_1),
    # ('BL PV intra- or extrahepatic <50%≥ (<=0, ≥=1)', ORDINAL_F_0_to_1),
    # ('BL Intrahepatic portal vein abnormalities (y/n)', ORDINAL_F_0_to_1),
    # ('BL Liver morphology consensus', ORDINAL_F_0_to_1),
    # ('BL segment 1 consensus', BasicStatsOrdinalFractions(min=0, max=2)), 
    # ("BL segment IV MW", BasicStatsOrdinalFractions(min=-1, max=1)),
    # ('BL Atrophy/hypertrophy complex consensus', ORDINAL_F_0_to_1),
    # #('BL FNH-like lesions consensus',   Nan?
    # ('BL intrahepatic shunts consensus', ORDINAL_F_0_to_1), 
    # ('BL TPMT mean', MEDIAN_IQR),
    # ('BL Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean', ORDINAL_F_0_to_1),
    # #
    # ('BL Splanchnic thrombosis consensus binary cat. 1', ORDINAL_F_0_to_1),
    # ('BL Splanchnic thrombosis consensus binary cat. 2', ORDINAL_F_0_to_1),
    # ('BL Intrahepatic portal abnormalities consensus binary cat. 1', ORDINAL_F_0_to_1),
    # ('BL Intrahepatic portal abnormalities consensus binary cat. 2', ORDINAL_F_0_to_1),
    # ('BL Intrahepatic portal abnormalities consensus binary cat. 3', ORDINAL_F_0_to_1),
    # ('BL Intrahepatic portal abnormalities consensus binary cat. 4', ORDINAL_F_0_to_1),
    # ('BL Location consensus binary cat.', ORDINAL_F_0_to_1),
]


# make table: 
def table1_psvd(dfc, key_stat_pairs=key_stat_pairs):
    mask_all = np.array([True for _ in range(len(dfc))])
    masks = [mask_all]
    mask_names = ["all"]
    masks_zip = list(zip(mask_names, masks))
    columns_desc = [x[0] for x in key_stat_pairs]
    desc_df = generate_descriptive_table_no_index(dfc[columns_desc], masks_zip, key_stat_pairs, verbose=False)
    return desc_df


# define descriptive statistic
key_stat_pairs_spleen = [
    ('BL Spleen size mean (cm)', MEAN_SD),  # some have spleenectomy (exclude for combined model, but include for stat !!)
]


def table1_psvd_spleen(dfc, key_stat_pairs=key_stat_pairs_spleen):
    c = 'BL Spleen size mean (cm)'
    m = dfc[c].isnull()
    masks = [~m]
    mask_names = ["patients with spleen"]
    masks_zip = list(zip(mask_names, masks))

    columns_desc = [x[0] for x in key_stat_pairs]
    desc_df = generate_descriptive_table_no_index(dfc[columns_desc], masks_zip, key_stat_pairs, verbose=False)
    return desc_df



def masks_for_endpoint_1__decompensation(dfc_):
    c = "1. Decompensation date"
    df = dfc_[c]
    mask_0 = df == 0

    mask_date = ~df.apply(lambda x: pd.to_datetime(x, errors='coerce')).isnull()
    mask_date = np.logical_and(mask_date, ~mask_0) # 0 is also a valid date...

    mask_BL_decomp =  df == "BL decomp."
    mask_lost =  df == "Lost"
    mask_string_as_last_visit = dfc_.apply(lambda x: isinstance(x['Last Visit'], str), axis=1)


    mask_other = ~mask_date & ~mask_0 & ~mask_BL_decomp & ~mask_lost & ~mask_string_as_last_visit
    mask_excluded = mask_other | mask_lost | mask_BL_decomp | mask_string_as_last_visit
    masks = {
        "decompensated": mask_date, 
        "no decompensated (0)": mask_0, 
        "decompensated @ BL (excluded)": mask_BL_decomp, 
        "lost (excluded)": mask_lost, 
        "other (excluded)": mask_other,
        'unspecified last visit (exlcuded)': mask_string_as_last_visit,
        "excluded (all)": mask_excluded
    }
    return masks


def masks_for_endpoint_2__death(dfc_):
    c = "Death"
    df = dfc_[c]
    mask_nan = df.isnull()
    mask_date = ~df.apply(lambda x: pd.to_datetime(x, errors='coerce')).isnull()

    mask_string_as_last_visit = dfc_.apply(lambda x: isinstance(x['Last Visit'], str), axis=1)

    mask_other = (~mask_date & ~mask_nan) | mask_string_as_last_visit
    masks = {
        "dead": mask_date, 
        "not dead": mask_nan, 
        #"other (excluded)": mask_other,
        # 'unspecified last visit (exlcuded)': mask_string_as_last_visit
        "excluded (all)": mask_other 
    }
    return masks


def descriptive_df_from_masks(masks_dict: dict):
    rows = []
    for key in masks_dict:
        rows.append({"status": key, "cases": sum(masks_dict[key])})
    return pd.DataFrame(rows).set_index("status")


def f_dt(x, t2, t1):
    return (pd.to_datetime(x[t2]) - pd.to_datetime(x[t1])).total_seconds() / (365.25 * 24*60*60)
    

def make_y_delta(df_raw: pd.DataFrame, event_column: str):
    df = df_raw.copy()
    
    masks_date = isdate_masks(df, event_column)
    m = ~masks_date["other"] # drop Lost, ? ...
    df = df[m] 
        
    mask_delta_is_1 = masks_date["date"][m]
    mask_delta_is_0 = masks_date["zero"][m]
    assert (mask_delta_is_1.to_numpy() == ~mask_delta_is_0.to_numpy()).all()
    
    m_lv_drop = isdate_masks(df, "Last Visit")["other"]
    if sum(m_lv_drop) > 0:
        print(f"Droping {sum(m_lv_drop)} strange values in 'Last Visit' column: ", np.unique(df["Last Visit"][m_lv_drop]))
    df = df[~m_lv_drop]
    mask_delta_is_1 = mask_delta_is_1[~m_lv_drop]
    mask_delta_is_0 = mask_delta_is_0[~m_lv_drop]
    
    df['status'] = np.select(
        [mask_delta_is_1, mask_delta_is_0],
        [1, 0],
        default=pd.NA
    )
        
    f_delta_1 = lambda x: f_dt(x, event_column, 'BL CT Date')
    f_delta_0 = lambda x: f_dt(x, 'Last Visit', 'BL CT Date')    

    # Use these new columns in np.select
    df["event"] = np.select(
    [mask_delta_is_1, mask_delta_is_0],
    [df.apply(f_delta_1, axis=1), df.apply(f_delta_0, axis=1)],
    default=pd.NA
    )
    
    return df


def drop_non_numeric_columns(df):
    columns2drop = ['ID', '1. Decompensation date', 'Last Visit', 'Death', 'BL CT Date']
    return df.drop(columns=columns2drop)


def table_of_valid_entries(df):
    N = len(df)
    rows = []
    for c in df.columns:
        n = sum(~df[c].isnull())
        nn = n#f"{n}/{N}"
        rows.append({"column": c, "entries": nn})
    return pd.DataFrame(rows)


def univariate_cox_ph_summary(XY, duration_col="time", event_col="status"):
    assert duration_col in XY.columns
    assert event_col in XY.columns
    n_features = XY.shape[1] - 2
    features = [f for f in XY.columns if f != duration_col and f != event_col]
    
    df_list = []
    for j in range(n_features):
        m = lifelines.CoxPHFitter()
        XY_p = XY[[features[j], duration_col, event_col]]
        XY_p = XY_p.dropna()
        m.fit(XY_p, duration_col=duration_col, event_col=event_col)
        s = m.summary.reset_index()
        s["c-index"] = m.concordance_index_
        s["included patients"] = len(XY_p)
        s["number of events"] = sum(XY_p["status"])
        df_list.append(s)
    df = pd.concat(df_list, ignore_index=True)
    return df


# Does normalization change something for p-value?
# Does not seem like it 
def normalize_df(df, columns_to_ignore=["status", "event"]):
    normalized_df=(df-df.mean())/df.std()
    for c in columns_to_ignore:
        normalized_df[c] = df[c]
    return normalized_df



#--------------------------------



def read_and_clean_PSVD_data__BL_FU_consensus(data_path: Path | str, 
                                           verbose=False, 
                                           file_name = "data_PSVD_unified_1.xlsx" #"data_PSVD_orig.xlsx"
                                           ):
    data_origin_path = data_path / file_name
    dfo = pd.read_excel(data_origin_path)
    
    df = dfo.copy()
    # Drop patient names and DOB
    df = df.drop(columns=["Name", "Prename", "DOB"])

    # df = df.drop(columns=["ID"])  # for data 

    # # rename some other inconsistend columns:
    # mapper = {
    #     "PV intra- or extrahepatic <50%≥ (<=0, ≥=1)":
    #     "BL_PV intra- or extrahepatic <50%≥ (<=0, ≥=1)",
    #     "PV intra- or extrahepatic <50%≥ (<=0, ≥=1).1":
    #     "FU_PV intra- or extrahepatic <50%≥ (<=0, ≥=1)",
    #     "Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean":
    #     "BL_Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean",
    #     'Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean.1':
    #     'FU_Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean'
    # }
    # df = df.rename(columns=mapper)


    # rename inconsistent column names: 
    df = rename_columns_by_prefix(df, prefix_old="BL_1 ", prefix_new="BL1 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_2 ", prefix_new="BL2 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_1", prefix_new="BL1", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL 1_", prefix_new="BL1 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_2", prefix_new="BL2", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL_", prefix_new="BL ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="BL-", prefix_new="BL ", verbose=verbose)

    df = rename_columns_by_prefix(df, prefix_old="FU_1 ", prefix_new="FU1 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="FU_2 ", prefix_new="FU2 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="FU_1", prefix_new="FU1", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="FU 1_", prefix_new="FU1 ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="FU_2", prefix_new="FU2", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="FU_", prefix_new="FU ", verbose=verbose)
    df = rename_columns_by_prefix(df, prefix_old="FU-", prefix_new="FU ", verbose=verbose)


    # # the table has the following structure: 
    # # id, ..., 
    # # BL_a, ..., BL_z, Sarc.,    
    # # FU CT Date, FU_a, ..., FU_z, 
    # # 'HISTOLOGY', ...,   
    # # 'CLINICAL ', ...,
    # # perhaps other not important stuff 
    # # '1. Decompensation date',	'Last Visit', 'Death', 'BL CT Date'

    
    # # for now we only focus on BL_a, ..., 
    # # all other parameters are filtered
    # l = list(df.columns)
    # il_1 = l.index("FU CT Date") 
    # il_2 = l.index("1. Decompensation date")
    # cols2drop = l[il_1 : il_2]
    # df = df.drop(columns=cols2drop)


    # # # Drop unnamed columns
    # df = df[df.columns.drop(list(df.filter(regex='Unnamed:')))]

    # # # Drop all follow ups for now 
    # # df = df[df.columns.drop(list(df.filter(regex='FU')))]

    # # # Drop first and second BL (just keep consensus)
    # df = df[df.columns.drop(list(df.filter(regex='BL1')))]
    # df = df[df.columns.drop(list(df.filter(regex='BL2')))]

    # Does not work ...
    # for c in ["Last Visit", "Death", "BL CT Date"]:
    #     matches_na = (df[c].isnull() == df[f"{c}.1"].isnull())
    #     matches = (df[c] == df[f"{c}.1"]).dropna()
    #     if matches.all() and matches_na.all():
    #         df = df.drop(columns=[c])
    #         df = df.rename(columns={f"{c}.1": c})
    #     else: 
    #         print(df[[c, f"{c}.1", "ID"]][~matches_na])
    #         print(df[[c, f"{c}.1", "ID"]][~matches])
    #         print(df[[c, f"{c}.1", "ID"]][~matches].dropna())
    #         raise Exception(f"not a duplicate for {c}")

    # # fix some inconsistencies: 
    # c = "BL Intrahepatic portal abnormalities consensus" 
    # df[c] = df[c].apply(lambda x: str(x))
    # df[c] = df[c].apply(lambda x: x.replace(".", ";"))

    # c = "BL Splanchnic thrombosis consensus"
    # df[c] = df[c].apply(lambda x: str(x))



    # # fill some nans with 0
    # fs =["PV intra- or extrahepatic"] 
    # for f in fs:
    #     p = list(df.filter(regex=f).columns)
    #     for c in p: 
    #         print(f"filling {c} NA with 0")
    #         df[c] = df[c].fillna(0) 

    return df


def flatten(xss):
    return [x for xs in xss for x in xs]


def categorize_PSVD_data_(df_old_names, drop_modfied_colums=False):
    """
    Convert some columns into categorical (one-hot encoded).
    
    Similar to `categorize_PSVD_data`, but now regex is used to find the relevent columns. 
    (Should) work for BL, BL1, BL2, FU, ... 
    """
    df = df_old_names.copy()
    cols2_drop = []
    
    
    f = "Splanchnic thrombosis consensus"
    cols = list(df.filter(regex=f).columns)
    cols2_drop.append(cols)
    for c in cols:
        print(f"{c}  ->  ... binary cat. 1,2")
        for i in range(1,3):
            cb = f"{c} binary cat. {i}"
            m = ~df[c].isnull()
            v = df[c].apply(lambda x: str(i) in str(x))
            df[cb] = np.select([m], [v], default=pd.NA)
    

        

    f = "portal abnormalities"
    cols = list(df.filter(regex=f).columns)
    cols2_drop.append(cols)
    for c in cols:
        print(f"{c}  ->  ... binary cat. 1,..,4")
        for i in range(1,5):
            cb = f"{c} binary cat. {i}"
            m = ~df[c].isnull()
            v = df[c].apply(lambda x: str(i) in str(x))
            df[cb] = np.select([m], [v], default=pd.NA)
    
        
        
    f = "Location consensus"
    cols = list(df.filter(regex=f).columns)
    cols2_drop.append(cols)
    for c in cols:
        print(f"{c}  ->  ... binary cat.")
        cb = f"{c} binary cat."
        mask_string = df.apply(lambda x: isinstance(x[c], str), axis=1)
        print("Unique values:", list(set(df[mask_string][c])), "all mapped to 1")
        mask_0 = df.apply(lambda x: x[c]==0, axis=1)
        df[cb] = np.select(
        [mask_string, mask_0],
        [1, 0],
        default=pd.NA
    )
    
    print(flatten(cols2_drop))
    if drop_modfied_colums:
        df = df.drop(columns=flatten(cols2_drop))
        
    return df


def columns_important_variables_BL_FU(df, flatten_output = False):
    
    # drop non consensus parts 
    cols_two_readers = list(df.filter(regex="^FU1|^FU2|^BL1|^BL2").columns)
    cols2_drop = cols_two_readers
    df = df.drop(columns=cols2_drop)


    vars = ['Location consensus',
    'Splanchnic thrombosis consensus',
    'Ascites',
    'PV intra- or extrahepatic <50%≥ (<=0, ≥=1)',
    'Sarcopenia (y/n) (male TPMT<12, female TPMT<8) based on mean'
    ]

    f0 = vars[0]
    f1 = vars[1]
    f2 = "BL Ascites mean|FU Ascites consensus"
    f3 = "PV intra- or extrahepatic"
    f4 = "Sarcopenia"
    filters = [f0, f1, f2, f3, f4, "CT Date"]


    column_pairs = []
    for i, f in enumerate(filters):
        print(i, f)
        p = list(df.filter(regex=f).columns)
        #assert len(p) == 2
        column_pairs.append(p)
        

    def flatten(xss):
        return [x for xs in xss for x in xs]
    flat_list = flatten(column_pairs)
    
    if flatten_output:
        return flat_list
    else: 
        return column_pairs







def relevant_column_names(dfo, regex_patterns, chill=True):
    out = []
    for r in regex_patterns:
        cols = list(dfo.filter(regex=r).columns)
        if chill: 
            if len(cols) != 1:
                print(f"for pattern: {r}  this data was found: {cols}")
            if len(cols) == 1:
                print(f"Worked for  {r}  this data was found: {cols}")
        else:
            assert len(cols) == 1,  f"for pattern: {r}  this data was found: {cols}"
        out.append(cols)
    if chill:
        return out
    else:
        return [b[0] for b in out]
    
regex_patterns_clinical = [
    "age|Age", 
    "sex|Sex", 
    # "BL decompensated", # This makes no sense for EP1 
    "Child-Pugh-Score", 
    "MELD", "Crea", "BL_Na", "Plt|PLT", "Alb", "WBC", "PSVD cause.*1.*2", "HVPG \(mmHg\)", "LSM \(kPa\)"
]
    
def relevant_column_names_clinical(dfo, chill=True):
    return relevant_column_names(dfo, regex_patterns=regex_patterns_clinical, chill=chill)



def categorize_PSVD_clinical_data(df_old_names, drop_modfied_colums=False):
    df = df_old_names.copy()
       
    c = 'PSVD cause (1=CVID/autoimmune/inflammatory, 2=drug induced/toxic, 3=bone marrow disorder, 4=infectious/granulomatose, 5=other)'; c1=c
    m0 = df[c] == "nan"
    df[c] = df[c].apply(str)
    # these are strings like: 
    # 'PSVD cause (1=CVID/autoimmune/inflammatory, 2=drug induced/toxic, 3=bone marrow disorder, 4=infectious/granulomatose, 5=other)'],
    df["PSVD cause 1 (CVID/autoimmune/inflammatory)"] = df[c].apply(lambda x: "1" in x)
    df["PSVD cause 2 (drug induced/toxic)"] = df[c].apply(lambda x: "2" in x)
    df["PSVD cause 3 (bone marrow disorder)"] = df[c].apply(lambda x: "3" in x)
    df["PSVD cause 4 (infectious/granulomatose)"] = df[c].apply(lambda x: "4" in x)
    df["PSVD cause 5 (other)"] = df[c].apply(lambda x: "5" in x)
    # convert to int
    bool_columns = df.select_dtypes(include='bool').columns
    df[bool_columns] = df[bool_columns].astype('int64')
    
    columns_to_modify = [
        "PSVD cause 1 (CVID/autoimmune/inflammatory)", 
        "PSVD cause 2 (drug induced/toxic)",
        "PSVD cause 3 (bone marrow disorder)",
        "PSVD cause 4 (infectious/granulomatose)",
        "PSVD cause 5 (other)"
        ]
    df.loc[m0, columns_to_modify] = pd.NA
    
    if drop_modfied_colums:
        df = df.drop(columns=[c1])
        
    return df


def load_EP1_EP2_data(data_path, file_name = "data_PSVD_unified_3.xlsx", drop_negative_times_to_event_cases=True):
    # read in data
    # df, mapper_old2new, mapper_new2old = read_and_clean_PSVD_data__BL_consensus(data_path=data_path, verbose=False, return_rename_dicts=True)

    df = read_and_clean_PSVD_data__BL_consensus_NEW(data_path=data_path, file_name = file_name, verbose=False, return_rename_dicts=False)
    df = reorder_some_categorical_values(df)

    df = categorize_PSVD_data(df, drop_modfied_colums=True)
        
    c1 = "1. Decompensation date"
    c2 = "Death"
    df1 = make_y_delta(df, c1)
    df2 = make_y_delta(df, c2)
    


    # drop negative event cases maybe: 
    m = df1["event"] <= 0
    if drop_negative_times_to_event_cases and sum(m>0):
        print(f"Drop cases with negative time-to-event: ", list(df1[m]["ID"]), " for EP1")
        df1 = df1[~m]
    elif (not drop_negative_times_to_event_cases) and sum(m>0):
        print(f"WARNING negative time-to-event: ", list(df1[m]["ID"]), " for EP1 were NOT dropped")
     

    m = df2["event"] <= 0
    if drop_negative_times_to_event_cases and sum(m>0):
        print(f"Drop cases with negative time-to-event: ", list(df2[m]["ID"]), " for EP2")
        df2 = df2[~m]
    elif (not drop_negative_times_to_event_cases) and sum(m>0):
        print(f"WARNING negative time-to-event: ", list(df2[m]["ID"]), " for EP2 were NOT dropped")
     

    df1 = df1.convert_dtypes()
    df2 = df2.convert_dtypes()

    df1 = drop_non_numeric_columns(df1)
    df2 = drop_non_numeric_columns(df2)

    # make one hot encoding for atrophy, normal, hypertrophy
    # df1 = cdata_utils.project_specific.psvd.categorize(df1, category_columns=["BL segment 1 consensus", "BL segment IV MW"])
    # df2 = cdata_utils.project_specific.psvd.categorize(df2, category_columns=["BL segment 1 consensus", "BL segment IV MW"])

    return df1, df2


def load_clinical_data(data_path, file_name = "data_PSVD_unified_3.xlsx", drop_modfied_colums=True):

    dfo = pd.read_excel(data_path / file_name)

    c1 = "BL CT Date"
    c2 = "BL CT Date.1"
    assert (dfo[c1] == dfo[c2]).all()
    dfo = dfo.drop(columns=[c2])


    # extract only the relevant parameters:
    cols_clinical = relevant_column_names_clinical(dfo, chill=False)
    p = "|".join([s.replace("(","\(").replace(")", "\)").replace("|", "\|")  for s in cols_clinical])
    #df = dfo.filter(regex=p)
    df = dfo.filter(regex=p + "|ID|Date|date|DATE|Visit|Death")  #'1. Decompensation date'

    df = categorize_PSVD_clinical_data(df, drop_modfied_colums=False)

    c = 'PSVD cause (1=CVID/autoimmune/inflammatory, 2=drug induced/toxic, 3=bone marrow disorder, 4=infectious/granulomatose, 5=other)'; c1=c
    m0 = df[c] == "nan"

    columns_to_modify = [
        "PSVD cause 1 (CVID/autoimmune/inflammatory)", 
        "PSVD cause 2 (drug induced/toxic)",
        "PSVD cause 3 (bone marrow disorder)",
        "PSVD cause 4 (infectious/granulomatose)",
        "PSVD cause 5 (other)"
        ]
    df.loc[m0, columns_to_modify] = pd.NA
    
    if drop_modfied_colums:
        df = df.drop(columns=[c])
    return df
