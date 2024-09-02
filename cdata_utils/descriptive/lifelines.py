import pandas as pd
import lifelines
import matplotlib.pyplot as plt
import numpy as np


def extract_summary_data_from_lifelines(cph):

    # Extracting summary statistics directly from the CoxPHFitter object
    summary_dict = {
        "duration_col": cph.duration_col,
        "event_col": cph.event_col,
        "weights_col": cph.weights_col,
        "entry_col": cph.entry_col,
        "cluster_col": cph.cluster_col,
        "penalizer": cph.penalizer,
        "l1_ratio": cph.l1_ratio,
        "robust_variance": cph.robust or cph.cluster_col,
        "strata": cph.strata,
        "number_of_baseline_knots": getattr(cph, "n_baseline_knots", None),
        "location_of_breaks": getattr(cph, "breakpoints", None),
        "baseline_estimation_method": cph.baseline_estimation_method,
        "number_of_observations": cph.weights.sum(),
        "number_of_events_observed": cph.weights[cph.event_observed > 0].sum(),
        "log_likelihood": cph.log_likelihood_,
        "time_fit_was_run": cph._time_fit_was_called,
    }

    # Concordance, AIC and other statistics
    if cph.baseline_estimation_method == "breslow":
        summary_dict.update({
            "concordance": cph.concordance_index_,
            "partial_AIC": cph.AIC_partial_,
        })
    elif cph.baseline_estimation_method in ["spline", "piecewise"]:
        summary_dict.update({
            "AIC": cph.AIC_,
        })

    # Log-likelihood ratio test
    sr = cph.log_likelihood_ratio_test()
    summary_dict.update({
        "log_likelihood_ratio_test": sr.test_statistic,
        "log_likelihood_ratio_test_df": sr.degrees_freedom,
        "neg_log2_p_value_ll_ratio_test": -np.log2(sr.p_value),
    })

    # Convert the summary dictionary to a DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_dict, index=[0])
    #summary_df.to_csv("cph_full_summary.csv", index=False)

    # Save the summary statistics table as CSV
    coefficients_summary_df = cph.summary
    #coefficients_summary_df.to_csv("cph_coefficients_summary.csv", index=True)
    
    return summary_df, coefficients_summary_df