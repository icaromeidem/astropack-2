import numpy as np
import pandas as pd


def calculate_mad(predictions, true_values, bins, param_unit):
    if not bins:
        bins = [min(true_values) - 1, max(true_values) + 1]

    df = pd.merge(left=true_values, left_index=True, right=predictions, right_index=True)

    df.columns = ["TRUE_VALUE", "PREDICTION"]

    bins_intervals = []
    bins_sizes = []
    bins_mads = []

    for bin_index in range(0, len(bins) - 1):
        bin_min = bins[bin_index]
        bin_max = bins[bin_index + 1]
        bins_intervals.append(f"[{bin_min} {param_unit}, {bin_max} {param_unit}]")

        df_bin = df[(df["TRUE_VALUE"] >= bin_min) & (df["TRUE_VALUE"] < bin_max)].copy()
        bins_sizes.append(df_bin.shape[0])

        errors = df_bin["PREDICTION"] - df_bin["TRUE_VALUE"]
        mad = np.median(np.abs(errors))
        bins_mads.append(mad)

    bins_intervals.append("Full Sample")
    bins_sizes.append(df.shape[0])
    error = df["PREDICTION"] - df["TRUE_VALUE"]
    mad = np.median(np.abs(error))
    bins_mads.append(mad)

    return pd.DataFrame({"bin": bins_intervals, "objects": bins_sizes, "mad": bins_mads})
