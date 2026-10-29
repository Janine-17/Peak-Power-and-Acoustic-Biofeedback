#---------------IMPORTS----------------#
from config import ALLOWED_GROUPBY_COLS, BY_SESSION_STATS_FILE, AGGREGATION_OUTPUT
import numpy as np
import pandas as pd
import os

#-----------FUNCTIONS-------------------#
def aggregate_metrics(df, groupby_cols):
    """
    Groups the data by the specified columns and calculates weighted means
    using the corresponding '_Counts' columns.

    Returns an aggregated DataFrame containing the mean, pooled standard
    deviation (SD), and standard error (SE) for each metric.

    Missing values are handled to avoid empty values in the output.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics and their counts.
        groupby_cols (list): Column names used to group the data.

    Returns:
        pd.DataFrame: Aggregated DataFrame containing the grouping columns
            and the calculated mean, SD, and SE for each metric.
    """
    metric_cols = [c for c in df.columns if c.endswith("Mean")]
    rows = []

    for keys, g in df.groupby(groupby_cols):
        row = dict(zip(groupby_cols, keys if isinstance(keys, tuple) else [keys]))

        for mean_col in metric_cols:
            base = mean_col.replace(" Mean", "")
            sd_col, count_col = base + " SD", base + "_Counts"

            if count_col not in g.columns:
                continue

            means = g[mean_col].values
            counts = g[count_col].values
            sds = np.nan_to_num(g[sd_col].values if sd_col in g.columns else np.zeros_like(means))

            valid = counts > 1
            if valid.sum() == 0:
                row[mean_col] = np.nan
                row[count_col] = int(counts.sum())
                row[base + " SD"] = 0
                row[base + " SE"] = 0
                continue

            means, counts, sds = means[valid], counts[valid], sds[valid]

            #ponderated mean
            weighted_mean = np.average(means, weights=counts)
            N = counts.sum()
            row[mean_col] = round(weighted_mean, 3)
            row[count_col] = int(N)

            # SD
            ss = np.sum((counts - 1) * sds**2)
            sd_pooled = np.sqrt(ss / np.sum(counts - 1)) if np.sum(counts - 1) > 0 else 0
            row[base + " SD"] = round(sd_pooled, 3)

            # SE
            row[base + " SE"] = round(sd_pooled / np.sqrt(N) if N > 0 else 0, 3)

        rows.append(row)

    return pd.DataFrame(rows)

def aggregate_mean_only(df, groupby_cols):
    """
    Groups the data and calculates the average of each numeric metric,
    rounded to three decimal places.
    """
    # Select numeric columns, excluding the grouping columns
    metric_cols = [
        c for c in df.columns
        if c not in groupby_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    df_agg = (df
        .groupby(groupby_cols)[metric_cols]
        .mean()
        .round(3)
        .reset_index()
    )

    return df_agg

def run_aggregation_interface():
    """
    Asks the user which columns to group by makes the aggregation
    """
    os.makedirs(AGGREGATION_OUTPUT, exist_ok=True)
    df = pd.read_csv(BY_SESSION_STATS_FILE, sep=',')
    df.columns = df.columns.str.strip()

    print("\nAllowed columns for aggregation:")
    print(", ".join(ALLOWED_GROUPBY_COLS))

    # --- ask for the input ---
    user_input = input( "\nEnter the columns you want to group by, separated by commas:\n")
    selected_cols = [c.strip() for c in user_input.split(",")]

    # validate if the name of the column is part of the df
    for c in selected_cols:
        if c not in ALLOWED_GROUPBY_COLS:
            print(f"❌ Column '{c}' is not allowed. Exiting.")
            return

    #the csv file will have the name based on the column names chosen before
    file_name = "_".join(selected_cols) + ".csv"
    output_path = os.path.join(AGGREGATION_OUTPUT, file_name)

    #verifies if already existed
    if os.path.exists(output_path):
        print(f"⚠️ File already exists: {output_path}")
        return


    #making the aggregation
    result = aggregate_mean_only(df, selected_cols)
    result.to_csv(output_path, index=False)
    #print(f" Aggregated file created: {output_path}")