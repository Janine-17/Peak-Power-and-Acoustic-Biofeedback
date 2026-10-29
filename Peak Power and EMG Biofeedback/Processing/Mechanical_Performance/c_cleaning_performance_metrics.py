#---------------IMPORTS-----------------------#
import numpy as np
import pandas as pd
from config import (METRICS_FILE,BAD_SIGNALS_FILE,CLEANING_OUTPUT,)


#----------------FUNCTIONS----------------------#
def read_csv_file(table_metrics):
    """
    function to read the table with all the metrics
    :return: dataframe
    """
    df = pd.read_csv(table_metrics, sep=',', engine='python', header=0).copy()

    return df

def remove_bad_signals(table_metrics, bad_signals_path):
    """
    Removes all signals listed in the bad signals file, as they were
    considered unsuitable for analysis.

    Args:
        table_metrics (pd.DataFrame): DataFrame containing the signal metrics.
        bad_signals_path (str): Path to the file containing the list of
            signals to exclude.

    Returns:
        pd.DataFrame: A copy of the metrics table with bad signals removed.

    Notes:
        Prints the number of rows removed from the table.
    """
    bad_files = pd.read_csv(bad_signals_path, header=None)[0]
    before = len(table_metrics)
    cleaned = table_metrics[~table_metrics["File Name"].isin(bad_files)].copy()
    after = len(cleaned)
    print(f"Removed {before - after} rows due to bad signals")
    return cleaned

def add_gender(table_metrics):
    """
    receives the table with the metrics and add a column with the information about the gender,
    after the column "Group". get gender by the column ID
    :param table_metrics:
    :return: The table with the previous metrics + gender
    """
    id_gender = {'1': "M", '2': "M", '3': "W", '4':"W", '5': "M",
              '6': "W", '7': "M", '8': "M", '10':"W", '11': "W",
              '12': "M", '14': "W", '15': "M", '17':"M", '18': "W",
              '19': "W", '20': "W", '21': "W", '22':"M", '23': "W",
              '24': "W", '25': "W", '26': "W"}

    table_metrics["Gender"] = table_metrics["ID"].astype(str).map(id_gender)

    if "Group" in table_metrics.columns:
        cols = list(table_metrics.columns)
        cols.remove("Gender")
        group_idx = cols.index("Group")
        cols.insert(group_idx + 1, "Gender") #after the group column
        table_metrics = table_metrics[cols]
    return table_metrics

def group_py_identifier(table_metrics):
    """
    Create sub-dataframe organized and group by the identifiers. it will be used to see outliers and metrics
    :param table_metrics: df
    :return: every_group which is the groups preperaed to be cleaned and to do statistics
    """
    identifiers = ["ID", "Session", "Exercise", "Repetition"]
    every_group = {}
    grouped = table_metrics.groupby(identifiers)

    for group_keys, group_df in grouped:
        every_group[group_keys] = group_df.copy()

    return every_group

def classify_columns(table_metrics):
    """
    Groups the metric columns into categories according to their role
    in the analysis.

    Args:
        table_metrics (pd.DataFrame): DataFrame containing the signal metrics.

    Returns:
        dict: Dictionary mapping each category to its list of columns:
            - identifiers: Columns used to group the data.
            - class_fatal: Metrics whose invalid values or outliers can
                invalidate an entire row.
            - class_peaks: Primary peak metrics whose outliers can invalidate
                their associated derived metrics.
            - class_derived_power: Metrics derived from peak power.
            - class_derived_force: Metrics derived from peak force.
    """
    identifiers = ["ID", "Session", "Exercise", "Repetition"]

    class_fatal = ["Total Distance (m)", "Duration of Cycle (s)", "Peak Velocity (m/s)"]
    class_peaks = ["Peak Power (W)", "Peak Force (N)"]
    class_derived_power =  ["Phase Peak Power (%)", "Time Peak Power (s)", "Phase Shift PP and PV (%)",
                            "Total Concentric Work (J)", "F(P=Max) (N)", "V(P=Max) (m/s)",
                            "Peak RPD (W/s)", "Phase Peak RPD (%)", "Time Peak RPD (s)"]
    class_derived_force =  ["Phase Peak Force (%)", "Time Peak Force (s)", "Impulse (N/s)",
                            "F(P=Max) (N)"]

    classified_columns = {'identifiers': identifiers, 'class_fatal': class_fatal, 'class_peaks': class_peaks,
                          'class_derived_power': class_derived_power, 'class_derived_force': class_derived_force}

    return classified_columns

def detect_outliers(group, classified_columns):
    """
    Detect outliers in a group of data based on column classification. through IQR method

    Rules:
    - class_fatal & class_peaks & class_medium → 1.5 IQR
    - Outliers are replaced by np.nan;
    :param group: sub df --> group that was organized by the identifiers
    :param classified_columns: dic with the class of columns based on gravidade
    :return: subgroup with the ouliers with new value --> np.nan
    """
    thresholds = {'class_fatal': 2, 'class_peaks': 2}

    for class_type in ['class_fatal', 'class_peaks']:
        if class_type not in classified_columns:
            continue #avoiding error

        threshold = thresholds[class_type]
        for column in classified_columns[class_type]:
            if column not in group.columns:
                continue  #avoiding error

            col_values = group[column].dropna() #ignore nan values

            if len(col_values) < 3:
                continue  #if there is less than 3 values, do nothing

            Q1 = np.percentile(col_values, 25)
            Q3 = np.percentile(col_values, 75)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            outlier_mask = (group[column] < lower_limit) | (group[column] > upper_limit)
            group.loc[outlier_mask, column] = np.nan

    return group

def propagate_outliers(group, columns_dict):
    """
    Defines how outliers should be propagated. Four options are available:

    - class_fatal: The entire row becomes invalid if any value is NaN or an outlier.
    - class_peaks: An outlier in a primary metric marks its associated derived
      metrics as NaN (class_derived_power or class_derived_force).
    - class_medium: An outlier in an intermediate variable marks both that
      variable and its associated variables as NaN.
    - identifiers: Columns used to group the data.

    Args:
        table_metrics (pd.DataFrame): DataFrame containing the signal metrics.

    Returns:
        dict: Dictionary grouping the columns by category.
    """
    exclude_cols = (columns_dict["identifiers"] + ["File Name", "Cycle Number", "Group", "Gender", "Time Shift"])

    fatal_cols = columns_dict["class_fatal"]
    fatal_mask = group[fatal_cols].isna().any(axis=1) #it gives true if any of the fatal columns has NaN
    #any axis= 1 --> ve linha a linha e se qualquer coluna for NaN, a linha passa a ser true
    metric_cols = [col for col in group.columns if col not in exclude_cols]
    group.loc[fatal_mask, metric_cols] = np.nan #para todas as linhas marcadas com fatal, todas as colunas com metricas recebem NaN


    if "Peak Power (W)" in group.columns:
        power_mask = group["Peak Power (W)"].isna()
        for col in columns_dict["class_derived_power"]:
            if col in group.columns:
                group.loc[power_mask, col] = np.nan

    if "Peak Force (N)" in group.columns:
        force_mask = group["Peak Force (N)"].isna()
        for col in columns_dict["class_derived_force"]:
            if col in group.columns:
                group.loc[force_mask, col] = np.nan

    #if both peaks are NaN, all the line is erased it
    if "Peak Power (W)" in group.columns and "Peak Force (N)" in group.columns:
        both_peaks_nan_mask = (group["Peak Power (W)"].isna() & group["Peak Force (N)"].isna())
        metric_cols = [col for col in group.columns if col not in exclude_cols]
        group.loc[both_peaks_nan_mask, metric_cols] = np.nan

    return group

def control_cleaning(group, columns_dict):
    """
    Controls the amount of valid data remaining after outlier removal.

    Counts the number of non-NaN values in each metric column for each
    group. These counts can be used to identify groups with fewer than
    three valid values per column and assess data completeness.

    Args:
        group (pd.DataFrame): DataFrame containing the metrics for a group.
        columns_dict (dict): Dictionary classifying columns by category.

    Returns:
        pd.DataFrame: Updated group DataFrame with an additional
            '_Counts' column for each metric, indicating the number
            of non-NaN values. The 'Cycle Number' column is removed
            if present.
    """

    exclude_cols = columns_dict["identifiers"] + ["File Name", "Group", "Gender"]

    metrics_cols = [c for c in group.columns if c not in exclude_cols]

    for col in metrics_cols:
        group[f"{col}_Counts"] = group[col].notna().sum()

    if "Cycle Number" in group.columns:
        group = group.drop(columns=["Cycle Number"])

    return group

def mean_sd_cv_metrics(group, columns_dict):
    """
    receives the group, do the mean, sd and cv for each group for each numerical columns (every one execpt the excluded_cols
    it ignores the NaN Values
    add news columns with the old name column + "Mean" or "SD" or "CV". maintain the old columns as well.
    so, all lines form that group will become only one line.
    :param group:
    :param columns_dict:
    :return: one line dataframe
    """
    exclude_cols = (columns_dict["identifiers"] + ["File Name", "Group", "Cycle Number", "Gender", "Time Shift"])

    metrics_columns = [col for col in group.columns if col not in exclude_cols and not col.endswith("_Counts")]

    summary = {}

    #creates the identifiers as before
    for col in columns_dict["identifiers"] + ["File Name", "Group", "Gender", "Time Shift"]:
        if col in group.columns:
            summary[col] = group[col].iloc[0]
        else:
            summary[col] = np.nan

    for col in metrics_columns:
        values = group[col]. dropna() #access values and ignore the NaN
        mean_val = round(values.mean(),3) if not values.empty else np.nan
        sd_val = round(values.std(),3) if not values.empty else np.nan
        cv_val = round((sd_val / mean_val)*100,3) if mean_val != 0 else np.nan

        summary[f"{col} Mean"] = mean_val
        summary[f"{col} SD"] = sd_val
        summary[f"{col} CV"] = cv_val

    count_cols = [c for c in group.columns if c.endswith("_Counts")]
    for col in count_cols:
        summary[col] = group[col].iloc[0]

    #it returns only one line
    return pd.DataFrame([summary])

def remove_third_rep(table_metrics):
    two_reps = table_metrics[table_metrics["Repetition"] != 3.0]
    return two_reps

def remove_one_four_session(table_metrics):
    six_sessions = table_metrics[~table_metrics["Session"].isin([1])]
    return six_sessions

def temporal_column(table_metrics):
    """
    creates a new columns with the temporal component: T1 (begginning), T2 (middle), T3 (end)
    :param table_metrics:
    :return:
    """
    table_metrics = table_metrics.copy()
    table_metrics["Time Shift"] = pd.Series(index=table_metrics.index, dtype="object")

    #T0- Baseline, T1-4, T2-7
    table_metrics.loc[table_metrics["Session"] == 4, "Time Shift"] = "T1"

    table_metrics.loc[(table_metrics["Session"] == 7) & (table_metrics["ID"] != 14), "Time Shift"] = "T2"
    table_metrics.loc[(table_metrics["Session"] == 8) & (table_metrics["ID"] == 14), "Time Shift"] = "T2"

    return table_metrics

def run_cleaning():
    """
    Runs the complete data-cleaning pipeline and exports three output files:

    - groups_all.csv: Cleaned data for all groups.
    - by_session_stats.csv: Statistics grouped by session.
    - filtered_session_stats.csv: Filtered statistics grouped by session.

    The output files are saved in the CLEANING_OUTPUT directory.
    """

    table_metrics = read_csv_file(METRICS_FILE)
    table_metrics = add_gender(table_metrics)
    table_metrics = remove_bad_signals(table_metrics, BAD_SIGNALS_FILE)

    every_group = group_py_identifier(table_metrics)
    class_columns = classify_columns(table_metrics)

    all_groups_clean = []
    by_session_stats_rows = []
    filtered_session_stats_rows = []

    for group_keys, group_df in every_group.items():
        # Cleaning data
        group_clean = detect_outliers(group_df.copy(), class_columns)
        group_clean = propagate_outliers(group_clean, class_columns)

        # 1. Cleaned data, maintaining one line per cycle
        all_groups_clean.append(group_clean.copy())

        # 2. Statistic per session before the filtering
        group_for_stats = control_cleaning(
            group_clean.copy(), class_columns
        )
        stats_row = mean_sd_cv_metrics(
            group_for_stats, class_columns
        )
        by_session_stats_rows.append(stats_row)

        # 3. Statistics only for the sessions and repetitions wanted
        filtered_group = remove_one_four_session(
            group_for_stats.copy()
        )
        filtered_group = remove_third_rep(filtered_group)
        filtered_group = temporal_column(filtered_group)

        if not filtered_group.empty:
            filtered_stats_row = mean_sd_cv_metrics(
                filtered_group, class_columns
            )
            filtered_stats_row = filtered_stats_row.drop(
                columns=["Session"], errors="ignore"
            )
            filtered_session_stats_rows.append(filtered_stats_row)

    # Creating the destiny folder
    CLEANING_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Export 3 files
    if all_groups_clean:
        final_df = pd.concat(all_groups_clean, ignore_index=True)
        final_df.to_csv( CLEANING_OUTPUT / "groups_all.csv", index=False)
        print("Exported: groups_all.csv")

    if by_session_stats_rows:
        stats_df = pd.concat(by_session_stats_rows, ignore_index=True)
        stats_df.to_csv(CLEANING_OUTPUT / "by_session_stats.csv", index=False)
        print("Exported: by_session_stats.csv")

    if filtered_session_stats_rows:
        filtered_stats_df = pd.concat(filtered_session_stats_rows, ignore_index=True)
        filtered_stats_df.to_csv(CLEANING_OUTPUT / "filtered_session_stats.csv",index=False,)
        print("Exported: filtered_session_stats.csv")
