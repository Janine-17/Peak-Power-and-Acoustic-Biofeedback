#-----------------IMPORTS----------------#
import pandas as pd
import re
from config import OUTPUT_ROOT, EMG_TEMPORAL_METRICS_TABLE

#--------------FUNCTIONS-----------------#
def extract_metadata(filename):
    parts = filename.replace("\\", "/").split("/")

    # ID
    id_match = re.search(r'ID(\d+)', filename)
    ID = int(id_match.group(1)) if id_match else None

    # Group
    if "Baseline_Data" in filename:
        Group = 0
        Repetition = 0
    elif "ABF" in filename:
        Group = "ABF"
        rep_match = re.search(r'rep(\d+)', filename)
        Repetition = int(rep_match.group(1)) if rep_match else 0
    else:
        Group = "GM"
        rep_match = re.search(r'rep(\d+)', filename)
        Repetition = int(rep_match.group(1)) if rep_match else 0

    # Session
    session_match = re.search(r'T\d+', filename)
    Session = session_match.group(0) if session_match else 0

    # Exercise
    if re.search(r'_BP_', filename, re.IGNORECASE):
        Exercise = "BP"
    elif re.search(r'_SQ_', filename, re.IGNORECASE):
        Exercise = "SQ"
    else:
        Exercise = None

    return ID, Group, Session, Exercise, Repetition

def to_seconds(row, cols):
    """
    Converts the values in the specified columns from milliseconds to seconds.

    Args:
        row (pd.Series): Row containing the values to convert.
        cols (list): Names of the columns to convert.

    Returns:
        pd.Series: Updated row with the specified values in seconds.
    """
    for col in cols:
        row[col] = row[col] / 1000
    return row

def compute_metrics(row):
    """
          Calculates temporal metrics related to muscle activation and movement.

          Args:
              row (pd.Series): Row containing cycle boundaries and muscle onset times.

          Returns:
              pd.Series: Updated row with the following metrics:
                  - Motion EMD Ag: Time between agonist onset and cycle start.
                  - Motion EMD Ant: Time between antagonist onset and cycle start.
                  - Delta Onset: Time difference between antagonist and agonist onset.
                  - Co-contraction duration: Time from the later muscle onset to
                    the end of the cycle.
          """
    row["Motion EMD Ag"] = round(row["cycle_start"] - row["agonist_onset"],3)
    row["Motion EMD Ant"] = round(row["cycle_start"] - row["antagonist_onset"],3)
    row["Delta Onset"] = round(row["antagonist_onset"] - row["agonist_onset"],3)
    row["Co-contraction duration"] = round((row["cycle_end"] - max(row["antagonist_onset"], row["agonist_onset"])),3)
    return row

def clean_data(df):
    """
    Removes rows with invalid temporal metrics.

    Excludes rows where the absolute Delta Onset is at least 1 second,
    either Motion EMD exceeds 1 second in absolute value, or the
    co-contraction duration is negative.

    Returns:
        pd.DataFrame: Cleaned DataFrame with the index reset.
    """
    mask = ( (df["Delta Onset"].abs() >= 1) |(df["Motion EMD Ag"].abs() > 1) | (df["Co-contraction duration"] < 0) | (df["Motion EMD Ant"].abs() > 1))
    return df.loc[~mask].reset_index(drop=True)

def run_emg_temporal_metrics():
    """
    Calculates EMG temporal metrics for each cycle, cleans the results,
    and saves the final table to a CSV file.

    The function loads the cycle onset data, extracts metadata, converts
    time values from milliseconds to seconds, calculates temporal metrics,
    removes invalid rows, and exports the results.
    """
    input_csv = OUTPUT_ROOT / "cycle_onsets_table.csv"
    output_csv = EMG_TEMPORAL_METRICS_TABLE

    df = pd.read_csv(input_csv, sep=",")
    df.columns = df.columns.str.strip()

    print(df.columns.tolist())
    new_rows = []

    time_cols = ["cycle_start","cycle_end","agonist_onset","antagonist_onset",]
    metric_cols = ["Motion EMD Ag","Motion EMD Ant","Delta Onset","Co-contraction duration",]
    meta_cols = ["ID","Group","Time Shift","Exercise","Repetition",]

    for _, row in df.iterrows():
        ID, Group, Session, Exercise, Repetition = extract_metadata(row["filename"])

        row["ID"] = ID
        row["Group"] = Group
        row["Time Shift"] = Session
        row["Exercise"] = Exercise
        row["Repetition"] = Repetition

        row = to_seconds(row, time_cols)
        row = compute_metrics(row)

        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)

    ordered_cols = (["filename"]+ meta_cols
        + [c for c in new_df.columns if c not in ["filename"] + meta_cols + metric_cols]
        + metric_cols)

    new_df = new_df[ordered_cols]
    new_df = clean_data(new_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_csv, index=False)

    #print("EMG temporal metrics completed (G module).")
    print(f"Results saved to: {output_csv}")
    #print(f"Rows remaining after cleaning: {len(new_df)}")
