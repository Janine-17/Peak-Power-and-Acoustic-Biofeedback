import pandas as pd
from config import CUT_MARKS_FILE

def segment_signal(signals, filepath):
    """
    Segments the input signals using the cut boundaries defined in CUT_MARKS_FILE.

    Args:
        signals (dict): Dictionary containing signal channels as keys and their
            corresponding sample data as values.
        filepath (str or Path): Path to the signal file to be segmented.

    Returns:
        dict: Dictionary containing the segmented signal channels. If no matching
            cut is found, returns the original signals unchanged.

    Notes:
        Matches file paths after normalizing path separators. If no cut is found,
        prints a warning and uses the complete, unsegmented signals.
    """
    cuts = pd.read_csv(CUT_MARKS_FILE,header=None,names=["filepath", "start_sample", "end_sample"],)
    filepath = str(filepath).replace("\\", "/")
    paths = cuts["filepath"].astype(str).str.replace("\\", "/", regex=False)
    matches = cuts[paths.apply(lambda p: filepath.endswith(p))]

    if matches.empty:
        print(f"WARNING: No cut found for '{filepath}'. Using the original signal.")
        return signals

    row = matches.iloc[0]
    return {name: values[int(row["start_sample"]):int(row["end_sample"])]for name, values in signals.items()}