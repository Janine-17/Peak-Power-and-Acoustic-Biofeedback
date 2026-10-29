#-----------------IMPORTS-------------#
import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert as scipy_hilbert
from scipy.ndimage import gaussian_filter1d
from config import EMG_COMPLETE_METRICS_TABLE, FS, EMG_TEMPORAL_METRICS_TABLE, FILE_EXTENSION, PROCESSED_ROOT
from Processing.Mechanical_Performance.a_analysis import emg_signal_by_filename, load_file_list, small_root_file, read_columns_csv

#---------------FUNCTIONS--------------#
def tkeo(x):
    """
    this function uses the Teager–Kaiser Energy Operator to calculate the instantaneous energy of the signal
    :param x: it's the emg signal
    :return: energy vector of Teager-Kaiser. First and last samples are zero.
    """
    x = x.astype(float)
    psi = np.zeros_like(x, dtype=float)
    psi[1:-1] = x[1:-1]**2 - x[0:-2]*x[2:] # formula ψ[n]=x[n]^2−x[n−1]x[n+1]
    return psi

def smooth_signal(signal, window_size=350):
    """Simple moving average smoothing."""
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def hilbert_x2_for_peak(signal):
    """
    Hilbert x2 applied only to identify the time of the EMG peak.
    """
    x_tkeo = tkeo(signal)
    x_feat = np.abs(x_tkeo)
    x_smooth = smooth_signal(x_feat, 25)

    a1 = scipy_hilbert(x_smooth)
    env1 = np.abs(a1)
    env1_smooth = smooth_signal(env1, 50)

    a2 = scipy_hilbert(env1_smooth)
    env2 = np.abs(a2)

    env2_smooth = gaussian_filter1d(env2, 80)
    return env2_smooth

def identify_peak_time(signal, onset, offset):
    """
    Identify peak time between onset and offset.
    """
    segment = signal[onset:offset]
    peak_idx = np.argmax(segment)
    return onset + peak_idx

def emg_rise(signal_envelope, time_peak, time_onset, fs=1000):
    """
    Compute normalized EMG rise.
    """
    emg_max = signal_envelope[time_peak]
    env_norm = signal_envelope / emg_max

    if time_peak > time_onset:
        rise_sec = ((env_norm[time_peak] - env_norm[time_onset])/ ((time_peak - time_onset) / fs))
    else:
        rise_sec = np.nan
    return env_norm, round(rise_sec, 3)

def env_ratio(env_ag, env_ant, onset_ag, onset_ant, cycle_end):
    """
    Ratio between agonist and antagonist envelope AUC.
    """
    ag_seg = env_ag[onset_ag:cycle_end]
    ant_seg = env_ant[onset_ant:cycle_end]

    auc_ag = np.trapezoid(ag_seg)
    auc_ant = np.trapezoid(ant_seg)

    if auc_ant == 0:
        return np.nan

    return round(auc_ag / auc_ant, 3)

def cleaning_wrong_values(df):
    """
    Removes invalid rows based on physiological and methodological criteria.
    """
    conditions = (
        (df['Delta Peak-Onset Ag (s)'] > 0) &  # The peak cannot occur before muscle onset
        (df['Delta Peak-Onset Ant (s)'] > 0) &  # The peak cannot occur before muscle onset
        (df['EMG Rise Norm Ag (/s)'] > 0) & # EMG rise values cannot be negative
        (df['EMG Rise Norm Ant (/s)'] > 0) & # EMG rise values cannot be negative
        (df['EMG Rise Norm Ag (/s)'] <= 10) &  #  Values above 10 are unrealistic due to the very short interval
        (df['EMG Rise Norm Ant (/s)'] <= 10) &   # Values above 10 are unrealistic
        (df['Env Ratio Ag-Ant'] > 0) & # Area ratios cannot be negative
        (df['Abs Delay Peaks (s)'] <= 1) & # The time difference between peaks cannot be excessively large
        ~((df['Motion EMD Ant'] < -0.2) & (df['Motion EMD Ag'] < -0.2))   #Both muscle onsets cannot occur after mechanical onset
    )

    df_clean = df[conditions].reset_index(drop=True)
    return df_clean


def run_emg_complete_metrics():
    csv_positions_emg = EMG_TEMPORAL_METRICS_TABLE
    output_file = EMG_COMPLETE_METRICS_TABLE

    # Load CSV with onsets and cycle information
    df_pos = pd.read_csv(csv_positions_emg)

    # List the processed files
    files_list = load_file_list(FILE_EXTENSION, PROCESSED_ROOT)

    results = []

    for filepath in files_list:
        rel_path = small_root_file(filepath)

        if rel_path is None:
            continue

        rel_path = os.path.normpath(rel_path)
        # Find the concentric cycles corespondent to the file
        df_file = df_pos[df_pos["filename"] == rel_path]

        #if it is not possible to find anything, move to the next one
        if df_file.empty:
            continue

        print("Processing:", rel_path)

        signals, fs, labels = read_columns_csv(filepath,fs=FS)

        emg_ag, emg_ant, emg_ag_env, emg_ant_env = (emg_signal_by_filename(signals, filepath))

        # Hilbert x2 only to determine peak timing
        env_x2_ag = hilbert_x2_for_peak(emg_ag)
        env_x2_ant = hilbert_x2_for_peak(emg_ant)

        for _, row in df_file.iterrows():
            onset_ag = int(row["agonist_onset"] * fs)
            onset_ant = int(row["antagonist_onset"] * fs)
            cycle_end = int(row["cycle_end"] * fs)
            cycle_start = int(row["cycle_start"] * fs)

            # position/time of peak in agonist and in antagonist
            peak_ag = identify_peak_time(env_x2_ag, onset_ag, cycle_end)
            peak_ant = identify_peak_time(env_x2_ant, onset_ant, cycle_end)

            env_ag_norm, rise_ag = emg_rise(emg_ag_env, peak_ag, onset_ag)
            env_ant_norm, rise_ant = emg_rise(emg_ant_env, peak_ant, onset_ant)

            ratio = env_ratio(env_ag_norm, env_ant_norm, onset_ag, onset_ant, cycle_end)

            cycle_start = row['cycle_start']
            cycle_end_sec = row['cycle_end']
            cycle_duration = round(cycle_end_sec - cycle_start, 3)

            delta_onset = row['Delta Onset']
            abs_delta_onset = round(abs(delta_onset), 3)

            delay_peak_time = round((peak_ant - peak_ag) / fs, 3)
            abs_delay_peak_time = round(abs(delay_peak_time), 3)

            peak_percent_ag = round((((peak_ag / fs) - cycle_start) / cycle_duration) * 100, 3)
            peak_percent_ant = round((((peak_ant / fs) - cycle_start) / cycle_duration) * 100, 3)

            delay_peaks_perc = round(peak_percent_ag - peak_percent_ant, 3)
            abs_delay_peaks_perc = abs(delay_peaks_perc)

            motion_emd_ag = row['Motion EMD Ag']
            motion_emd_ant = row['Motion EMD Ant']
            motion_emd_ag_perc = round(-(motion_emd_ag / cycle_duration) * 100, 3)
            motion_emd_ant_perc = round(-(motion_emd_ant / cycle_duration) * 100, 3)

            co_contraction = row['Co-contraction duration']

            # Relationship between the delay between the peaks and between the onsets
            if abs(delay_peak_time) < 0.001:
                ratio_delays_peaks_onsets = 0
            else:
                ratio_delays_peaks_onsets = round(
                    abs_delta_onset / abs(delay_peak_time), 3
                )

            results.append({
                **row.to_dict(),

                # --- Cycle timing ---
                'Cycle Duration': cycle_duration,
                'Co-Contraction/ Cycle Duration': round((co_contraction / cycle_duration), 3),

                # --- Motion EMD %Cycle ---
                'Motion EMD Ag %Cycle': motion_emd_ag_perc,
                'Motion EMD Ant %Cycle': motion_emd_ant_perc,

                # --- Peak times ---
                'Time Env Peak Ag': round(peak_ag / fs, 3),
                'Time Env Peak Ant': round(peak_ant / fs, 3),

                'Peak Env Ag % Cycle': round(peak_percent_ag, 1),
                'Peak Env Ant % Cycle': round(peak_percent_ant, 1),

                'Delta Peaks-Time Env % Cycle': delay_peaks_perc,
                'Abs Delta Peaks-Time Env % Cycle': abs_delay_peaks_perc,

                # --- Peak delays ---
                'Delay Peaks Ant-Ag (s)': delay_peak_time,
                'Abs Delay Peaks (s)': abs_delay_peak_time,

                # --- Onset-to-peak ---
                'Delta Peak-Onset Ag (s)': round((peak_ag - onset_ag) / fs, 3),
                'Delta Peak-Onset Ant (s)': round((peak_ant - onset_ant) / fs, 3),

                # --- Onset metrics ---
                'Abs Delta Onset (s)': abs_delta_onset,
                'Abs Delta Onset %Cycle': abs(round(motion_emd_ag_perc - motion_emd_ant_perc, 3)),

                # --- EMG dynamics ---
                'EMG Rise Norm Ag (/s)': rise_ag,
                'EMG Rise Norm Ant (/s)': rise_ant,

                # --- Coordination ---
                'Env Ratio Ag-Ant': ratio,
                'Ratio Peaks-Onsets': ratio_delays_peaks_onsets
            })

    df_out = pd.DataFrame(results)

    # Data cleaning
    df_out_clean = cleaning_wrong_values(df_out)

    # Export cleaned dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out_clean.to_csv(output_file, index=False)

    print("(H Module): Analysis completed and cleaned data saved to:")
    print(output_file)
    #print(f"Original rows: {len(df_out)} | "f"After cleaning: {len(df_out_clean)}")
