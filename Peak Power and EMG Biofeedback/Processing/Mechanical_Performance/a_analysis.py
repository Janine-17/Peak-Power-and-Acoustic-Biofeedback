#---------------------IMPORTS----------------------#
import os, glob
import pandas as pd
from scipy.signal import find_peaks
from config import *
from Processing.Mechanical_Performance.b_signal_performance_metrics import (
    total_measure,area_under_curve,calculate_rpd,peaks_metrics, median_frequency,
    get_force, get_distance, get_power, get_time, peaks_on_file)
from Processing.Mechanical_Performance.plots import *
from Preprocessing.c_segmentation import segment_signal

#--------------------Functions---------------------#
def load_file_list(FILE_EXTENSION,DATA_ROOT):
    """
    Load all .txt files from the 'raw/' folder recursively.

    :return: sorted list of file paths
    """
    file_list = glob.glob(os.path.join(DATA_ROOT, "**", FILE_EXTENSION), recursive=True)
    return sorted(file_list)  # sort files alphabetically

def small_root_file (filepath):
    """
    Extracts the relative file path after the 'processed' directory.

    Args:
        filepath (str): Path to a CSV or TXT file.

    Returns:
        str or None: Relative file path after 'processed', or None if
        the 'processed' directory is not found.
    """
    full_path = os.path.normpath(filepath)
    parts = full_path.split(os.sep)
    if "processed" in parts:
        index = parts.index("processed")
        return os.path.join(*parts[index+1:])
    else:
        return None

def read_columns_csv(filepath, fs=1000, keywords= None):
    """
    Read csv file and extract selected columns, and only extract between the cutting samples.
    :param filepath: csv/txt file
    :param fs: sampling frequency
    :param keywords: list of strings to filter columns (e.g. ["Velocity", "EMG"]) (optional)
    :return: signals dict, fs, column labels
    """
    df = pd.read_csv(filepath, sep=None, engine='python')

    if keywords is None:
        labels = df.columns.tolist()
    else:
        labels = [col for col in df.columns if any (key in col for key in keywords)]
    signals = {label: df[label].values for label in labels}

    return signals, fs, labels

def emg_signal_by_filename(signals, filepath):
    """
    function that receives the all the emg_signals and the corresponding filepath. return the agonist-antagonist emg signals
    based on the name of the file ("BP", "DR" or "SQ")
    :param signals: dict with the emg signals
    :param filepath: filepath of the signal
    :return: 1D signals
    """
    fname_upper = filepath.upper()

    if "DR" in fname_upper:
        emg_ag = signals["EMG_BB"].astype(float)
        emg_ant = signals["EMG_TB"].astype(float)
        emg_ag_env = signals["EMG_BB_ENV"].astype(float)
        emg_ant_env = signals["EMG_TB_ENV"].astype(float)

    elif "BP" in fname_upper:
        emg_ag = signals["EMG_TB"].astype(float)
        emg_ant = signals["EMG_BB"].astype(float)
        emg_ag_env = signals["EMG_TB_ENV"].astype(float)
        emg_ant_env = signals["EMG_BB_ENV"].astype(float)

    elif "SQ" in fname_upper:
        emg_ag = signals["EMG_RF"].astype(float)
        emg_ant = signals["EMG_BF"].astype(float)
        emg_ag_env = signals["EMG_RF_ENV"].astype(float)
        emg_ant_env = signals["EMG_BF_ENV"].astype(float)
    else:
        raise ValueError(f"There is no specific type of exercise in this file: {filepath}]")


    return emg_ag, emg_ant,emg_ag_env, emg_ant_env

def move_peak_to_plateau_direction(signal, peaks, mode='high', max_search=500, relative_threshold=0.95):
    """
    Adjusts peaks to the beginning of the plateau in a sensitive and safe way.

    - High: moves backward, never increasing the value.
    - Low: moves forward, never decreasing the value.

    Args:
        signal (array): Original signal.
        peaks (array): Peak indices.
        mode (str): 'high' or 'low'.
        max_search (int): Number of samples to search in the appropriate direction.
        relative_threshold (float): Fraction of the peak value used to define
            the beginning of the plateau.
    """
    N = len(signal)
    adjusted_peaks = []

    for p in peaks:

        if mode == 'high':
            left = max(p - max_search, 0)
            seg = signal[left:p + 1]
            threshold = signal[p] * relative_threshold
            idx_candidates = np.where(seg < threshold)[0]
            idx = idx_candidates[-1] + 1 if len(idx_candidates) > 0 else 0
            # guarantee that it will not move forward
            idx = min(idx, p - left)
            adjusted_peaks.append(left + idx)

        elif mode == 'low':
            right = min(p + max_search + 1, N)
            seg = signal[p:right]
            tol = 0.005  # small micro flutuations are allowed
            idx_candidates = np.where(seg <= signal[p] + tol)[0]  # evey point of the plateau
            idx = idx_candidates[-1] if len(idx_candidates) > 0 else 0
            adjusted_peaks.append(p + idx)

    return np.array(adjusted_peaks)

def concentric_phase(displacement, max_peaks = 6, points = 1100, threshold_sd = 1.5, max_peak_peak = 2000):
    high_peaks, _ = find_peaks(displacement, distance=points-50)
    low_peaks, _ = find_peaks(-displacement, distance=points-50)

    threshold = threshold_sd * np.std(displacement)


    #if len(low_peaks) > max_peaks:
        #low_values = displacement[low_peaks]
        #sorted_indices = np.argsort(low_values)  # from lowest to the biggest
        #low_peaks = low_peaks[sorted_indices[:max_peaks]]

    valid_highs = []
    for hp in high_peaks:
        lows_left = low_peaks[low_peaks < hp]
        lows_left.sort()

        if len(lows_left) > 0 and (hp-lows_left[-1]) < 2000:
            valid_highs.append(hp)
    high_peaks = np.array(valid_highs)

    if len(high_peaks) > max_peaks:
        high_values = displacement[high_peaks]
        sorted_indices = np.argsort(high_values)[::-1]  # from the biggest to lowest
        high_peaks = high_peaks[sorted_indices[:max_peaks]]
    high_peaks = np.sort(high_peaks)
    low_peaks = np.sort(low_peaks)

    # making sue that the first peak is a "low"
    if len(low_peaks) > 0 and len(high_peaks) > 0:
        if high_peaks[0] < low_peaks[0]:
            high_peaks = high_peaks[1:] #remove the high outlier at the beggining

        if low_peaks[-1] > high_peaks[-1]: #remove the lower outlier at the end
            low_peaks = low_peaks[:-1]

    valid_lows = []
    valid_highs = []
    for lp in low_peaks:
        candidates = high_peaks[high_peaks > lp]
        if len(candidates) > 0:
            hp = candidates[0]
            if displacement[hp] - displacement[lp] >= threshold:
                valid_lows.append(lp)
                valid_highs.append(hp)

    low_peaks = np.array(valid_lows)
    high_peaks = np.array(valid_highs)

    #Optional: uncomment these prints to see if the function is working in a proper way
    #print("DEBUG ─ cleaned low_peaks :", low_peaks)
    #print("DEBUG ─ cleaned high_peaks:", high_peaks)

    pairs = list(zip(low_peaks, high_peaks))
    seen_lows = {}
    seen_highs = {}

    for l, h in pairs:
        diff = h - l

        #if low is duplicated
        if l in seen_lows:
            if diff < seen_lows[l][1]: #pair with low differences
                seen_lows[l] = (h, diff)
        else:
            seen_lows[l] = (h, diff)

        #if high is duplicated
        if h in seen_highs:
            if diff < seen_highs[h][1]:
                seen_highs[h] = (l, diff)
        else:
            seen_highs[h] = (l, diff)

    #make the lists again
    final_pairs = []
    for l, (h, _) in seen_lows.items():
        if seen_highs[h][0] == l:
            final_pairs.append((l, h))

    low_peaks = np.array([l for l, h in final_pairs])
    high_peaks = np.array([h for l, h in final_pairs])
    pairs = list(zip(low_peaks, high_peaks))

    #Optional: uncomment these prints to see if the function is working in a proper way
    #print("DEBUG ─ cleaned low_peaks :", low_peaks)
    #print("DEBUG ─ cleaned high_peaks:", high_peaks)


    # control the sensibility (backward until 95 per cent)
    high_peaks = move_peak_to_plateau_direction(displacement, high_peaks, mode='high', max_search=500,relative_threshold=0.95)
    low_peaks = move_peak_to_plateau_direction(displacement, low_peaks, mode='low', max_search=500,relative_threshold=0.95)
    pairs = list(zip(low_peaks, high_peaks))

    return high_peaks, low_peaks, pairs

def check_concentric_phase(high_peaks, low_peaks, displacement):
    """
    Check concentric phase peaks for issues.

    Returns a list of trigger messages if any check fails.
    Returns None if all checks pass.
    """
    triggers = []

    # 1. Number of highs and lows must be equal
    if len(high_peaks) != len(low_peaks):
        triggers.append("Number of highs and lows is different")

    # 2. First peak
    if len(high_peaks) > 0 and len(low_peaks) > 0:
        if high_peaks[0] < low_peaks[0]:
            triggers.append("First peak is a high")

        # 3. Last peak
        if low_peaks[-1] > high_peaks[-1]:
            triggers.append("Last peak is a low")

    # 4. Interleaving
    combined = np.sort(np.concatenate([high_peaks, low_peaks]))
    for i in range(len(combined) - 1):
        if (combined[i] in low_peaks) == (combined[i + 1] in low_peaks):
            triggers.append("Peaks not interleaved properly")
            break

    # 5. Negative slope
    for lp, hp in zip(low_peaks, high_peaks):
        if displacement[hp] - displacement[lp] < 0:
            triggers.append(f"Negative slope between low={lp} and high={hp}")

    # 6. Less peaks (highs or lows) that what was expected
    if (len(high_peaks) < 5):
        triggers.append("number of High peaks problem")

    if (len(low_peaks) < 5):
        triggers.append("number of Low peaks problem")

    return triggers if triggers else None

def norm_concentric_phase(pairs, displacement, velocity, force, power, emg_ag, emg_ant, emg_ag_env, emg_ant_env, emg_container = 300):
    """
    Normalizes each cycle between 0 (low) and 1 (high).

    The analysis window includes an additional `emg_container` samples
    before the low point and after the high point, where available.

    Args:
        pairs (list): List of tuples containing the low and high indices
            of each cycle.
        displacement (array): Displacement signal.
        velocity (array): Velocity signal.
        force (array): Force signal.
        power (array): Power signal.
        emg_ag (array): Agonist EMG signal.
        emg_ant (array): Antagonist EMG signal.
        emg_ag_env (array): Agonist EMG envelope signal.
        emg_ant_env (array): Antagonist EMG envelope signal.
        emg_container (int): Number of additional samples to include
            before and after each cycle. Defaults to 300.

    Returns:
        list: A list of dictionaries, one per cycle, containing the
            normalized phase, corresponding signal segments, and cycle
            duration in seconds.

    Notes:
        The phase is calculated relative to the low and high indices,
        so it may extend below 0 or above 1 in the additional window
        included for EMG analysis.
    """
    cycles = []
    n_samples = len(displacement)
    for low, high in pairs:
        start = low - emg_container if low > emg_container else 0  #ms to have the emg signal
        end = high + emg_container if high + emg_container < n_samples else n_samples #ms
        duration_cycle = round((high - low)/1000, 3) #add_16_01
        displacement_cycle = displacement[start:end]
        velocity_cycle = velocity[start:end]
        force_cycle = force[start:end]
        power_cycle = power[start:end]
        emg_ag_cycle = emg_ag[start:end] #novo2
        emg_ant_cycle = emg_ant[start:end] #novo2
        emg_ag_env_cycle = emg_ag_env[start:end]
        emg_ant_env_cycle = emg_ant_env[start:end]
        idx = np.arange(start,end)
        phase = (idx- low)/(high-low) #we don't use start and end here because we are padding the concentric phase, not the window that includes the emg signal

        cycles.append({"phase": phase, "displacement": displacement_cycle, "velocity": velocity_cycle,
                       "force": force_cycle, "power": power_cycle,
                       "emg_ag": emg_ag_cycle, "emg_ant": emg_ant_cycle,
                       "duration_cycle" : duration_cycle,
                       "emg_ag_env": emg_ag_env_cycle, "emg_ant_env": emg_ant_env_cycle, "duration_cycle" : duration_cycle
                       })

    return cycles

def add_pairs(nome, pairs, txt_path):
    pairs_str = ";".join(f"{int(a)}-{int(b)}" for a, b in pairs)
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"{nome},{pairs_str}\n")



def run_analysis():
    """Run the analysis for all processed files."""

    files_list = load_file_list(FILE_EXTENSION, PROCESSED_ROOT)

    if not files_list:
        print(f"No files found in {PROCESSED_ROOT}")
        return

    for filepath in files_list:
        analyze_file(filepath)


def analyze_file(filepath):
    """Analyze one file at a time."""

    file_name = small_root_file(filepath)
    print("Processing:", file_name)

    # 1. Read the signals
    signals, fs, labels = read_columns_csv(filepath, fs=FS, keywords=None)

    # Apply the selected cut
    signals = segment_signal(signals, filepath)

    vel = signals["Velocity"].astype(float)
    loadcell = signals["LOADCELL"].astype(float)

    emg_ag, emg_ant, emg_ag_env, emg_ant_env = (emg_signal_by_filename(signals, filepath))

    # 2. Calculate physical quantities
    time = get_time(vel, fs)

    min_len = min(len(time), len(vel))
    time = time[:min_len]
    vel = vel[:min_len]

    displacement = get_distance(vel, time)
    force = get_force(loadcell)
    power = get_power(force, vel)

    # 3. Detect concentric phases
    high, low, pairs = concentric_phase(displacement)

    # 4. (Optional) Plot the complete signal
    #plot_grandezas(power,force,vel,displacement,time,file_name,emg_ag_env,emg_ant_env,emg_ag,emg_ant,high_peaks=high,low_peaks=low,)

    # 5. Concatenate the values that were found into pairs
    add_pairs(file_name, pairs, POSITION_CYCLES_FILE)

    # 6. (Optional) Check if the pairs were well implemented based on some caractheristics
    triggers = check_concentric_phase(high, low, displacement)
    # if triggers is not None:
    # print(f"Triggers for {filepath}")
    # for t in triggers:
    # print(t)

    # 7. Normalization of the concentric cycle
    cycles = norm_concentric_phase(pairs,displacement,vel,force,power,emg_ag,emg_ant,emg_ag_env,emg_ant_env)

    # 8. Calculate metrics for each cycle
    for n_cycle, cycle in enumerate(cycles):
        phase = cycle["phase"]
        velocity = cycle["velocity"]
        force = cycle["force"]
        power = cycle["power"]
        emg_ag = cycle["emg_ag"]
        emg_ant = cycle["emg_ant"]
        emg_ag_env= cycle["emg_ag_env"]
        emg_ant_env = cycle["emg_ant_env"]
        duration_cycle = cycle["duration_cycle"]
        displacement = cycle["displacement"]
        cycle_number = n_cycle + 1
        median_f_ag = median_frequency(emg_ag, fs)
        median_f_ant = median_frequency(emg_ant, fs)
        total_distance = total_measure(displacement)
        auc_force = area_under_curve(force)
        auc_power = area_under_curve(power)
        rpd = calculate_rpd(power)
        peaks, velocity_at_peak_power, force_at_peak_power = peaks_metrics(phase, velocity, force, power, emg_ag_env, emg_ant_env, rpd)

    # 8.1 Create a csv file with all the metrics calculated above
        peaks_on_file(peaks, median_f_ag, median_f_ant, file_name, cycle_number, duration_cycle, total_distance,
                      auc_power, auc_force, velocity_at_peak_power, force_at_peak_power)

    # 9. (Optional) Plot the normalized cycle
    #plot_all_concentric_phase(cycles, file_name)

    # 10. (Optional) Plot the perfomance metrics inside the concentric phase (0-100%)
    #plot_metrics_norm([cycles[2]], file_name, rpd)
