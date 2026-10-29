#----------------IMPORTS---------------------#

from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks, welch
import numpy as np
import re, csv, os
from config import METRICS_FILE

#--------------FUNCTIONS------------------------#

def get_distance (v, time):
    """

    :param v: velocity signal
    :param time: time signal
    :return: distance signal
    """
    v_zero_mean = v
    return cumulative_trapezoid(v_zero_mean,time)

def get_force(loadcell):
    force = loadcell
    return force

def get_power(force, velocity):
    power = force * velocity
    return power

def get_time(data, fs):
    """
    Getting the time vector from a signal.
    :param data: np.array with the array to retrieve the time points from.
    :param fs: int with the sampling frequency of the signal.
    :return: np.array with the time points of each signal sample, of the same length of the signal array.
    """
    return np.arange(0, len(data)/fs, 1/fs)

def total_measure(signal):
    """
    total distance covered by the interval
    :param signal: distance signal
    :return: distance percorrida along the cycle based on the acc signal
    """
    distance = np.abs(signal[-1] - signal[0])
    distance = round(distance, 3)
    return distance

def calculate_rpd(power, fs=1000):
    """
    Measure the peak of Rate of Power Development
    """
    power = np.array(power)

    dP_dt = np.diff(power) * fs

    return dP_dt

def median_frequency(signal_emg, fs):
    """
    Calculates the median frequency of an EMG signal using Welch's method.

    Args:
        signal_emg (array-like): EMG signal.
        fs (float): Sampling frequency in Hz.

    Returns:
        float: Median frequency in Hz, rounded to two decimal places.
    """
    frequencies, psd = welch(signal_emg, fs, nperseg=len(signal_emg))
    cumulative_power = np.cumsum(psd) #cumulative sum
    total_power = cumulative_power[-1]
    median_f = frequencies[np.where(cumulative_power >= total_power/2)[0][0]]
    median_f = round(median_f, 2)

    return median_f

def area_under_curve(y):
    """
    gets the area under the curve receivingthe signals.
    Useful for force, power
    :param y: signal with indexs between the cycle
    :return: area under the curve
    """
    auc = np.trapezoid(y)
    auc = round(auc, 3)
    return auc

def get_info_file_name(file_name):
    """
    Extracts the participant ID, session, exercise type, repetition number,
    and group from the file path.

    Args:
        file_name (str): File path containing participant and exercise information.

    Returns:
        tuple: A tuple containing:
            - id_num (str or None): Participant ID.
            - session (str or None): Session number.
            - ex_type (str or None): Exercise type (BP, SQ, or DR).
            - rep_n (str or None): Repetition number.
            - group (str or None): Participant group.
    """
    parts = [p for p in file_name.split(os.sep) if p]
    filename = parts[-1]
    filename_upper = filename.upper()
    id_num = None
    session = None
    ex_type = None
    rep_n = None
    group = None

    for part in parts[:-1]:
        part_upper = part.upper()
        if part_upper.startswith("ID"):
            id_num = part[2:]
        elif part_upper.startswith("T"):
            session = part[1:]
        elif part_upper.startswith("DADOS"):
            group = part[6:]

    if ex_type is None:
        match = re.search(r'_(BP|SQ|DR)_', filename)
        if match:
            ex_type = match.group(1)

    # Special case: baseline or final files
    if "Baseline" or "Final" in filename_upper:
        session = "0"

        # The exercise type is specified in the filename
        match = re.search(r"(?:^|_)(BP|SQ|DR)(?:_|$)", filename_upper)
        if match:
            ex_type = match.group(1)

    else:
        # Determine the exercise type for other files
        match = re.search(r"_(BP|SQ|DR)_", filename_upper)
        if match:
            ex_type = match.group(1)

    match = re.search(r'rep(\d+)', filename)
    if match:
        rep_n = match.group(1)

    return id_num, session, ex_type, rep_n, group

def peaks_metrics(phase, velocity, force, power, emg_ag_env, emg_ant_env, rpd):
    """
    Calculates peak-related metrics for a cycle.

    Args:
        phase (array-like): Normalized cycle phase.
        velocity (array-like): Velocity signal.
        force (array-like): Force signal.
        power (array-like): Power signal.
        emg_ag_env (array-like): Agonist EMG envelope signal.
        emg_ant_env (array-like): Antagonist EMG envelope signal.
        rpd (float): Relative phase duration or other parameter used
            in the peak calculations.

    Returns:
        tuple: A tuple containing:
            - peaks: Calculated peak metrics.
            - velocity_at_peak_power: Velocity at the peak power.
            - force_at_peak_power: Force at the peak power.
    """
    peaks =[]

    s = np.searchsorted(phase, 0, side='left')
    e = np.searchsorted(phase, 1, side='right')

    distance_points = e - s
    #another possible approach
    #distance_points = len(phase)

    velocity_idx, _ = find_peaks(velocity[s:e], distance=distance_points)
    force_idx, _ = find_peaks(force, distance=distance_points)

    power_idx, _ = find_peaks(power[s:e], distance=distance_points)

    emg_ag_env_idx, _ = find_peaks(emg_ag_env[s:e], distance=distance_points)
    emg_ant_env_idx, _ = find_peaks(emg_ant_env[s:e], distance=distance_points)

    rpd_idx, _ = find_peaks(rpd[s:e], distance=distance_points)



    velocity_idx_cycle = velocity_idx
    force_idx_cycle = force_idx -s
    power_idx_cycle = power_idx
    emg_ag_env_idx_cycle = emg_ag_env_idx
    emg_ant_env_idx_cycle = emg_ant_env_idx
    rpd_idx_cycle = rpd_idx #novo



    velocity_idx = velocity_idx + s
    force_idx = force_idx
    power_idx = power_idx + s
    emg_ag_env_idx = emg_ag_env_idx + s
    emg_ant_env_idx = emg_ant_env_idx +s
    rpd_idx = rpd_idx + s

    if len(power_idx) == 0:
        velocity_at_peak_power = np.nan
        force_at_peak_power = np.nan
    else:
        velocity_at_peak_power = round(velocity[power_idx[0]], 3)
        force_at_peak_power = round(force[power_idx[0]], 3)


    peaks.append({
        "velocity": list(zip(velocity_idx, velocity_idx_cycle, phase[velocity_idx],velocity[velocity_idx])),
        "force": list(zip(force_idx, force_idx_cycle, phase[force_idx], force[force_idx])),
        "power": list(zip(power_idx, power_idx_cycle, phase[power_idx],power[power_idx])),
        "emg_ag_env": list(zip(emg_ag_env_idx, emg_ag_env_idx_cycle, phase[emg_ag_env_idx], emg_ag_env[emg_ag_env_idx])),
        "emg_ant_env": list(zip(emg_ant_env_idx , emg_ant_env_idx_cycle, phase[emg_ant_env_idx], emg_ant_env[emg_ant_env_idx])),
        "rpd": list(zip(rpd_idx, rpd_idx_cycle, phase[rpd_idx], rpd[rpd_idx]))
    })
    return peaks, velocity_at_peak_power, force_at_peak_power

def peaks_on_file(peaks,  median_f_ag, median_f_ant, file_name, cycle_number, duration_cycle,
                  total_distance, auc_power, auc_force, velocity_at_peak_power, force_at_peak_power):
    """
    function that receives the peaks values of each metric per cycle and add it to a .csv file from all cycles
    Order: File_Name, Cycle number, Velocity, Force, Power,..., Cycle number, Velocity, Force, Power.
    :param peaks:
    :return: table_performance_metrics_values.csv
    """
    os.makedirs(METRICS_FILE.parent, exist_ok=True)
    csv_file = str(METRICS_FILE)
    file_exists = os.path.exists(csv_file)

    id, session, ex_type, rep_n, group = get_info_file_name(file_name)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["File Name", "ID", "Group", "Session", "Exercise", "Repetition", "Cycle Number", "Duration of Cycle (s)",
                             "Peak Power (W)","Phase Peak Power (%)", "Time Peak Power (s)",
                             "Peak Force (N)", "Phase Peak Force (%)", "Time Peak Force (s)",
                             "Peak Velocity (m/s)", "Phase Peak Velocity (%)", "Time Peak Velocity (s)",
                             "Peak Env EMG Agonist (mV)", "Phase Peak Env EMG Agonist (%)" , "Time Peak Env Agonist (s)",
                             "Peak Env EMG Antagonist (mV)", "Phase Peak Env EMG Antagonist(%)" , "Time Peak Env Antagonist (s)",
                              "Median Freq Agonist (Hz)", "Median Freq Antagonist (Hz)", "Phase Shift PP and PV (%)", "Total Distance (m)",
                             "Impulse (N/s)", "Total Concentric Work (J)","V(P=Max) (m/s)", "F(P=Max) (N)", "Peak RPD (W/s)", "Phase Peak RPD (%)", "Time Peak RPD (s)" ])

        power_sec = power_perc = power_value = np.nan
        force_sec = force_perc = force_value = np.nan
        velocity_sec = velocity_perc = velocity_value = np.nan
        emg_ag_env_sec = emg_ag_env_perc = emg_ag_env_value = np.nan
        emg_ant_env_sec = emg_ant_env_perc = emg_ant_env_value = np.nan
        rpd_sec = rpd_perc = rpd_value = np.nan


        try:
            _,power_sec, power_perc, power_value = peaks[0]["power"][0]
        except Exception:
            pass
        try:
            _,force_sec, force_perc, force_value = peaks[0]["force"][0]
        except Exception:
            pass
        try:
            _,velocity_sec, velocity_perc, velocity_value = peaks[0]["velocity"][0]
        except Exception:
            pass
        try:
            _,emg_ag_env_sec, emg_ag_env_perc, emg_ag_env_value = peaks[0]["emg_ag_env"][0]
        except Exception:
            pass
        try:
            _,emg_ant_env_sec, emg_ant_env_perc, emg_ant_env_value = peaks[0]["emg_ant_env"][0]
        except Exception:
            pass

        try:
            _,rpd_sec, rpd_perc, rpd_value = peaks[0]["rpd"][0] #novo
        except Exception:
            pass

        power_value = round(power_value,3)
        force_value = round(force_value,3)
        velocity_value = round(velocity_value,3)
        emg_ag_env_value = round(emg_ag_env_value, 3)
        emg_ant_env_value = round(emg_ant_env_value,3)
        rpd_value = round(rpd_value,3)

        power_sec = round(power_sec/1000, 3)
        force_sec = round(force_sec / 1000, 3)
        velocity_sec = round(velocity_sec/1000, 3)
        emg_ag_env_sec = round(emg_ag_env_sec / 1000, 3)
        emg_ant_env_sec = round(emg_ant_env_sec / 1000, 3)
        rpd_sec = round(rpd_sec/1000,3)

        power_perc = round(power_perc*100, 3)
        force_perc = round(force_perc * 100, 3)
        velocity_perc = round(velocity_perc*100, 3)
        emg_ag_env_perc = round(emg_ag_env_perc*100, 3)
        emg_ant_env_perc = round(emg_ant_env_perc*100, 3)
        rpd_perc = round(rpd_perc*100,3)

        delta_pv_pp = round((velocity_perc-power_perc),2)


        writer.writerow([file_name, id, group, session, ex_type, rep_n, cycle_number, duration_cycle,
                         power_value, power_perc,power_sec,
                         force_value, force_perc, force_sec,
                         velocity_value, velocity_perc, velocity_sec,
                         emg_ag_env_value, emg_ag_env_perc, emg_ag_env_sec,
                         emg_ant_env_value, emg_ant_env_perc, emg_ant_env_sec,
                         median_f_ag, median_f_ant, delta_pv_pp, total_distance,
                         auc_power, auc_force, velocity_at_peak_power, force_at_peak_power, rpd_value, rpd_perc, rpd_sec])
    return
