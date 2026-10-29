
#------------IMPORTS-------------#
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, argrelmin, find_peaks, argrelextrema
import glob
import os
from Preprocessing.c_segmentation import segment_signal
from config import FS, POSITION_CYCLES_FILE, FILE_EXTENSION, PROCESSED_ROOT, OUTPUT_ROOT


#--------------FUNCTIONS------------#
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

def load_file_list(FILE_EXTENSION,DATA_ROOT):
    """
    Load all .txt files from the 'raw/' folder recursively.
    :return: sorted list of file paths
    """
    file_list = glob.glob(os.path.join(DATA_ROOT, "**", FILE_EXTENSION), recursive=True)
    return sorted(file_list)  # sort files alphabetically

def smooth_signal(signal, window_size=350):
    """Simple moving average smoothing."""
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def tkeo(x):
    '''
    this function uses the Teager–Kaiser Energy Operator to calculate the instantaneous energy of the signal
    :param x: it's the emg signal
    :return: energy vector of Teager-Kaiser. First and last samples are zero.
    '''
    x = x.astype(float)
    psi = np.zeros_like(x, dtype=float)
    psi[1:-1] = x[1:-1]**2 - x[0:-2]*x[2:] # formula ψ[n]=x[n]^2−x[n−1]x[n+1]
    return psi

def detect_onset_squat(s, cycles= None, plot =False, title=None):
    """
    Detects EMG onsets during squat cycles and plots the results if requested.

    Uses the appropriate functions to calculate the onsets and optionally
    displays a plot showing the concentric cycles and detected EMG onsets.

    Args:
        s (array-like): Input signal.
        cycles (list, optional): List of cycles. Defaults to None.
        plot (bool): Whether to display the plot. Defaults to False.
        title (str, optional): Title for the plot. Defaults to None.
    """
    # Step 1 - Getting the TKEO
    x_tkeo = tkeo(s)
    x_feat = np.abs(x_tkeo)
    # smoothing the tkeo  absolute envelope
    x_smooth = smooth_signal(x_feat, 25) ##

    # Step 2 - Computing the Hilbert Transform of the TKEO transform
    a_sig1 = hilbert(x_smooth) ##
    amplitude_envelope = np.abs(a_sig1)
    amp_env_smooth = smooth_signal(amplitude_envelope, 50)
    # Step 3- Hilbert transform on the amplitude envelope
    a_sig2 = hilbert(amp_env_smooth)
    amp_env_2 = abs(a_sig2)
    amp_env_2_smooth = gaussian_filter1d(amp_env_2, 80)
    # Step 4 Imaginary part
    imag_part = np.imag(a_sig2)
    imag_part_s = gaussian_filter1d(imag_part, 100)
    # Step 5 - Local minima from imag part
    onsets = argrelmin(imag_part_s, order=900)
    if plot:
        # Plotting the result
        plt.figure()
        plt.plot(s/max(s), label='EMG signal', color='black', alpha=0.3)
        #plt.plot(amp_env_2_smooth/max(amp_env_2_smooth), color='orange', label= 'Hilbert Envelope')
        plt.vlines(onsets, ymin=-1, ymax=1, label='Onsets', colors='red')
        if cycles is not None:
            for start, end in cycles:
                plt.axvline(start, color='blue',ymin=-1, ymax=1, linestyle='--')
                plt.axvline(end, color='black', ymin=-1, ymax=1, linestyle='--')
        if title is not None:
            plt.title(title)
        plt.legend()
        plt.show()

    onsets_idx = onsets[0].tolist()
    return amp_env_2_smooth, onsets_idx

def detect_onset_bp(s, cycles = None, plot=False, agonist=True, title=None):
    """
    Detects EMG onsets during bench press cycles and plots the results if requested.
    It has a different variation depending on the muscle that is being analyzed

    Uses the appropriate functions to calculate the onsets and optionally
    displays a plot showing the concentric cycles and detected EMG onsets.

    Args:
        s (array-like): Input signal.
        cycles (list, optional): List of cycles. Defaults to None.
        plot (bool): Whether to display the plot. Defaults to False.
        agonist (bool): Whether is an agonist or antagonist analysis
        title (str, optional): Title for the plot. Defaults to None.
    """
    # Step 1- Getting the analytical components from the TKEO Evelope
    x_tkeo = tkeo(s)
    x_feat = np.abs(x_tkeo)
    # smoothing the tkeo  absolute envelope
    x_smooth = smooth_signal(x_feat, 25)
    a_sig1 = hilbert(x_smooth)
    amplitude_envelope = np.abs(a_sig1)
    amp_env_smooth = smooth_signal(amplitude_envelope, 50)
    # Step 3- Hilbert transform on the amplitude envelope
    a_sig2 = hilbert(amp_env_smooth)
    amp_env_2 = abs(a_sig2)
    amp_env_2_smooth = gaussian_filter1d(amp_env_2, 80)
    # Getting the phase
    ins_phase = np.unwrap(np.angle(a_sig1))
    ins_phase_s = gaussian_filter1d(ins_phase, sigma=150)

    imag_part = np.imag(a_sig1)
    imag_part_s = gaussian_filter1d(imag_part, sigma=100)

    # Finding local minima of the smoothed envelope (beginning of eccentric contraction )
    mini_phase = argrelmin(ins_phase_s, order=900)
    mini_imag = argrelmin(imag_part_s, order=900)

    n = min(len(mini_phase[0]), len(mini_imag[0]))
    mini = (mini_phase[0][:n] + mini_imag[0][:n]) // 2
    if plot:
        plt.figure()
        plt.plot(s / max(s), color='green', alpha=0.3, label='EMG Signal')
        plt.plot(ins_phase_s / max(ins_phase_s), label='Instantaneous Phase', color='orange', alpha=0.5)
        plt.plot(imag_part_s / max(imag_part_s), label='Imaginary Part', color='purple', alpha=0.5)
        plt.plot(amp_env_2_smooth / max(amp_env_2_smooth), color = 'blue', label='Envelope', alpha=0.3)

        plt.vlines(mini, -1, 1, colors='r', label='EMG Onset')
        plt.xlabel('Samples')
        plt.ylabel('EMG (normalized by max peak)')

        if cycles is not None and len(cycles) > 0:
            #legend
            plt.axvline(cycles[0][0], color='blue', linestyle='--', label="Beg Conc Phase")
            plt.axvline(cycles[0][1], color='black', linestyle='--', label="End Conc Phase")

            for start, end in cycles:
                plt.axvline(start, color='blue', linestyle='--')
                plt.axvline(end, color='black', linestyle='--')

        if title is not None:
            plt.title(title)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        plt.show()

    if not agonist:
        return ins_phase_s, mini_phase[0].tolist()
    else:
        return ins_phase_s, mini.tolist()

def load_cycles_from_txt(txt_path):
    """
    Loads the concentric cycles from a text file for EMG analysis.

    Args:
        txt_path (str): Path to the text file containing the cycle information.

    Returns:
        dict: Dictionary mapping each file path to a list of cycles,
            where each cycle is represented by a (start, end) tuple.
    """
    cycles_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            rel_path = os.path.normpath(parts[0])
            cycles = []

            if len(parts) > 1:
                for window in parts[1].split(';'):
                    start, end = map(int, window.split('-'))
                    cycles.append((start, end))

            cycles_dict[rel_path] = cycles

    return cycles_dict

def get_paths_from_position_txt(txt_path):
    """
    Extracts file paths from a text file.

    Args:
        txt_path (str): Path to the text file containing file paths.

    Returns:
        list[str]: List of normalized file paths.
    """
    paths = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_path = line.split(',')[0]
            paths.append(os.path.normpath(raw_path))
    return paths

def save_onsets_txt(output_path, results):
    """
    Saves cycle information and detected EMG onsets to a text file.

    Args:
        output_path (str): Path to the output text file.
        results (list): List of dictionaries containing file paths, cycles,
            valid agonist and antagonist onsets, and invalid onsets.

    Notes:
        Each line contains the file path, cycle boundaries, valid onsets,
        and invalid onsets for both agonist and antagonist muscles.
        Missing cycle information is saved as an empty field.
    """
    with open(output_path, "w") as f:
        for r in results:
            # cycles
            if r["cycles"] is not None:
                cycles_str = ";".join([f"{s}-{e}" for s, e in r["cycles"]])
            else:
                cycles_str = ""

            # good onsets
            ag_good = ";".join(map(str, r["agonist_onsets"]))
            ant_good = ";".join(map(str, r["antagonist_onsets"]))

            # bad onsets
            ag_bad = ";".join(map(str, r.get("bad_ag", [])))
            ant_bad = ";".join(map(str, r.get("bad_ant", [])))

            line = (
                f"{r['path']},"
                f"cycles:{cycles_str},"
                f"agonist_onsets:{ag_good},"
                f"antagonist_onsets:{ant_good},"
                f"agonist_bad:{ag_bad},"
                f"antagonist_bad:{ant_bad}\n"
            )

            f.write(line)

def validate_onsets_interactive(signal, onsets, title="", plot=True):
    """
    Click on onsets to mark/unmark them as BAD.
    BAD onsets appear in red.
    Returns: good_onsets, bad_onsets
    """

    if not plot:
        return list(onsets), []

    fig, ax = plt.subplots()
    ax.plot(signal / np.max(signal), color='black', alpha=0.4)

    lines = {}
    bad_onsets = set()

    for o in onsets:
        line = ax.axvline(o, color='green', linewidth=1)
        lines[o] = line

    ax.set_title(title)

    def on_click(event):
        if event.xdata is None:
            return

        x = int(event.xdata)
        closest = min(onsets, key=lambda o: abs(o - x))

        if closest in bad_onsets:
            bad_onsets.remove(closest)
            lines[closest].set_color('green')
            lines[closest].set_linewidth(1)
        else:
            bad_onsets.add(closest)
            lines[closest].set_color('red')
            lines[closest].set_linewidth(2)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    good_onsets = [o for o in onsets if o not in bad_onsets]
    return good_onsets, list(bad_onsets)

def run_emg_analysis():
    """
    Runs the EMG analysis pipeline, ask the user if wants to see the plots, detects and validats muscle activation
    onsets and saving the results to an output file.
    """
    show_plots = input("Show EMG plots? (y/n): ").strip().lower() == "y"
    results = []

    valid_paths = get_paths_from_position_txt(POSITION_CYCLES_FILE)
    cycles_dict = load_cycles_from_txt(POSITION_CYCLES_FILE)

    files_list = load_file_list(FILE_EXTENSION, PROCESSED_ROOT)

    for filepath in files_list:
        rel_path = small_root_file(filepath)
        if rel_path is None:
            continue

        rel_path = os.path.normpath(rel_path)

        if rel_path not in valid_paths:
            continue

        print("Processing EMG:", rel_path)

        signals, fs, labels = read_columns_csv(filepath,fs=FS,keywords=None,)

        # Apply the selected cut
        signals = segment_signal(signals, filepath)

        emg_ag, emg_ant, emg_ag_env, emg_ant_env = (emg_signal_by_filename(signals, filepath))

        cycles = cycles_dict.get(rel_path, None)

        if cycles is None:
            print(f"No cycles found for: {rel_path}")
            continue

        ag_good = []
        ant_good = []
        ag_bad = []
        ant_bad = []

        if "SQ" in rel_path.upper():
            _, ag_onsets = detect_onset_squat(emg_ag,cycles=cycles,plot=show_plots,title=f"{rel_path} — Agonist",)

            ag_good, ag_bad = validate_onsets_interactive(emg_ag,ag_onsets,plot=show_plots, title=f"{rel_path} — Agonist (click bad onsets)",)

            _, ant_onsets = detect_onset_squat(emg_ant,cycles=cycles,plot=show_plots,title=f"{rel_path} — Antagonist",)

            ant_good, ant_bad = validate_onsets_interactive(emg_ant,ant_onsets,plot=show_plots, title=f"{rel_path} — Antagonist (click bad onsets)",)

        elif "BP" in rel_path.upper():
            _, ag_onsets = detect_onset_bp(emg_ag,cycles=cycles,plot=show_plots,agonist=True,title=f"{rel_path} — Agonist",)

            ag_good, ag_bad = validate_onsets_interactive(emg_ag,ag_onsets, plot=show_plots, title=f"{rel_path} — Agonist (click bad onsets)",)

            _, ant_onsets = detect_onset_bp(emg_ant, cycles=cycles,plot=show_plots, agonist=False,title=f"{rel_path} — Antagonist",)

            ant_good, ant_bad = validate_onsets_interactive( emg_ant,ant_onsets, plot=show_plots, title=f"{rel_path} — Antagonist (click bad onsets)",)
        else:
            print(f"Unsupported exercise: {rel_path}")
            continue

        results.append({"path": rel_path,"cycles": cycles,"agonist_onsets": ag_good,"antagonist_onsets": ant_good,"bad_ag": ag_bad,"bad_ant": ant_bad,})

    output_txt = OUTPUT_ROOT / "onsets_emg_results.txt"
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    save_onsets_txt(str(output_txt), results)

    print(f"EMG analysis complete. Results saved to: {output_txt}")
    #print(f"Number of Files processed: {len(results)}")