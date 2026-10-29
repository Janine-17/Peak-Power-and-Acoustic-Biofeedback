#-------------IMPORTS--------------#
import os
import re
import glob
import numpy as np
import pandas as pd
import json
from scipy.ndimage import gaussian_filter1d
from scipy.signal import sosfilt, butter, filtfilt
from scipy import integrate
from config import *
from Processing.EMG_Analysis.e_emg_analysis import small_root_file

#-------------FUNCTIONS-----------#

def load_file_list():
    """
    Finds all raw data files recursively inside RAW_ROOT.

    Returns:
        list[str]: A sorted list of file paths matching FILE_EXTENSION.
    """
    return sorted(glob.glob(os.path.join(str(RAW_ROOT), "**", FILE_EXTENSION), recursive=True))

def get_processed_path(file_path):
    """
    Generates the output file path while preserving the directory structure
    of the original file and appending '_processed' to its filename.

    Args:
        file_path (str): Path to the original raw data file.

    Returns:
        str: Output path inside PROCESSED_ROOT, preserving the relative
             directory structure and adding '_processed' before the extension.
    """
    relative_path = os.path.relpath(file_path, RAW_ROOT)
    base, ext = os.path.splitext(relative_path)

    return os.path.join(PROCESSED_ROOT, base + "_processed" + ext)

def extract_data_from_file(file_path):
    """
    Reads Opensignals (.txt with a JSON header) or simples files as:

        Nseq    LOADCELL    ACCX    ACCY ...

    Args:
        file_path (str): Path to the original raw data file.

    Returns:
        tuple:
            params (dict): Dictionary containing information about the file format.
            df (pd.DataFrame): Data read from the file.
    """
    params = None
    df = None

    try:
        if not file_path.endswith(".txt"):
            return None, None

        with open(file_path, "r") as f:
            lines = f.readlines()

        json_header_line = None
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("# {"):
                json_header_line = stripped.strip("# \n")
                break

        if json_header_line is not None:
            metadata = json.loads(json_header_line)

            device_key = list(metadata.keys())[0]
            column_labels = metadata[device_key]["column"]
            sensor_labels = metadata[device_key]["sensor"]

            #Rename the EMG and ACC sensors
            sensor_labels[-7:] = ["ACCX", "ACCY", "ACCZ",
                                  "EMG_BB", "EMG_TB", "EMG_RF", "EMG_BF"]

            labels = column_labels[:2] + sensor_labels
            params = metadata[device_key]

            # Extract data ( all fot he signals that do not start with "#"
            data_lines = [line.strip().split("\t")
                          for line in lines
                          if not line.startswith("#") and line.strip() != ""]

            df = pd.DataFrame(data_lines, columns=labels)
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        else:
            df = pd.read_csv(file_path, sep=r"\s+|\t", engine="python")

            if "Nseq" in df.columns and "nSeq" not in df.columns:
                df = df.rename(columns={"Nseq": "nSeq"})
            params = {"format": "simple_txt"}

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

    return params, df

def separate_signals(file_path, df, split_index1, split_index2):
    """
    Splits a RAW file into two signals based on row index.

    - Original file keeps rows [0 : split_index]
    - New RAW file gets rows [split_index : end]
    - New file has rep+1 in filename

    :return: df_signal1
    """

    df_signal1 = df.iloc[:split_index1].copy()
    df_signal2 = df.iloc[split_index2:].copy()

    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)

    match = re.search(r"rep(\d+)", base_name)
    if not match:
        raise ValueError(f"No rep found in filename: {base_name}")

    rep_number = int(match.group(1))
    new_rep = rep_number + 1

    new_base_name = re.sub(r"rep\d+", f"rep{new_rep}", base_name)
    new_file_path = os.path.join(dir_name, new_base_name)

    df_signal2.to_csv(new_file_path, sep="\t", index=False)

    return df_signal1

def change_axis(df, axis1, axis2):
    """
    Receives a DataFrame and swaps two axis columns.
    :return: corrected DataFrame
    """
    df[[axis1, axis2]] = df[[axis2, axis1]]
    return df


#-------- General Functions---------#

def get_time(data, fs):
    """
    Getting the time vector from a signal.
    :param data: np.array with the array to retrieve the time points from.
    :param fs: int with the sampling frequency of the signal.
    :return: np.array with the time points of each signal sample, of the same length of the signal array.
    """
    return np.arange(0, len(data)/fs, 1/fs)

def smooth_signal(signal, window_size=350):
    """
    Applies a simple moving average to smooth the input signal.

    Args:
        signal (array-like): Input signal to be smoothed.
        window_size (int): Number of samples used to calculate the moving average.
            Defaults to 350.

    Returns:
        np.ndarray: Smoothed signal with the same length as the input signal.
    """
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def butter_lowpass(cutoff, fs, order=5):
    """
    Create a Butterworth lowpass filter.

    Parameters:
    cutoff : float - cutoff frequency (Hz)
    fs     : float - sampling frequency (Hz)
    order  : int   - filter order

    Returns:
    b, a : filter coefficients
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs=1000, order=5):
    """
    Apply a Butterworth lowpass filter to a signal.

    Parameters:
    data   : array-like - input signal
    cutoff : float      - cutoff frequency (Hz)
    fs     : float      - sampling frequency (Hz)
    order  : int        - filter order

    Returns:
    y : array - filtered signal
    """
    data = data - np.mean(data)  # remove DC component
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, np.abs(data))     # apply filter with zero-phase
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Create a Butterworth bandpass filter.

    Parameters:
    lowcut : float - lower cutoff frequency (Hz)
    highcut : float - upper cutoff frequency (Hz)
    fs     : float - sampling frequency (Hz)
    order  : int   - filter order

    Returns:
    b, a : filter coefficients
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs=1000, order=3):
    """
    Apply a Butterworth bandpass filter to data.

    Parameters:
    data    : array-like - input signal
    lowcut  : float      - lower cutoff frequency (Hz)
    highcut : float      - upper cutoff frequency (Hz)
    fs      : float      - sampling frequency (Hz)
    order   : int        - filter order

    Returns:
    y : array - filtered signal
    """
    data = data - np.mean(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)

    return y


#---------- Loacell Signal-------------#
def transfer_weight(bit_value):
    """
    Function that transforms the bit values into mass in Kg based on a local calibration.
    :param bit_value: np.array with the bit values from the load cell sensor.
    :return: np.array with the mass values in kg.
    """
    return -0.003 * bit_value + 96.633

def process_loadcell(a):
    """
    Converts raw load cell measurements to Newtons and applies Gaussian
    filtering followed by moving average smoothing.

    Args:
        a (array-like): Raw load cell signal.

    Returns:
        np.ndarray: Filtered and smoothed load cell signal in Newtons.
    """
    loadcell_kg = transfer_weight(a)
    loadcell_N = loadcell_kg * 9.8
    loadcell_filt = gaussian_filter1d(loadcell_N, sigma=15)
    loadcell_smooth = smooth_signal(loadcell_filt, window_size=100)
    return loadcell_smooth


#---------- Acceleration Signal---------#
def getAccCalibration(axis:str):
    """
    Function that retrieves the Callibration values for the Accelerometer.
    For this we need to have a callibration file that contains the data when performing the callibration protocol.
    This procedure can be found at: https://support.pluxbiosignals.com/wp-content/uploads/2021/11/Accelerometer_ACC_Datasheet.pdf.
    This function works for tri-axial accelerometers.
    :return: tuple with Callibration values for the three axes.
    """
    calib_file = np.loadtxt(ACC_FILE)
    if axis == 'x':
        C_min = min(calib_file[:, 2])
        C_max = max(calib_file[:, 2])
    if axis == 'y':
        C_min = min(calib_file[:, 3])
        C_max = max(calib_file[:, 3])
    if axis == 'z':
        C_min = min(calib_file[:, 4])
        C_max = max(calib_file[:, 4])

    return C_min, C_max

def transfer_acc(data, axis):
    """
    This Function transforms the bit values into m.s^-2 units from the Accelerometer sensor.
    :param data: array with bit values from the ACC sensor.
    :param axis: str with the axes we want to transform.
    :return: array with the acceleration values in g-force for the specified axis.
    """
    C_min, C_max = getAccCalibration(axis)
    d = (2 * (data - C_min) / (C_max - C_min)) - 1
    return d *9.8

def gravitational_filter(acc_data: np.ndarray, fs: int) -> np.ndarray:
    """
    Brief
    -----
    Function to filter out the gravitational component of ACC signals using a 3rd order butterworth lowpass filter with
    a cuttoff frequency o f 0.3 Hz
    The implementation is based on:
    "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
    https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf
    Parameters
    ----------
    acc_data: 1-D or (MxN) array
         Where M is the signal length in samples and N is the number of signals / channels.
    fs: int
        The sampling frequency of the acc data.
    Returns
    -------
    gravity_data: 1-D or (MxN) array
        The gravitational component of each signal/channel contained in acc_data.
    """

    # define the filter
    order = 3
    f_c = 0.3
    filter = butter(order, f_c, fs=fs, output='sos')

    # copy the array
    gravity_data = acc_data.copy()

    # check the dimensionality of the input
    if gravity_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(gravity_data.shape[1]):
            # get the channel
            sig = acc_data[:, channel]

            # apply butterworth filter
            gravity_data[:, channel] = sosfilt(filter, sig)

    else:  # 1-D array

        gravity_data = sosfilt(filter, acc_data)

    return gravity_data

def process_acc(a, fs, axis):
    """
        Brief
        ----------
        Processes raw accelerometer data by converting it to physical units
        and applying bandpass filtering.

        Parameters
        ----------
        a: array
            Raw accelerometer signal.
        fs: int
            Sampling frequency (Hz).
        axis: str
            Accelerometer axis to be converted.

        Returns
        -------
        acc_filt: array
            Filtered accelerometer signal in m·s⁻².
        """
    acc_ms = transfer_acc(a, axis)
    acc_no_gravity = acc_ms - gravitational_filter(acc_ms, fs)
    acc_filt = butter_bandpass_filter(acc_no_gravity, 0.3, 10, fs, 3)
    return acc_filt

#------- Velocity Signal----------------#
def velocity_from_acc(data, time):
    """
    Function that computes velocity from the accelerometer data.
    :param data: np.array with the accelerometer data in m.s^-2
    :param time: np.array with the time stamps.
    :return: np.array with the derived velocity values.
    """
    return integrate.cumulative_trapezoid(data, time, initial=0)


#------- Electromyography Signal---------#
def tf_emg(values,n):
    """
    Function that tranforms raw EMG signal into physical quantities (mV)
    :param values: raw data
    :param n: ADC resolution (16 bits)
    :return: np.array with values in mV
    """
    vcc=3
    trans_value = (values/(2**n) -0.5)* (vcc/1000) * 1000
    return trans_value

def process_emg(emg_raw, fs, n_bits=16):
    """
    Processes the EMG signal:
    1) Converts ADC counts to mV using the function tf_emg.
    2) Applies a fourth order band-pass filter (20–499 Hz).

    Args:
        emg_raw (array-like): Raw EMG signal in ADC counts.
        fs (int): Sampling frequency in Hz.
        n_bits (int): ADC resolution in bits (default: 16).
    """
    emg_mv = tf_emg(emg_raw, n_bits)
    emg_filt = butter_bandpass_filter(emg_mv, 20, 499, fs, 4)

    return emg_filt

#---- Electromyography Envelope Signal------#
def process_emg_env(emg_raw, fs, n_bits=16):
    """
       Processes the EMG envelope signal:
       1) Converts ADC counts to mV using tf_emg.
       2) Applies a low-pass filter with a 4 Hz cutoff frequency.

       Args:
           emg_raw (array-like): Raw EMG signal in ADC counts.
           fs (int): Sampling frequency in Hz.
           n_bits (int): ADC resolution in bits (default: 16).

       Returns:
           np.ndarray: Filtered EMG signal representing the envelope.
       """
    emg_mv = tf_emg(emg_raw, n_bits)
    emg_filt = butter_lowpass_filter(emg_mv, cutoff=4, fs= fs, order=4)

    return emg_filt


def run_preprocessing():
    file_list = load_file_list()

    if not file_list:
        print(f"No files were found at: {RAW_ROOT}")
        return

    for file_path in file_list:
        file_name = small_root_file(file_path)
        print("Processing:", file_name)

        params, df = extract_data_from_file(file_path)

        if df is None:
            print(f"Não foi possível processar: {file_path}")
            continue

        # Optional: uncomment this block if some files
        # contain signals that need to be separated and that list is
        # inside of "FILES_TO_SPLIT" .

        #for key, (start, end) in FILES_TO_SPLIT.items():
        #    if key in file_path:
        #        df = separate_signals(
        #            file_path, df, int(start), int(end)
        #        )
        #        break


        # Optional: uncomment this block if some files
        # require an axis swap.
        #for key, (axis1, axis2) in FILES_WITH_AXIS_SWAP.items():
        #    if key in file_path:
        #        df = change_axis(df, axis1, axis2)
        #        break


        df["Time"] = get_time(df, FS)[:len(df)]

        # Process the loadlcell
        if "LOADCELL" in df.columns:
            df["LOADCELL"] = process_loadcell(
                df["LOADCELL"].values
            ).astype("float64")

        #Process ACC and velocity
        if "ACCY" in df.columns:
            df["ACCY"] = process_acc(
                df["ACCY"].values, FS, "y"
            ).astype("float64")

            df["Velocity"] = -1 * velocity_from_acc(
                df["ACCY"].values, df["Time"].values
            )
            df["Velocity"] -= np.mean(df["Velocity"])

        for axis in ["ACCX", "ACCZ"]:
            if axis in df.columns:
                df[axis] = process_acc(
                    df[axis].values, FS, axis[-1].lower()
                ).astype("float64")

        #Process EMG and calculate the envelope
        for col in ["EMG_BB", "EMG_TB", "EMG_RF", "EMG_BF"]:
            if col in df.columns:
                emg_filt = process_emg(df[col].values, FS)
                emg_env = process_emg_env(
                    df[col], FS, n_bits=16
                )

                df[col] = emg_filt
                df[col + "_ENV"] = emg_env

        # Index of all signal
        df["nSeq"] = range(len(df))

        #Sav the processed signal
        new_file = get_processed_path(file_path)
        os.makedirs(os.path.dirname(new_file), exist_ok=True)

        df.to_csv(new_file, sep="\t", index=False)

        #print(f"Saved at: {new_file}")

