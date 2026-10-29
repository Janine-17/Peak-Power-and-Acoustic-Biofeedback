# ASTROPOWER
## Project Overview
This repository contains the Python scripts developed to process and analyse kinetic, kinematic, and electromyographic recordings collected during the ASTROPOWER project's study. 
The dataset will be publicly available on Zenodo (https://zenodo.org/records/22960690) and the processing pipeline was developed as part of a Master Thesis

The intervention study aimed to investigate the effects of acoustic biofeedback (ABF) compared with conventional gym music (GM) on muscle power production in older males and females over time. Specifically, the study examined changes in muscle power throughout the training intervention and the impact of ABF on the progression of muscle power with training. Additionally, the study explored whether peak power could be predicted from neuromuscular characteristics.

ASTROPOWER is a European Space Agency (ESA)-funded project under the PRODEX programme that explores the parallels between the physiological effects of ageing on Earth and exposure to microgravity in space. 
The project focuses particularly on the decline in skeletal muscle power-generating capacity and investigates the potential of electromyography-based acoustic biofeedback as a strategy to enhance exercise performance and support countermeasures against neuromuscular deterioration.

The processing pipeline evaluates mechanical performance and neuromuscular responses during bench press and squat exercises performed under ABF and GM conditions. 
It includes signal calibration, filtering, trimming, concentric phase detection, calculation of performance and neuromuscular metrics,
electromyographic analysis, and the extraction of temporal and mechanical performance metrics. It also includes the regression linear models parameters that predict the Peak Power.

The scripts are designed to process the standardized ASTROPOWER dataset and generate diagnostic visualizations and structured CSV and TXT files containing repetition-level metrics for subsequent statistical analyses.

## Dataset Structure

The main folder, **Data Acoustic BioPower**, contains four data folders:

- **Data_Baseline**: Baseline measurements for each participant.
- **Data_ABF**: Experimental data collected during the ABF condition.
- **Data_GM**: Experimental data collected during the GM condition.
- **Data_Endline**: Endline measurements for each participant.

Baseline and endline measurements were collected using loads based on percentages of each participant's previously determined one-repetition maximum (1-RM). Loads ranged from 30% to 70% of 1-RM for the bench press (BP) and from 40% to 80% for the squat (SQ).

The experimental data are organized by participant ID and training session (T1–T8).


### Folder Organization

```text
Data Acoustic BioPower/
├── Data_Baseline/
│   ├── ID1/
│   └── ...
├── Data_ABF/
│   ├── ID1/
│   │   ├── T1/
│   │   ├── T2/
│   │   └── ... T8/
│   └── ...
├── Data_GM/
│   ├── ID2/
│   │   ├── T1/
│   │   └── ... T8/
│   └── ...
├── Data_Endline/
│   ├── ID1/
│   └── ...
```

### Data Format
The recordings are stored in plain-text (.txt) files containing eight whitespace-separated columns. Each file includes a single header row identifying the recorded signals, followed by the corresponding samples.
Each recording contains the following columns:

| Column | Description |
|---|---|
| `Nseq` | Sequence number. |
| `LOADCELL` | Load cell sensor data. |
| `ACCx` | Accelerometer data along the X-axis. |
| `ACCy` | Accelerometer data along the Y-axis. |
| `ACCz` | Accelerometer data along the Z-axis. |
| `EMG_BB` | Electromyography (EMG) data from the Biceps Brachii. |
| `EMG_TB` | EMG data from the Triceps Brachii. |
| `EMG_RF` | EMG data from the Rectus Femoris. |
| `EMG_BF` | EMG data from the Biceps Femoris. |

All physiological and biomechanical signals are provided in their original, unprocessed digital format (raw acquisition values). Consequently, sensor-specific transfer functions are required to obtain the corresponding physical units before analysis.
These signals were acquired at a sampling frequency of 1000 Hz, as specified in the config.py module.

The processing pipeline automatically selects the appropriate EMG channel according to the exercise: `EMG_BB` and `EMG_TB` for bench press; `EMG_RF` and `EMG_BF` for squat.

## Repository Strcuture

The project is organized into modules, each responsible for a specific step of the signal-processing pipeline.

| Module / File | Description |
|---|---|
| `Preprocessing/a_preprocessing.py` | Preprocesses the raw signals. |
| `Preprocessing/b_cutting_interface.py` | Provides the interface for selecting signal segments. |
| `Preprocessing/c_segmentation.py` | Segments the signals using the selected boundaries. |
| `Processing/Mechanical_Performance/a_analysis.py` | Performs mechanical signal analysis. |
| `Processing/Mechanical_Performance/b_signal_performance_metrics.py` | Calculates mechanical performance metrics. |
| `Processing/Mechanical_Performance/c_cleaning_performance_metrics.py` | Cleans and filters mechanical performance metrics. |
| `Processing/Mechanical_Performance/d_aggregation_performance_metrics.py` | Aggregates mechanical performance metrics and calculates summary statistics. |
| `Processing/Mechanical_Performance/plots.py` | Visualization utilities for the project. |
| `Processing/EMG_Analysis/e_emg_analysis.py` | Detects and analyzes EMG onsets. |
| `Processing/EMG_Analysis/f_emg_cycle_onset_association.py` | Associates detected EMG onsets with their corresponding movement cycles. |
| `Processing/EMG_Analysis/g_emg_temporal_metrics.py` | Calculates temporal metrics based on EMG onset data. |
| `Processing/EMG_Analysis/h_emg_complete_metrics.py` | Calculates the complete set of EMG metrics. |
| `config.py` | Defines file names and project paths used throughout the pipeline. |
| `main.py` | Runs the different modules in the correct order. |
| `output/` | Stores the `.txt` and `.csv` files generated during processing. |

Each module can be run as part of the main pipeline, which is controlled by `main.py`. The main script acts as the central entry point and calls the required modules in sequence.

The `config.py` file centralizes file names and paths, making it easier to manage input and output locations without changing paths throughout the codebase.

## Installation
This project was developed using Python 3.12.
After downloading or cloning the repository, open a terminal in the project directory and install the required Python packages using:

'pip install -r requirements.txt'
This command installs the dependencies specified in requirements.txt.

## Configuration

Before running the processing pipeline, users should check the following settings:

- **Raw data directory:** Make sure that the raw data is stored in the directory specified in `config.py`. If not, move the data to the specified directory or update the path in the configuration file.

- **Appendix files:** All additional files required by the pipeline are stored in the `appendix/` folder. These include:
  - `ACC_calib.txt`: Accelerometer calibration data.
  - `bad_signals.txt`: List of signals to be ignored during processing.
  - `pre_cut_marks`: File containing predefined cut boundaries for most signals, allowing them to be divided into smaller segments.

- **Signal segmentation:** The pipeline uses the predefined cuts in `pre_cut_marks`. If users prefer to define their own cuts, they can perform the segmentation manually. A new `cut_marks` file will then be created.

- **User input:** At certain stages, the pipeline requires user input to proceed. Follow the instructions displayed in the terminal.

## Usage
Run the  `main.py` script to start the processing pipeline:
python main.py

## Data Quality and Exclusion Criteria
The recordings included in the dataset are provided in their original, unprocessed format, without interpolation or correction of missing values. This allows users to independently select and apply missing-data handling strategies according to their analytical objectives.

## Citation
If you use or adapt the Python scripts provided in this repository, please acknowledge the ASTROPOWER project and cite the repository and associated scientific resources.

## Data Availability 
"Acoustic BioPower Training in Aging Dataset" available on Zenodo: https://zenodo.org/records/22960690
