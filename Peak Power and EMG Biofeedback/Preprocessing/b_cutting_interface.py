#---------------------IMPORTS----------------------#
import glob
import csv
import pandas as pd
from matplotlib.backend_bases import MouseButton
import shutil
from config import *
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button


#--------Progress Functions and Markdowns----------#
def load_file_list():
    """
    Finds all raw data files recursively inside RAW_ROOT.

    Returns:
        list[str]: A sorted list of file paths matching FILE_EXTENSION.
    """
    return sorted(glob.glob(os.path.join(PROCESSED_ROOT, "**", FILE_EXTENSION), recursive=True))

def save_progress(index):
    """
    Saves the current processing progress to a file.

    Args:
        index (int): Index of the current processing position.
    """
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(index))

def load_progress():
    """
    Load the progress of the cutting activity based on the last activity
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_mark(filename, start, end):
    """
    Saves the selected cut boundaries for a processed signal to the cut marks CSV file.

    Args:
        filename (str): Path to the processed signal file.
        start (int): Starting sample index of the cut.
        end (int): Ending sample index of the cut.

    Notes:
        Creates the output directory if it does not exist.
        Stores the file path relative to PROCESSED_ROOT, along with the
        start and end sample indices. Writes the CSV header if the file
        does not already exist.
    """
    os.makedirs(CUT_MARKS_FILE.parent, exist_ok=True)

    write_header = not CUT_MARKS_FILE.exists()
    rel_path = os.path.relpath(filename, PROCESSED_ROOT)

    with open(CUT_MARKS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["filepath", "start_sample", "end_sample"])
        writer.writerow([rel_path, start, end])

    print(f" The information was saved: {rel_path} | start={start}, end={end}")


#--------------Graphic Interface------------------#
class CutGUI:
    """
    This class allows opening the files and choose where to do the trimming of the signals.
    Left click on the mouse - defines the beggining of the cut
    Right click on the mouse - defines the ending of the cut
    Back button - allows to go back to the previous signal and change the position of the cutting edge
    Save button - saves the 2 index (beggining and end of the cut) and moves to the next signal

    It saves the marks in the file cut_marks.csv and add a plot of the signals with the cutting marks in
    the same folder as the signal.

    """
    def __init__(self, file_list):

        self.file_list = file_list
        self.idx = load_progress()
        self.load_file()

    def load_file(self):
        plt.close("all")
        self.current_file = self.file_list[self.idx]

        print(f"\n📂 Opening: {self.current_file}")
        self.df = pd.read_csv(self.current_file, sep="\t")

        if 'Velocity' not in self.df.columns:
            print("⚠ Velocity was not found in this file")

        selected_channels = ['LOADCELL', 'EMG_BB', 'ACCY', 'EMG_TB',
                             'Velocity', 'EMG_RF', 'ACCX', 'EMG_BF', 'ACCZ']

        available_channels = [ch for ch in selected_channels if ch in self.df.columns]
        signals = self.df[available_channels].values
        self.n_channels = len(available_channels)

        ncols = 2 #to have the subplots in 2 columns
        nrows = (self.n_channels + 1) // 2
        self.fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.5), sharex=True)

        axes = axes.flatten() if self.n_channels > 1 else [axes] #guarantee that the axes is a list

        filename_only = os.path.basename(self.current_file) #titulo
        filename_only = os.path.splitext(filename_only)[0]
        self.fig.suptitle(f"Analizing: {filename_only}", fontsize=16)

        for i, ch in enumerate(available_channels):
            ax = axes[i]

            # 1) mains signals (ACC, loadcell, velcoity...)
            ax.plot(self.df[ch].values, label=ch)

            # 2) if it is a EMG signal, plot the envelope on top of that
            env_col = ch + "_ENV"
            if env_col in self.df.columns:
                ax.plot(self.df[env_col].values,
                        linestyle="-",
                        linewidth=2.0,
                        alpha=0.9,
                        label=env_col)

            ax.set_ylabel(ch)
            ax.set_title(ch)

            # Legend (only when there is an envelope)
            if env_col in self.df.columns:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

        for j in range(self.n_channels, len(axes)):
            axes[j].axis('off')

        self.axes = axes
        self.start, self.end = 0, signals.shape[0]


        self.start_lines = []
        self.end_lines = []

        for i, ax in enumerate(self.axes):
            if i < self.n_channels:
                sl = Line2D([self.start, self.start], ax.get_ylim(), color="red")
                el = Line2D([self.end, self.end], ax.get_ylim(), color="green")
                ax.add_line(sl)
                ax.add_line(el)
                self.start_lines.append(sl)
                self.end_lines.append(el)
            else:
                self.start_lines.append(None)
                self.end_lines.append(None)

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        ax_save = plt.axes([0.85, 0.01, 0.1, 0.05])
        self.button_save = Button(ax_save, "Save")
        self.button_save.on_clicked(self.save_and_next)

        ax_back = plt.axes([0.73, 0.01, 0.1, 0.05])
        self.button_back = Button(ax_back, "Back")
        self.button_back.on_clicked(self.prev_file)

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        plt.show()

    #user interactions
    def onclick(self, event):
        # only reacts to clicks inside the graphic box
        if event.inaxes not in self.axes:
            return

        if event.xdata is None:
            return

        idx = int(round(event.xdata))
        idx = max(0, min(idx, self.df.shape[0] - 1))

        # Left button -> start
        if event.button in (MouseButton.LEFT, 1):
            self.start = idx
            for sl, ax in zip(self.start_lines, self.axes):
                if sl is not None:       # <<< ignore sempy subplots
                    sl.set_data([self.start, self.start], ax.get_ylim())

        # Right Button -> end
        elif event.button in (MouseButton.RIGHT, 3):
            self.end = idx
            for el, ax in zip(self.end_lines, self.axes):
                if el is not None:      # <<< ignore sempy subplots
                    el.set_data([self.end, self.end], ax.get_ylim())

        self.fig.canvas.draw_idle()

    # ===================== Save and Continue =====================
    def save_and_next(self, event):
        save_mark(self.current_file, self.start, self.end)

        base = os.path.splitext(os.path.basename(self.current_file))[0]  # ex: "baseline_BP_30_processed"
        plot_name = f"plot_{base}.png"  # ex: "plot_baseline_BP_30_processed.png"
        plot_path = os.path.join(os.path.dirname(self.current_file), plot_name)

        self.fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f" The plot was saved at: {plot_path}")

        self.idx += 1
        save_progress(self.idx)

        if self.idx < len(self.file_list):
            self.load_file()
        else:
            print("\n🎉 You have cut all the files!")
            plt.close("all")


    def prev_file(self, event):
        if self.idx > 0:
            self.idx -= 1
            save_progress(self.idx)
            self.load_file()
        else:
            print("You are already in the first file!!")


def choose_cut_method():
    """
    The user can choose betwwen making new cuts, use
    the pre-defined cuts in the "pre_cut_marks" file.
    For tht, it is needed to choose between the number 1 or 2
    :return:
    """
    print("1 - Make new cuts")
    print("2 - Use pre-defined cuts")

    choice = input("Choose an option: ").strip()

    if choice == "1":
        run_cut_interface()

    elif choice == "2":
        pre_cut_marks = PRE_CUT_MARKS_FILE
        cut_marks = CUT_MARKS_FILE

        if not pre_cut_marks.exists():
            raise FileNotFoundError(
                f"File not found: {pre_cut_marks}"
            )

        shutil.copy2(pre_cut_marks, cut_marks)

    else:
        raise ValueError("Invalid option.")


def run_cut_interface():
    file_list = load_file_list()

    if not file_list:
        print("No files were found. Please check the defined root!")
        return

    CutGUI(file_list)