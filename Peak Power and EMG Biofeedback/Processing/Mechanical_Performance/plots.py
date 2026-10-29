#--------------IMPORTS-----------#
import matplotlib.pyplot as plt
import numpy as np
from Processing.Mechanical_Performance.b_signal_performance_metrics import peaks_metrics



#----------PLOT FUNCTIONS-------------------#
def plot_grandezas(power, force, velocity, displacement, time, file_name, emg_ag_env, emg_ant_env, emg_ag, emg_ant, high_peaks=None, low_peaks=None):
    min_len = min(len(time), len(displacement), len(velocity), len(force), len(power), len(emg_ag_env), len(emg_ant_env))
    time_plt = time[:min_len]
    displacement_plt = displacement[:min_len]
    velocity_plt = velocity[:min_len]
    force_plt =force[:min_len]
    power_plt = power[:min_len]
    emg_ag_env_plt = emg_ag_env[:min_len]
    emg_ant_env_plt = emg_ant_env[:min_len]
    emg_ag_plt = emg_ag[:min_len]
    emg_ant_plt = emg_ant[:min_len]
    fig, axs = plt.subplots(6, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(time_plt, displacement_plt, linewidth=1.7, label="Displacement", color='orange')
    axs[0].plot(time_plt[high_peaks], displacement_plt[high_peaks],'ro', markersize=6, label="High Peaks")
    axs[0].plot(time_plt[low_peaks], displacement_plt[low_peaks],'go', markersize=6, label="Low Peaks")
    for lp, hp in zip(low_peaks, high_peaks):
        axs[0].axvspan(time[lp], time[hp], color='purple', alpha=0.2)

    axs[0].set_ylabel("Displacement (m)")
    axs[0].legend(loc= "upper right")
    axs[0].grid(True)

    axs[1].plot(time_plt, emg_ag_plt, linewidth=1.7, label="EMG", color='brown')
    axs[1].plot(time_plt, emg_ag_env_plt, linewidth=1.7, label="Env Agonist", color='red')
    for lp, hp in zip(low_peaks, high_peaks):
        axs[1].axvspan(time[lp], time[hp], color='purple', alpha=0.2)
    axs[1].set_ylabel("mV")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    axs[2].plot(time_plt, emg_ant_plt, linewidth=1.7, label="EMG", color='red')
    axs[2].plot(time_plt, emg_ant_env_plt, linewidth=1.7, label="Env Antagonist", color='brown')
    for lp, hp in zip(low_peaks, high_peaks):
        axs[2].axvspan(time[lp], time[hp], color='purple', alpha=0.2)
    axs[2].set_ylabel("mV")
    axs[2].legend(loc="upper right")
    axs[2].grid(True)

    axs[3].plot(time_plt, velocity_plt, linewidth=1.7, label="Velocity", color='black')
    for lp, hp in zip(low_peaks, high_peaks):
        axs[3].axvspan(time[lp], time[hp], color='purple', alpha=0.2)
    axs[3].set_ylabel("Velocity (m/S)")
    axs[3].legend(loc= "upper right")
    axs[3].grid(True)

    axs[4].plot(time_plt, force_plt, linewidth=1.7, label="Force", color='green')
    for lp, hp in zip(low_peaks, high_peaks):
        axs[4].axvspan(time[lp], time[hp], color='purple', alpha=0.2)
    axs[4].set_ylabel("Force (N)")
    axs[4].legend(loc= "upper right")
    axs[4].grid(True)

    axs[5].plot(time_plt, power_plt, linewidth=1.7, label="Power", color='blue')
    for lp, hp in zip(low_peaks, high_peaks):
        axs[5].axvspan(time[lp], time[hp], color='purple', alpha=0.2)
    axs[5].set_ylabel("Power (W)")
    axs[5].set_xlabel("Time (s)")
    axs[5].legend(loc= "upper right")
    axs[5].grid(True)

    plt.suptitle(f"Physical Quantities Analysis- {file_name}")
    plt.tight_layout()
    plt.show()

def plot_all_concentric_phase(cycles, file_name):
    """
    Function to plot all the concentric phases from one signals
    :param cycles: list of dict with all the parameters: phase (norm), displacement, velocity, force, power
    :return: one plot with all the concentric phases in one plot
    """
    plt.figure()

    for n_cycle, cycle in enumerate(cycles, start=1):
        phase = cycle["phase"]
        displacement = cycle["displacement"]
        plt.plot(phase, displacement, alpha=0.8, label=f"Cycle {n_cycle}")

    plt.axvline(0, linestyle="--", color="black")
    plt.axvline(1, linestyle="--", color="black")
    plt.xlabel("Concentric Phases")
    plt.ylabel("Displacement (m)")
    plt.title(f"Displacement of All Concentric Phases in One Signal - {file_name}")
    plt.legend()
    plt.show()

def plot_metrics_norm(cycles, file_name, rpd, n_show = 1):
    n_cycles = len(cycles)
    n_plot = min(n_cycles, n_show)  #only 1 cycle now. it can be more
    fig, axs = plt.subplots(1, n_plot, figsize=(10, 6))

    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    for n_cycle, (cycle, ax) in enumerate(zip(cycles[:n_plot], axs), start=1):
        phase = cycle["phase"]
        displacement = cycle["displacement"]
        velocity = cycle["velocity"]
        force = cycle["force"]
        power = cycle["power"]
        emg_ag_env = cycle["emg_ag_env"]
        emg_ant_env = cycle["emg_ant_env"]
        peaks, _, _ = peaks_metrics(phase, velocity, force, power, emg_ag_env, emg_ant_env, rpd )


        #secondary axes
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax4 = ax.twinx()
        ##ax5 = ax.twinx()
        ##ax6 = ax.twinx()

        # Ajustar posições de deslocamento para o 3º e 4º eixo
        ax3.spines["right"].set_position((('outward', 45)))
        ax4.spines["right"].set_position((('outward', 90)))
        ##ax5.spines["right"].set_position((('outward', 130)))
        ##ax6.spines["right"].set_position((('outward', 170)))

        ax.plot(phase, displacement, label="Displacement", color="blue", alpha= 0.9)
        ax2.plot(phase, force, label="Force", color="green", alpha= 0.9)
        ax3.plot(phase, velocity, label="Velocity", color="red", alpha= 0.9)
        ax4.plot(phase, power, label="Power", color="purple", alpha= 0.9)
        ##ax5.plot(phase, emg_ag_env, label="EMG Agonist", color="skyblue", alpha=0.9)
        ##ax6.plot(phase, emg_ant_env, label="EMG Antagonist", color="orange", alpha=0.9)

        handles, labels = [], []

        # Plots of the peaks
        for idx, _, ph, val in peaks[0]["force"]:
            ax2.scatter(ph, val, color="green", marker="o", s=30, edgecolor="k")
            ax2.axvline(ph, linestyle="--", color="green", alpha=0.5)
            handles.append(ax2.scatter([], [], color="green", marker="o"))
            labels.append(f"Peak Force: {val:.2f} N at {(ph * 100):.0f}%")
        for idx, _, ph, val in peaks[0]["velocity"]:
            ax3.scatter(ph, val, color="red", marker="o", s=30, edgecolor="k")
            ax3.axvline(ph, linestyle="--", color="red", alpha=0.5)
            handles.append(ax3.scatter([], [], color="red", marker="o"))
            labels.append(f"Peak Velocity: {val:.2f} m/s at {(ph * 100):.0f}%")
        for idx, _, ph, val in peaks[0]["power"]:
            ax4.axvline(ph, linestyle="--", color="purple", alpha=0.5)
            ax4.scatter(ph, val, color="purple", marker="o", s=30, edgecolor="k")
            handles.append(ax4.scatter([], [], color="purple", marker="o"))
            labels.append(f"Peak Power: {val:.2f} W at {(ph * 100):.0f}%")
        ##for idx,_, ph, val in peaks[0]["emg_ag_env"]:
            ##ax5.axvline(ph, linestyle="--", color="skyblue", alpha=0.5)
            ##ax5.scatter(ph, val, color="skyblue", marker="o", s=30, edgecolor="k")
            ##handles.append(ax5.scatter([], [], color="skyblue", marker="o"))
            ##labels.append(f"Peak Env Agonist: {val:.2f} mV at {(ph * 100):.0f}%")
        ##for idx, _, ph, val in peaks[0]["emg_ant_env"]:
            ##ax6.scatter(ph, val, color="orange", marker="o", s=30, edgecolor="k")
            ##ax6.axvline(ph, linestyle="--", color="orange", alpha=0.5)
            ##handles.append(ax6.scatter([], [], color="orange", marker="o"))
            ##labels.append(f"Peak Env Antagonist: {val:.2f} mV at {(ph * 100):.0f}%")

        ax.set_ylabel("Displacement (m)", color='blue', fontsize=10)
        ax.tick_params(axis='y', colors='blue', labelsize=8)
        ax2.set_ylabel("Force (N)", color='green', fontsize=10)
        ax2.tick_params(axis='y', colors='green', labelsize=8)
        ax3.set_ylabel("Velocity (m/s)", color='red',  fontsize=10)
        ax3.tick_params(axis='y', colors='red', labelsize=8)
        ax4.set_ylabel("Power (W)", color='purple', fontsize=10)
        ax4.tick_params(axis='y', colors='purple', labelsize=8)
        ##ax5.set_ylabel("Amp Ag (mV)", color='skyblue', fontsize=10)
        ##ax5.tick_params(axis='y', colors='skyblue', labelsize=8)
        ##ax6.set_ylabel("Amp Ant (mV)", color='orange', fontsize=10)
        ##ax6.tick_params(axis='y', colors='orange', labelsize=8)

        ##for a in [ax, ax2, ax3, ax4, ax5, ax6]: #with ax5 and ax6
        for a in [ax, ax2, ax3, ax4]:
            a.axvline(0, linestyle="--", color="black")
            a.axvline(1, linestyle="--", color="black")

        ax.set_title(f"Cycle {n_cycle}")
        ax.set_xlabel("Phase")

        # create legend together (peaks + labels)
        #lines, labels = [], []
        ##for a in [ax, ax2, ax3, ax4, ax5, ax6]: #with ax5 and ax6
        for a in [ax, ax2, ax3, ax4]:
            l, lab = a.get_legend_handles_labels()
            handles += l
            labels += lab
        ax.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        #ax.legend(handles, labels_peak, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.35))

    axs[0].set_ylabel("Displacement (m)")
    fig.suptitle(f"pPrfomance metrics inside the concentric phase. File: {file_name}")
    plt.tight_layout()
    plt.show()