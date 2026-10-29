from Preprocessing.b_cutting_interface import choose_cut_method
from Preprocessing.a_preprocessing import run_preprocessing
from Processing.Mechanical_Performance.a_analysis import run_analysis
from Processing.Mechanical_Performance.c_cleaning_performance_metrics import run_cleaning
from Processing.Mechanical_Performance.d_aggregation_performance_metrics import run_aggregation_interface
from Processing.EMG_Analysis.e_emg_analysis import run_emg_analysis
from Processing.EMG_Analysis.f_emg_cycle_onset_association import run_emg_cycle_onset_association
from Processing.EMG_Analysis.g_emg_temporal_metrics import run_emg_temporal_metrics
from Processing.EMG_Analysis.h_emg_complete_metrics import run_emg_complete_metrics


def main():
    #1 --- Preprocess the complete raw signals (e.g. transfer functions, filtering)
    #run_preprocessing()
    print("Module A complete!")
    #2 --- Choose between manul cutting or pre-defined cut marks
    choose_cut_method()
    print("Module B complete!")
    #3 --- Detect concentric phases and calculate the performance metrics
    run_analysis()
    print("Module A complete!")
    #4 --- Remove invalid results and clean the calculated metrics
    run_cleaning()
    print("Module C complete!")
    #5 --- Group performance metrics by user-selected columns
    run_aggregation_interface()
    print("Module D complete!")

    #6 --- Detect and validate muscle activation onsets
    run_emg_analysis()
    print("Module E complete!")

    #7 --- Associate muscle activation onsets with concentric cycles
    run_emg_cycle_onset_association()
    print("Module F complete!")

    #8 --- Calculate temporal metrics from movement cycles and muscle onsets
    run_emg_temporal_metrics()
    print("Module G complete!")
    #9 --- Calculate additional EMG metrics and export the final results
    run_emg_complete_metrics()
    print("Module H complete!")



if __name__ == "__main__":
    main()