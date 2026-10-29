#-------------IMPORTS------------------#
import csv
from config import OUTPUT_ROOT

#--------------FUNCTIONS----------------#
def parse_results_txt(txt_path):
    """Reads the txt with the results of the validated onsets"""
    results = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            path = parts[0]

            # cycles
            cycles_part = [p for p in parts if p.startswith("cycles:")]
            cycles = []
            if cycles_part:
                cycles_str = cycles_part[0].replace("cycles:", "")
                cycles = [tuple(map(int, c.split("-"))) for c in cycles_str.split(";") if c]

            # agonist
            ag_part = [p for p in parts if p.startswith("agonist_onsets:")]
            ag_onsets = [int(x) for x in ag_part[0].replace("agonist_onsets:", "").split(";") if x] if ag_part else []

            # antagonist
            ant_part = [p for p in parts if p.startswith("antagonist_onsets:")]
            ant_onsets = [int(x) for x in ant_part[0].replace("antagonist_onsets:", "").split(";") if x] if ant_part else []

            results.append({"path": path,"cycles": cycles,"agonist_onsets": ag_onsets,"antagonist_onsets": ant_onsets})
    return results

def associate_onsets_to_cycles(cycles, onsets):
    """
    Associates each cycle with the onset closest to the start of the cycle.

    Returns:
        tuple: A list of onset tuples for each cycle and a list of unassociated
            onsets ("Alone").
    """
    associated = []
    used = set()
    for start, end in cycles:
        if not onsets:
            associated.append((-1,))  # Nenhum onset
            continue

        # onset clsoer to the beggining of the cycle
        nearest = min(onsets, key=lambda x: abs(x - start))
        associated.append((nearest,))
        used.add(nearest)

    # onsets that were not associated with any cycle
    alone = [o for o in onsets if o not in used]

    return associated, alone

def create_cycle_onsets_table(results):
    """
    Creates a list of dictionaries for export, associating each cycle with the
    onsets of each muscle separately.
    """
    table = []
    for r in results:
        # Associate agonist and antagonist separately
        ag_assoc, ag_alone = associate_onsets_to_cycles(r["cycles"], r["agonist_onsets"])
        ant_assoc, ant_alone = associate_onsets_to_cycles(r["cycles"], r["antagonist_onsets"])

        # Line for each cycle
        for idx, (cycle, ag_on) in enumerate(zip(r["cycles"], ag_assoc), 1):
            ant_on = ant_assoc[idx-1] if idx-1 < len(ant_assoc) else (-1,)
            s, e = cycle
            table.append({
                "filename": r["path"],
                "cycle_number": idx,
                "cycle_start": s,
                "cycle_end": e,
                "agonist_onset": ag_on[0] if ag_on else -1,
                "antagonist_onset": ant_on[0] if ant_on else -1
            })

        # Lines for the onsets "alone" (that didn't match the cycle pairs)
        for ag_o in ag_alone:
            table.append({
                "filename": r["path"],
                "cycle_number": "Alone",
                "cycle_start": -1,
                "cycle_end": -1,
                "agonist_onset": ag_o,
                "antagonist_onset": -1
            })
        for ant_o in ant_alone:
            table.append({
                "filename": r["path"],
                "cycle_number": "Alone",
                "cycle_start": -1,
                "cycle_end": -1,
                "agonist_onset": -1,
                "antagonist_onset": ant_o
            })

    return table

def export_cycle_onsets_to_csv(table, output_csv):
    """Export the final table for a .csv"""
    keys = ["filename","cycle_number","cycle_start","cycle_end","agonist_onset","antagonist_onset"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in table:
            writer.writerow(row)

def run_emg_cycle_onset_association():
    """
    Runs the pipeline that associated the concentric cycles with the emg analysis
    :return: table with the name "cycle_onsets_table.csv"
    """
    INPUT_TXT = OUTPUT_ROOT / "onsets_emg_results.txt"
    OUTPUT_CSV = OUTPUT_ROOT / "cycle_onsets_table.csv"

    results = parse_results_txt(INPUT_TXT)
    table = create_cycle_onsets_table(results)
    export_cycle_onsets_to_csv(table, OUTPUT_CSV)

    print("Table was created and exported successfully:", OUTPUT_CSV)

