import os
import time
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from src.data_parser import parse_or_library_instance
from src.models.pli_strong import solve_strong_formulation, solve_strong_relaxation
from src.models.pli_weak import solve_weak_formulation, solve_weak_relaxation
from src.algorithms.greedy_heuristic import solve_greedy_heuristic
from src.algorithms.erlenkotter import solve_erlenkotter

ALGORITHMS = {
    'strong': solve_strong_formulation,
    'strong_rl': solve_strong_relaxation,
    'weak': solve_weak_formulation,
    'weak_rl': solve_weak_relaxation,
    'greedy': solve_greedy_heuristic,
    'erlenkotter': solve_erlenkotter,
}


def write_instance_report(instance_name, results_df, output_dir='reports'):
    """
    Scrive un report testuale dettagliato per una singola istanza.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Rimuove l'estensione .txt e aggiungi .log
    base_name = os.path.splitext(instance_name)[0]
    report_path = os.path.join(output_dir, f"{base_name}_report.txt")

    n_fac = results_df['num_facilities'].iloc[0]
    n_cust = results_df['num_customers'].iloc[0]

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"ANALYSIS REPORT FOR INSTANCE: {instance_name}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Instance Details:\n")
        f.write(f"  - Facilities: {n_fac}\n")
        f.write(f"  - Customers:  {n_cust}\n\n")

        # Ordina i risultati per la scrittura nel report
        algo_order = ['strong', 'weak', 'strong_rl', 'weak_rl', 'erlenkotter', 'greedy']
        results_df['algorithm'] = pd.Categorical(results_df['algorithm'], categories=algo_order, ordered=True)
        sorted_results = results_df.sort_values('algorithm')

        for _, row in sorted_results.iterrows():
            f.write(f"--- Algorithm: {row['algorithm']} ---\n")
            f.write(f"  - Objective Cost:          {row['objective']}\n")
            f.write(f"  - Computational Time (s):  {row['time_sec']:.4f}\n")

            if row['optimality_gap_%'] != '-':
                f.write(f"  - Optimality Gap:          {row['optimality_gap_%']}%\n")

            if row['opened_facilities_count'] != -1:
                f.write(f"  - Opened Facilities Count: {row['opened_facilities_count']}\n")
                # Assumiamo che la lista delle facility sia salvata
                opened_list_str = ', '.join(map(str, row['opened_facilities']))
                f.write(f"  - Opened Facilities (IDs): [{opened_list_str}]\n")

            f.write("\n")

    print(f"--> Detailed report saved to {report_path}")



def main():
    parser = argparse.ArgumentParser(description="Run UFL algorithms on OR-Library instances.")
    parser.add_argument(
        '--instance',
        type=str,
        help='Path to a single instance file or a directory of instances.'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=list(ALGORITHMS.keys()) + ['all'],
        default='all',
        help='The algorithm to run.'
    )
    parser.add_argument(
        '--solver',
        type=str,
        default='gurobi',
        help='The AMPL solver to use for ILP models (e.g., cplex, gurobi).'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Run MIP solvers in deterministic mode (1 thread).'
    )

    args = parser.parse_args()

    # --- LOGICA DI RICERCA FILE E PERCORSI ROBUSTA ---
    instance_files = []
    root_path_str = args.instance

    if os.path.isdir(root_path_str):
        for dirpath, _, filenames in os.walk(root_path_str):
            for filename in filenames:
                if filename.endswith('.txt'):
                    instance_files.append(os.path.join(dirpath, filename))
    elif os.path.isfile(root_path_str):
        instance_files = [root_path_str]
    else:
        print(f"Error: Path '{root_path_str}' does not exist or is not a file/directory.")
        return

    if not instance_files:
        print(f"No instance files (.txt) found in '{root_path_str}'.")
        return


    ordered_algos = [
        'weak', 'strong', 'weak_rl', 'strong_rl', 'erlenkotter', 'greedy'
    ]
    algos_to_run = [algo for algo in ordered_algos if
                    algo in ALGORITHMS and (args.algorithm == 'all' or algo == args.algorithm)]

    all_results = []
    for instance_path in sorted(instance_files):
        print(f"\n--- Processing Instance: {os.path.relpath(instance_path)} ---")

        try:
            num_fac, num_cust, fixed_costs, transport_costs = parse_or_library_instance(instance_path)
            print(f"Parsed: {num_fac} facilities, {num_cust} customers.")
        except Exception as e:
            print(f"Error parsing {instance_path}: {e}")
            continue

        instance_results = []
        optimal_value = None

        for algo_name in algos_to_run:
            print(f"  -> Running algorithm: {algo_name}...")
            solver_func = ALGORITHMS[algo_name]
            result_obj = None
            try:
                start_time = time.perf_counter()
                if algo_name in ['strong', 'weak', 'strong_rl', 'weak_rl']:
                    result_obj = solver_func(fixed_costs, transport_costs, solver=args.solver,
                                             deterministic=args.deterministic)
                else:
                    result_obj = solver_func(fixed_costs, transport_costs)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"     Objective: {result_obj['objective']:.2f}")
                print(f"     Time: {elapsed_time:.4f} seconds")
                objective = result_obj['objective']
                opened_facilities_list = result_obj.get('opened_facilities', [])
                opened_count = len(opened_facilities_list) if opened_facilities_list else -1

                if algo_name in ['strong', 'weak'] and optimal_value is None:
                    optimal_value = objective
                instance_results.append({
                    'algorithm': algo_name, 'objective': objective, 'time_sec': elapsed_time,
                    'num_facilities': num_fac, 'num_customers': num_cust,
                    'opened_facilities_count': opened_count, 'opened_facilities': opened_facilities_list
                })


            except Exception as e:
                print(f"     ERROR running {algo_name}: {e}")
                instance_results.append({
                    'algorithm': algo_name, 'objective': 'Error', 'time_sec': -1,
                    'num_facilities': num_fac, 'num_customers': num_cust,
                    'opened_facilities_count': -1, 'opened_facilities': []
                })

            # --- CALCOLO DI TUTTI I GAP ---
            if optimal_value is not None:
                for res in instance_results:
                    algo = res['algorithm']
                    value = res['objective']

                    # Salta se il risultato è un errore
                    if not isinstance(value, (int, float)):
                        res['optimality_gap_%'] = '-'
                        continue

                    gap = 0.0
                    # Calcola il GAP DI OTTIMALITÀ per l'Upper Bound (greedy)
                    if algo == 'greedy':
                        gap = ((value - optimal_value) / optimal_value) * 100 if optimal_value > 0 else 0
                        res['optimality_gap_%'] = round(gap, 2)

                    # Calcola il GAP DI DUALITÀ per i Lower Bounds
                    elif algo in ['strong_rl', 'weak_rl', 'erlenkotter']:
                        gap = ((optimal_value - value) / optimal_value) * 100 if optimal_value > 0 else 0
                        res['optimality_gap_%'] = round(gap, 2)

                    # Per le soluzioni ottime, il gap è 0 (o non si applica)
                    else:
                        res['optimality_gap_%'] = '-'

        instance_basename = os.path.basename(instance_path)

        for res in instance_results:
            res['instance'] = instance_basename


        # --- LOGICA DI CREAZIONE SOTTOCARTELLE CORRETTA ---
        # La directory di base per il calcolo del percorso relativo è quella passata come argomento
        root_input_dir_for_relpath = args.instance if os.path.isdir(args.instance) else os.path.dirname(
            args.instance)

        instance_dir_abs = os.path.abspath(os.path.dirname(instance_path))
        root_input_abs = os.path.abspath(root_input_dir_for_relpath)

        relative_subdir = os.path.relpath(instance_dir_abs, root_input_abs)
        report_output_dir = os.path.join('reports', relative_subdir) if relative_subdir != '.' else 'reports'

        df_instance_results = pd.DataFrame(instance_results)
        # Passiamo la lista delle facility alla funzione di report
        for res in instance_results:
            res['instance'] = instance_basename

        write_instance_report(instance_basename, df_instance_results, output_dir=report_output_dir)

        all_results.extend(instance_results)


    if all_results:
        df_results = pd.DataFrame(all_results)

        # Definiamo l'ordine desiderato per la visualizzazione
        algo_order = ['strong', 'weak', 'strong_rl', 'weak_rl', 'erlenkotter', 'greedy']

        # Convertiamo la colonna 'algorithm' in un tipo categorico con il nostro ordine
        # Questo ci permette di ordinare il DataFrame secondo la nostra logica
        df_results['algorithm'] = pd.Categorical(df_results['algorithm'], categories=algo_order, ordered=True)

        for instance_name, group in df_results.groupby('instance'):
            n_fac = group['num_facilities'].iloc[0]
            n_cust = group['num_customers'].iloc[0]

            # Ordiniamo il gruppo secondo l'ordine personalizzato
            sorted_group = group.sort_values('algorithm')

            print(f"\n\n--- Analysis for Instance: {instance_name} (Facilities: {n_fac}, Customers: {n_cust}) ---")

            # 1. Tabella di Qualità
            print("\n[1] Solution Quality (Cost, Gap, and Facilities Opened)")
            quality_cols = ['algorithm', 'objective', 'optimality_gap_%', 'opened_facilities_count']
            print(sorted_group[quality_cols].to_string(index=False))

            # 2. Tabella del Tempo
            print("\n[2] Computational Time")
            time_cols = ['algorithm', 'time_sec']
            print(sorted_group[time_cols].to_string(index=False))

        # Salva il CSV completo, anch'esso ordinato
        output_filename = 'ufl_results_detailed.csv'
        final_cols = ['instance', 'algorithm', 'objective', 'time_sec', 'optimality_gap_%',
                      'opened_facilities_count', 'num_facilities', 'num_customers']
        # Ordiniamo il DataFrame completo prima di salvarlo
        df_results = df_results.sort_values(['instance', 'algorithm'])
        df_results = df_results.reindex(columns=final_cols)
        df_results.to_csv(output_filename, index=False)
        print(f"\n\nDetailed results saved to {output_filename}")


        # --- ANALISI STATISTICA AGGREGATA ---
        print("\n\n" + "=" * 80)
        print("--- AGGREGATE PERFORMANCE ANALYSIS (COMPUTATIONAL TIME) ---".center(80))
        print("=" * 80)

        summary_data = []
        # Filtra solo le esecuzioni andate a buon fine
        df_successful = df_results[df_results['time_sec'] >= 0].copy()

        for algo_name, group in df_successful.groupby('algorithm'):
            times = group['time_sec']
            n = len(times)
            mean_time = times.mean()
            min_time = times.min()
            max_time = times.max()

            # Calcola l'intervallo di confidenza solo se ci sono abbastanza dati
            if n > 1:
                std_dev = times.std()
                # Standard error of the mean
                sem = std_dev / np.sqrt(n)
                # t-value per 95% di confidenza con n-1 gradi di libertà
                t_crit = stats.t.ppf(0.975, df=n - 1)
                ci_lower = mean_time - t_crit * sem
                ci_upper = mean_time + t_crit * sem
                ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            else:
                ci_str = "N/A"

            summary_data.append({
                "Algorithm": algo_name,
                "Instances": n,
                "Mean Time (s)": mean_time,
                "95% CI": ci_str,
                "Time Range (s)": f"[{min_time:.4f}, {max_time:.4f}]"
            })

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary = df_summary.set_index('Algorithm').reindex(algo_order).dropna().reset_index()
            print(df_summary.to_string(index=False))
        print("=" * 80)


if __name__ == '__main__':
    main()