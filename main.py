import os
import time
import argparse
import pandas as pd
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

    args = parser.parse_args()

    if os.path.isdir(args.instance):
        instance_files = [os.path.join(args.instance, f) for f in os.listdir(args.instance) if f.endswith('.txt')]
    else:
        instance_files = [args.instance]

    if args.algorithm == 'all':
        algos_to_run = list(ALGORITHMS.keys())
    else:
        algos_to_run = [args.algorithm]

    results = []

    for instance_path in sorted(instance_files):
        print(f"\n--- Processing Instance: {os.path.basename(instance_path)} ---")

        try:
            num_fac, num_cust, fixed_costs, transport_costs = parse_or_library_instance(instance_path)
            print(f"Parsed: {num_fac} facilities, {num_cust} customers.")
        except Exception as e:
            print(f"Error parsing {instance_path}: {e}")
            continue

        for algo_name in algos_to_run:
            print(f"\n  -> Running algorithm: {algo_name}...")

            solver_func = ALGORITHMS[algo_name]
            result_obj = None
            elapsed_time = -1

            try:
                start_time = time.perf_counter()

                # Logica di chiamata corretta
                if algo_name in ['strong', 'weak', 'strong_rl', 'weak_rl']:
                    result_obj = solver_func(fixed_costs, transport_costs, solver=args.solver)
                else:
                    result_obj = solver_func(fixed_costs, transport_costs)

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                print(f"     Objective: {result_obj['objective']:.2f}")
                print(f"     Time: {elapsed_time:.4f} seconds")

                # Elaborazione robusta dei risultati
                objective = result_obj['objective']
                opened_count = len(result_obj['opened_facilities']) if 'opened_facilities' in result_obj else -1

                results.append({
                    'instance': os.path.basename(instance_path),
                    'algorithm': algo_name,
                    'objective': objective,
                    'time_sec': elapsed_time,
                    'num_facilities': num_fac,
                    'num_customers': num_cust,
                    'opened_facilities_count': opened_count
                })

            except Exception as e:
                # Se c'è un errore, lo stampiamo e registriamo il fallimento
                print(f"     ERROR running {algo_name}: {e}")
                results.append({
                    'instance': os.path.basename(instance_path),
                    'algorithm': algo_name,
                    'objective': 'Error',
                    'time_sec': -1,
                    'num_facilities': num_fac,
                    'num_customers': num_cust,
                    'opened_facilities_count': -1
                })

    if results:
        df_results = pd.DataFrame(results)
        if results:
            df_results = pd.DataFrame(results)

            # --- STAMPA MIGLIORATA E RAGGRUPPATA ---
            print("\n\n--- Performance Summary ---")

            # Filtra e stampa ogni categoria separatamente
            df_pli = df_results[df_results['algorithm'].isin(['strong', 'weak'])]
            df_rl = df_results[df_results['algorithm'].isin(['strong_rl', 'weak_rl'])]
            df_heuristics = df_results[df_results['algorithm'].isin(['greedy', 'erlenkotter'])]

            print("\n[1] Exact MIP Models (Optimal Solutions)")
            if not df_pli.empty:
                print(df_pli.sort_values(by='time_sec').to_string(index=False))

            print("\n[2] Linear Relaxations (Lower Bounds)")
            if not df_rl.empty:
                # Ordina per valore dell'obiettivo per vedere la gerarchia
                print(df_rl.sort_values(by='objective').to_string(index=False))

            print("\n[3] Heuristics (Approximate Solutions)")
            if not df_heuristics.empty:
                print(df_heuristics.sort_values(by='objective').to_string(index=False))

            # Salva i risultati completi nel CSV
            output_filename = 'ufl_results.csv'
            # Ordiniamo il DataFrame prima di salvarlo per coerenza
            df_results = df_results.sort_values(by=['instance', 'algorithm'])
            df_results.to_csv(output_filename, index=False)
            print(f"\nResults saved to {output_filename}")


if __name__ == '__main__':
    main()