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
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Run MIP solvers in deterministic mode (1 thread).'
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

    ordered_algos = [
        'weak', 'strong', 'weak_rl', 'strong_rl', 'erlenkotter', 'greedy'
    ]
    algos_to_run = [algo for algo in ordered_algos if algo in algos_to_run]

    all_results = []

    for instance_path in sorted(instance_files):
        print(f"\n--- Processing Instance: {os.path.basename(instance_path)} ---")

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

                # Le funzioni per PLI e RL devono essere modificate per accettare `deterministic`
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
                opened_count = len(result_obj['opened_facilities']) if 'opened_facilities' in result_obj else -1

                if algo_name in ['strong', 'weak'] and optimal_value is None:
                    optimal_value = objective

                instance_results.append({
                    'algorithm': algo_name,
                    'objective': objective,
                    'time_sec': elapsed_time,
                    'num_facilities': num_fac,
                    'num_customers': num_cust,
                    'opened_facilities_count': opened_count  # Salviamo subito il conteggio
                })

            except Exception as e:
                print(f"     ERROR running {algo_name}: {e}")
                instance_results.append({
                    'algorithm': algo_name,
                    'objective': 'Error',
                    'time_sec': -1,
                    'num_facilities': num_fac,
                    'num_customers': num_cust,
                    'opened_facilities_count': -1
                })

        if optimal_value is not None:
            for res in instance_results:
                if res['algorithm'] in ['greedy', 'erlenkotter'] and isinstance(res['objective'], (int, float)):
                    z_heur = res['objective']
                    gap = ((z_heur - optimal_value) / optimal_value) * 100 if optimal_value > 0 else 0
                    res['optimality_gap_%'] = round(gap, 2)
                else:
                    res['optimality_gap_%'] = '-'

        for res in instance_results:
            res['instance'] = os.path.basename(instance_path)
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


if __name__ == '__main__':
    main()