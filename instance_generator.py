import numpy as np
import argparse
import os
from datetime import datetime


def generate_ufl_instance(num_facilities, num_customers, seed=None):
    """
    Generates data for a UFL instance with costs scaled to be similar
    to OR-Library benchmarks.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Genera coordinate in uno spazio più grande per aumentare le distanze
    facility_coords = np.random.rand(num_facilities, 2) * 10000
    customer_coords = np.random.rand(num_customers, 2) * 10000

    # 2. Domande dei clienti (come richiesto in precedenza)
    customer_demands = np.random.randint(15, 501, size=num_customers)

    # 3. Costi di trasporto come distanze euclidee
    diff = facility_coords[:, np.newaxis, :] - customer_coords[np.newaxis, :, :]
    transport_costs = np.sqrt(np.sum(diff ** 2, axis=-1))

    # --- SCALING DEI COSTI ---
    # Applichiamo un moltiplicatore casuale per variare la scala dei costi di trasporto
    # Questo simula diverse unità di misura o densità geografiche
    transport_costs *= np.random.uniform(0.5, 5.0)

    # 4. Costi fissi più realistici
    # Nelle istanze ORLIB, i costi fissi sono spesso dello stesso ordine di grandezza
    # dei costi di trasporto totali che una facility potrebbe servire.
    # Calcoliamo una base e aggiungiamo variabilità.

    # Costo medio di trasporto per servire un cliente
    avg_transport_cost_per_customer = np.mean(transport_costs)

    # Base per il costo fisso: costo per servire una frazione dei clienti
    base_fixed_cost = avg_transport_cost_per_customer * (num_customers / num_facilities) * np.random.uniform(5, 20)

    # Aggiungiamo grande variabilità ai costi fissi per renderli diversi tra loro
    # Usiamo una distribuzione log-normale per avere alcuni costi molto alti e molti più bassi
    fixed_costs = np.random.lognormal(mean=np.log(base_fixed_cost), sigma=0.5, size=num_facilities)

    # Arrotondiamo per pulizia
    transport_costs = np.round(transport_costs, 2)
    fixed_costs = np.round(fixed_costs, 2)

    return fixed_costs, transport_costs, customer_demands


def write_instance_to_file(filepath, num_facilities, num_customers, fixed_costs, transport_costs, customer_demands):
    """
    Writes the generated UFL instance to a .txt file.
    Format: Fixed Cost first, then Capacity, with real random demands.
    """
    with open(filepath, 'w') as f:
        # Linea 1: Numero di facility, numero di clienti
        f.write(f"{num_facilities} {num_customers}\n")

        # Sezione Dati delle Facility
        # Scriviamo una capacità fittizia (1.0) e il costo fisso (15000).
        for cost in fixed_costs:
            f.write(f"1.0 {cost}\n")

        # Sezione Dati dei Clienti
        all_customer_data = []
        for j in range(num_customers):
            # Aggiungi la domanda REALE pseudo-casuale per il cliente j
            all_customer_data.append(str(customer_demands[j]))

            # Aggiungi i costi di trasporto per il cliente j
            for i in range(num_facilities):
                all_customer_data.append(str(transport_costs[i, j]))

        # Scriviamo i dati dei clienti in un unico flusso
        line_break_every = 7
        for i in range(0, len(all_customer_data), line_break_every):
            line = " ".join(all_customer_data[i:i + line_break_every])
            f.write(line + "\n")

    print(f"Successfully generated instance file at: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate UFL instances in OR-Library format.")
    parser.add_argument(
        '-f', '--facilities', type=int, required=True,
        help='Number of potential facility locations.'
    )
    parser.add_argument(
        '-c', '--customers', type=int, required=True,
        help='Number of customers.'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, default='data',
        help='Output directory to save the instance file (default: data).'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility.'
    )

    args = parser.parse_args()

    num_facilities = args.facilities
    num_customers = args.customers
    seed = args.seed if args.seed is not None else int(datetime.now().timestamp())

    print(f"Generating instance with {num_facilities} facilities, {num_customers} customers, and seed={seed}...")

    fixed_costs, transport_costs, customer_demands = generate_ufl_instance(
        num_facilities, num_customers, seed
    )

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"gen_f{num_facilities}_c{num_customers}_s{seed}.txt"
    filepath = os.path.join(args.output_dir, filename)

    write_instance_to_file(
        filepath, num_facilities, num_customers, fixed_costs, transport_costs, customer_demands
    )


if __name__ == '__main__':
    main()
