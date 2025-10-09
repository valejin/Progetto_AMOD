import numpy as np


def solve_erlenkotter(fixed_costs, transport_costs):
    """
    Implements a simplified version of Erlenkotter's DUALOC algorithm.
    It consists of a dual ascent phase, a primal construction, and a simple
    local search improvement phase.

    Args:
        fixed_costs (np.array): Fixed costs for facilities.
        transport_costs (np.array): Transportation costs (facilities x customers).

    Returns:
        dict: A dictionary with results (objective, opened_facilities).
    """
    num_facilities, num_customers = transport_costs.shape

    # --- 1. Dual Ascent Phase ---
    # Initialize dual variables v_j to the cost of the cheapest facility for customer j
    v = np.min(transport_costs, axis=0)

    # S_i = sum over j of max(0, v_j - c_ij)
    # CORREZIONE: v (50,) - transport_costs (16, 50) -> broadcast a (16, 50)
    # La somma va fatta lungo l'asse 1 (clienti) per ottenere un valore per ogni facility
    savings = np.maximum(0, v - transport_costs).sum(axis=1)

    # Facilities that are "profitable" or "over-funded"
    I0 = set(np.where(savings >= fixed_costs)[0])

    # --- 2. Primal Construction & Improvement ---
    # (La fase di "dual ascent" iterativa è complessa, passiamo direttamente
    # alla costruzione primale basata sulla soluzione duale iniziale)
    opened_facilities = set(I0)

    # Se I0 è vuoto, apri la facility più economica per iniziare
    if not opened_facilities:
        # Calcola il costo totale per ogni facility se fosse l'unica aperta
        total_costs_if_single = fixed_costs + np.sum(transport_costs, axis=1)
        best_initial_facility = np.argmin(total_costs_if_single)
        opened_facilities.add(best_initial_facility)

    # Simple Local Search Improvement (Add/Drop)
    while True:
        current_obj = calculate_objective(opened_facilities, fixed_costs, transport_costs)
        improved = False

        # Try to ADD a facility
        best_add_candidate = -1
        best_add_saving = 0
        for i in range(num_facilities):
            if i not in opened_facilities:
                new_set = opened_facilities.union({i})
                new_obj = calculate_objective(new_set, fixed_costs, transport_costs)
                saving = current_obj - new_obj
                if saving > best_add_saving:
                    best_add_saving = saving
                    best_add_candidate = i

        if best_add_candidate != -1 and best_add_saving > 1e-6:  # Use tolerance
            opened_facilities.add(best_add_candidate)
            improved = True
            continue

        # Try to DROP a facility
        best_drop_candidate = -1
        best_drop_saving = 0
        for i in list(opened_facilities):
            if len(opened_facilities) > 1:
                new_set = opened_facilities.difference({i})
                new_obj = calculate_objective(new_set, fixed_costs, transport_costs)
                saving = current_obj - new_obj
                if saving > best_drop_saving:
                    best_drop_saving = saving
                    best_drop_candidate = i

        if best_drop_candidate != -1 and best_drop_saving > 1e-6:  # Use tolerance
            opened_facilities.remove(best_drop_candidate)
            improved = True
            continue

        if not improved:
            break

    final_objective = calculate_objective(opened_facilities, fixed_costs, transport_costs)

    return {
        'objective': final_objective,
        'opened_facilities': sorted(list(opened_facilities))
    }


def calculate_objective(opened_facilities, fixed_costs, transport_costs):
    """Helper function to calculate total cost for a given set of open facilities."""
    if not opened_facilities:
        return np.inf

    opened_list = list(opened_facilities)
    total_fixed_cost = np.sum(fixed_costs[opened_list])
    relevant_transport_costs = transport_costs[opened_list, :]
    min_transport_costs = np.min(relevant_transport_costs, axis=0)
    total_transport_cost = np.sum(min_transport_costs)

    return total_fixed_cost + total_transport_cost