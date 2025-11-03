import numpy as np


def solve_greedy_heuristic(fixed_costs, transport_costs):
    """
    Solves the UFL problem using a simple greedy ADD heuristic.
    It iteratively opens the facility that provides the best cost saving.

    Args:
        fixed_costs (np.array): Fixed costs for facilities.
        transport_costs (np.array): Transportation costs (facilities x customers).

    Returns:
        dict: A dictionary with results (objective, opened_facilities).
    """
    num_facilities, num_customers = transport_costs.shape

    opened_facilities = set()
    closed_facilities = set(range(num_facilities))

    # Inizialmente, assegnare a ciascun cliente una struttura fittizia con costi infiniti
    current_assignment_costs = np.full(num_customers, np.inf)

    while True:
        best_saving = -np.inf
        best_facility_to_open = -1

        # Valutare l'apertura di ogni struttura attualmente chiusa
        for facility_idx in closed_facilities:
            # Calcola i potenziali nuovi costi di assegnazione se apriamo questa struttura
            potential_new_costs = np.minimum(current_assignment_costs, transport_costs[facility_idx, :])

            # Il risparmio è la riduzione dei costi di trasporto meno il costo fisso di apertura
            transport_saving = np.sum(current_assignment_costs - potential_new_costs)
            total_saving = transport_saving - fixed_costs[facility_idx]

            if total_saving > best_saving:
                best_saving = total_saving
                best_facility_to_open = facility_idx

        # Se nessuna facility chiusa può offrire un risparmio positivo, stop
        if best_saving > 0:
            opened_facilities.add(best_facility_to_open)
            closed_facilities.remove(best_facility_to_open)
            # Aggiorna i costi di assegnazione correnti per la prossima iterazione
            current_assignment_costs = np.minimum(current_assignment_costs, transport_costs[best_facility_to_open, :])
        else:
            break

    # Calcola il valore obiettivo finale
    final_fixed_cost = np.sum(fixed_costs[list(opened_facilities)])
    final_transport_cost = np.sum(current_assignment_costs)
    objective_value = final_fixed_cost + final_transport_cost

    return {
        'objective': objective_value,
        'opened_facilities': sorted(list(opened_facilities))
    }