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

    # Initially, assign each customer to a dummy facility with infinite cost
    current_assignment_costs = np.full(num_customers, np.inf)

    while True:
        best_saving = -np.inf
        best_facility_to_open = -1

        # Evaluate opening each currently closed facility
        for facility_idx in closed_facilities:
            # Calculate the potential new assignment costs if we open this facility
            potential_new_costs = np.minimum(current_assignment_costs, transport_costs[facility_idx, :])

            # Saving is the reduction in transport costs minus the fixed cost
            transport_saving = np.sum(current_assignment_costs - potential_new_costs)
            total_saving = transport_saving - fixed_costs[facility_idx]

            if total_saving > best_saving:
                best_saving = total_saving
                best_facility_to_open = facility_idx

        # If no facility provides a positive saving, stop
        if best_saving > 0:
            opened_facilities.add(best_facility_to_open)
            closed_facilities.remove(best_facility_to_open)
            # Update the current assignment costs for the next iteration
            current_assignment_costs = np.minimum(current_assignment_costs, transport_costs[best_facility_to_open, :])
        else:
            break

    # Calculate final objective value
    final_fixed_cost = np.sum(fixed_costs[list(opened_facilities)])
    final_transport_cost = np.sum(current_assignment_costs)
    objective_value = final_fixed_cost + final_transport_cost

    return {
        'objective': objective_value,
        'opened_facilities': sorted(list(opened_facilities))
    }