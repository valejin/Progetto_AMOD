import numpy as np


def parse_or_library_instance(file_path):
    """
    Parses an UFL instance file from the OR-Library.

    Args:
        file_path (str): The path to the instance file.

    Returns:
        tuple: A tuple containing:
            - num_facilities (int): Number of potential facility locations.
            - num_customers (int): Number of customers.
            - fixed_costs (np.array): A 1D array of fixed costs for opening each facility.
            - transport_costs (np.array): A 2D array (num_facilities x num_customers)
                                          of transportation costs.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # First line: num_facilities, num_customers
    num_facilities, num_customers = map(int, lines[0].strip().split())

    fixed_costs = np.zeros(num_facilities)
    transport_costs = np.zeros((num_facilities, num_customers))

    # Read facility data (capacity and fixed cost)
    # For UFL, we only need the fixed cost.
    current_line_idx = 1
    for i in range(num_facilities):
        parts = lines[current_line_idx].strip().split()
        # The format can be just fixed cost or capacity + fixed cost
        fixed_costs[i] = float(parts[0])
        current_line_idx += 1

    # Read customer data (demand and transportation costs)
    all_costs_flat = []
    for line in lines[current_line_idx:]:
        all_costs_flat.extend(map(float, line.strip().split()))

    # The costs are grouped by customer. We need to restructure them.
    cost_idx = 0
    for j in range(num_customers):
        # The first number in a customer block is the demand, which we skip for UFL.
        # However, many formats just list costs consecutively. We need to be careful.
        # Let's assume the provided format where demand is followed by costs.
        # A robust way is to read all numbers and then structure them.

        # The OR-Lib format for these instances is tricky. A common variant is:
        # For each customer:
        #   demand
        #   costs c_1j, c_2j, ..., c_Ij
        # This means the total numbers after facility data are num_customers * (1 + num_facilities)
        # Let's re-read the cost part more carefully.

        cost_pointer = 0
        all_numbers = []
        for line in lines[current_line_idx:]:
            all_numbers.extend(line.strip().split())

        for j in range(num_customers):
            # The first number is demand, we skip it
            demand = float(all_numbers[cost_pointer])
            cost_pointer += 1
            for i in range(num_facilities):
                transport_costs[i, j] = float(all_numbers[cost_pointer])
                cost_pointer += 1

    return num_facilities, num_customers, fixed_costs, transport_costs