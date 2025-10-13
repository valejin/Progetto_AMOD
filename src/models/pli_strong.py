from amplpy import AMPL
import pandas as pd
import numpy as np


def solve_strong_formulation(fixed_costs, transport_costs, solver='gurobi', deterministic=False):
    """
    Solves the UFL problem using the strong ILP formulation.

    Args:
        fixed_costs (np.array): Fixed costs for facilities.
        transport_costs (np.array): Transportation costs (facilities x customers).
        solver (str): The name of the solver to use (e.g., 'cplex', 'gurobi').

    Returns:
        dict: A dictionary with results (objective, opened_facilities, etc.).
    """
    num_facilities, num_customers = transport_costs.shape

    ampl = AMPL()
    ampl.setOption('solver', solver)

    # Imposta le opzioni per rendere il solver deterministico
    # Per Gurobi: threads=1, seed=0
    # Per CPLEX: threads=1, randomseed=1234
    # Usiamo una sintassi generica che molti solver capiscono
    # Opzioni per il determinismo + disabilitazione del presolve
    options = []
    if deterministic:
        options.append('threads=1')
        if solver == 'gurobi':
            options.append('seed=0')
        elif solver == 'cplex':
            options.append('randomseed=1234')

    if options:
        ampl.setOption(f'{solver}_options', ' '.join(options))

    # Define the AMPL model
    ampl.eval(f"""
        # Sets
        set FACILITIES := 0..{num_facilities - 1};
        set CUSTOMERS := 0..{num_customers - 1};

        # Parameters
        param f {{FACILITIES}};
        param c {{FACILITIES, CUSTOMERS}};

        # Decision Variables
        var x {{FACILITIES}} binary;
        var y {{FACILITIES, CUSTOMERS}} >= 0, <= 1;

        # Objective Function
        minimize TotalCost:
            sum {{i in FACILITIES}} f[i] * x[i] +
            sum {{i in FACILITIES, j in CUSTOMERS}} c[i,j] * y[i,j];

        # Constraints
        # Each customer must be served by exactly one facility
        subject to Demand {{j in CUSTOMERS}}:
            sum {{i in FACILITIES}} y[i,j] = 1;

        # Strong Formulation: A customer can only be served by an open facility
        subject to Linking {{i in FACILITIES, j in CUSTOMERS}}:
            y[i,j] <= x[i];
    """)

    # Load data into AMPL
    ampl.getParameter('f').setValues({i: fixed_costs[i] for i in range(num_facilities)})

    # Crea una Series con un MultiIndex (FACILITIES, CUSTOMERS)
    # Questo è il modo più robusto per passare dati 2D ad amplpy
    c_series = pd.DataFrame(
        transport_costs,
        index=pd.Index(range(num_facilities), name='FACILITIES'),
        columns=pd.Index(range(num_customers), name='CUSTOMERS')
    ).stack()

    ampl.getParameter('c').setValues(c_series)

    # Solve the problem
    ampl.solve()

    # Extract results
    objective_value = ampl.getObjective('TotalCost').value()
    x_sol = ampl.getVariable('x').getValues().toPandas()
    opened_facilities = [int(i) for i, row in x_sol.iterrows() if row['x.val'] > 0.5]

    ampl.close()

    return {
        'objective': objective_value,
        'opened_facilities': sorted(opened_facilities)
    }


def solve_strong_relaxation(fixed_costs, transport_costs, solver='gurobi', deterministic=False):
    """
    Solves the Linear Relaxation of the UFL problem using the strong formulation.

    Args:
        fixed_costs (np.array): Fixed costs for facilities.
        transport_costs (np.array): Transportation costs (facilities x customers).
        solver (str): The name of the solver to use.

    Returns:
        dict: A dictionary with the objective value of the relaxation.
    """
    num_facilities, num_customers = transport_costs.shape

    ampl = AMPL()
    ampl.setOption('solver', solver)

    # Opzioni per il determinismo + disabilitazione del presolve
    options = []
    if deterministic:
        options.append('threads=1')
        if solver == 'gurobi':
            options.append('seed=0')
        elif solver == 'cplex':
            options.append('randomseed=1234')

    # Aggiungiamo l'opzione presolve=0
    # La sintassi per opzioni multiple è separarle con uno spazio
    if 'relaxation' in 'nome_funzione':
        options.append('presolve=0')

    if options:
        ampl.setOption(f'{solver}_options', ' '.join(options))


    # Definiamo il modello con variabili continue invece che binarie
    ampl.eval(f"""
        # Sets
        set FACILITIES := 0..{num_facilities - 1};
        set CUSTOMERS := 0..{num_customers - 1};

        # Parameters
        param f {{FACILITIES}};
        param c {{FACILITIES, CUSTOMERS}};

        # Decision Variables (rilasciate)
        var x {{FACILITIES}} >= 0, <= 1;
        var y {{FACILITIES, CUSTOMERS}} >= 0, <= 1;

        # Objective Function
        minimize TotalCost:
            sum {{i in FACILITIES}} f[i] * x[i] +
            sum {{i in FACILITIES, j in CUSTOMERS}} c[i,j] * y[i,j];

        # Constraints
        subject to Demand {{j in CUSTOMERS}}:
            sum {{i in FACILITIES}} y[i,j] = 1;

        subject to Linking {{i in FACILITIES, j in CUSTOMERS}}:
            y[i,j] <= x[i];
    """)

    # Load data
    ampl.getParameter('f').setValues({i: fixed_costs[i] for i in range(num_facilities)})
    c_series = pd.DataFrame(
        transport_costs,
        index=pd.Index(range(num_facilities), name='FACILITIES'),
        columns=pd.Index(range(num_customers), name='CUSTOMERS')
    ).stack()
    ampl.getParameter('c').setValues(c_series)

    # Solve
    ampl.solve()

    objective_value = ampl.getObjective('TotalCost').value()
    ampl.close()

    return {
        'objective': objective_value,
    }