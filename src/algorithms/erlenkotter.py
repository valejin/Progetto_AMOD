import numpy as np


def solve_erlenkotter(fixed_costs, transport_costs):
    """
    Implements Erlenkotter's Dual Ascent procedure to compute a dual
    lower bound for the UFL problem.
    """
    num_facilities, num_customers = transport_costs.shape

    # --- Pre-computation for Dual Ascent ---
    # Per ogni cliente j, ordiniamo i costi di trasporto c_ij in modo crescente.
    # Questo è necessario per trovare c_j^k (il k-esimo costo più basso).
    # sorted_indices mantiene traccia degli indici originali delle facility.
    sorted_indices = np.argsort(transport_costs, axis=0)
    sorted_costs = np.take_along_axis(transport_costs, sorted_indices, axis=0)

    # --- DUAL ASCENT PROCEDURE ---

    # Step 1: Initialization
    # Inizializziamo v_j al costo della facility più economica per il cliente j (c_j^1)
    v = sorted_costs[0, :].copy() # Usiamo .copy() per evitare modifiche inattese

    # Calcoliamo il surplus (slack) iniziale s_i per ogni facility
    savings = np.maximum(0, v - transport_costs).sum(axis=1)
    surplus = fixed_costs - savings

    # k_j tiene traccia dell'indice del "prossimo miglior costo" per il cliente j.
    # In 0-based indexing, k=1 significa il secondo costo più basso (sorted_costs[1, j]).
    k_j = np.ones(num_customers, dtype=int)

    # Gestione delle parità (ties): se v_j è già uguale al prossimo miglior costo, incrementiamo k_j
    for j in range(num_customers):
        while k_j[j] < num_facilities and np.isclose(v[j], sorted_costs[k_j[j], j]):
            k_j[j] += 1

    # Step 2: Loop principale
    while True:
        delta_flag = False

        # Loop sui clienti j = 0 a n-1
        for j in range(num_customers):

            # Step 3: Se non possiamo più aumentare v_j, saltiamo il cliente
            if k_j[j] >= num_facilities:
                continue

            # Step 4: Calcola il primo candidato per l'aumento Delta_j
            # Troviamo le facility 'i' che contribuiscono a v_j (dove v_j >= c_ij)
            contributing_mask = (v[j] - transport_costs[:, j]) >= -1e-9 # Tolleranza per float

            if not np.any(contributing_mask):
                # Caso raro, se nessuna facility contribuisce, non possiamo aumentare basandoci sul surplus
                delta_j = np.inf
            else:
                # Delta_j è il surplus minimo tra le facility che contribuiscono
                delta_j = np.min(surplus[contributing_mask])

            # Step 5: Limita Delta_j in base al prossimo miglior costo
            gap_to_next_cost = sorted_costs[k_j[j], j] - v[j]

            if delta_j > gap_to_next_cost:
                delta_j = gap_to_next_cost
                # Se abbiamo modificato delta_j, incrementiamo k_j per il prossimo giro
                # e segnaliamo che una modifica importante è avvenuta in questo passo
                if delta_j > 1e-9:
                    delta_flag = True
                    k_j[j] += 1

            # Step 6: Applica l'aumento Delta_j
            if delta_j > 1e-9:
                # Diminuisci il surplus per le facility che contribuiscono
                surplus[contributing_mask] -= delta_j
                # Aumenta la variabile duale v_j
                v[j] += delta_j
                # Qualsiasi aumento di v_j significa che il processo non è ancora stabile
                delta_flag = True

        # Step 8: Se nessuna variabile v_j è stata modificata in un intero ciclo, termina.
        # Altrimenti, ricomincia dal passo 2 (il nostro `while True` loop).
        if not delta_flag:
            break

    # --- FINE DUAL ASCENT ---

    # --- CALCOLO DEL LOWER BOUND DUALE ---
    # Il valore dell'obiettivo duale è la somma dei v_j
    dual_lower_bound = np.sum(v)


    # Restituisce SOLO il lower bound come 'objective'
    return {
        'objective': dual_lower_bound
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