import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_instance_summary(df, instance_name, num_facilities, num_customers, output_dir='reports'):
    """Crea un grafico a barre per una singola istanza."""

    instance_df = df[df['instance'] == instance_name].copy()

    # Controlla se ci sono dati per questa istanza
    if instance_df.empty:
        print(f"No data found for instance {instance_name}. Skipping plot.")
        return

    # Gestiamo il caso in cui l'ottimo non sia stato trovato.
    try:
        optimal_value = instance_df[instance_df['algorithm'].isin(['strong', 'weak'])]['objective'].iloc[0]
    except IndexError:
        print(f"Optimal value not found for {instance_name}. Cannot plot reference line.")
        optimal_value = None


    # Aggiungiamo una colonna per il tipo di valore
    def get_type(algo):
        if algo in ['strong', 'weak']:
            return 'Optimal'
        if algo == 'greedy':  # Solo greedy è il upper bound euristico
            return 'Upper Bound (Heuristic)'
        # erlenkotter è un lower bound
        if 'rl' in algo or algo == 'erlenkotter':
            return 'Lower Bound'
        return 'Other'

    instance_df['type'] = instance_df['algorithm'].apply(get_type)

    # Ordiniamo per visualizzazione
    algo_order = ['strong', 'weak', 'greedy', 'strong_rl', 'weak_rl', 'erlenkotter']
    instance_df['algorithm'] = pd.Categorical(instance_df['algorithm'], categories=algo_order, ordered=True)
    instance_df = instance_df.sort_values('algorithm')

    # Creiamo il grafico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.barplot(
        data=instance_df,
        x='objective',
        y='algorithm',
        hue='type',
        palette={'Optimal': '#2ca02c', 'Upper Bound (Heuristic)': '#ff7f0e', 'Lower Bound': '#1f77b4'},
        dodge=False,
        ax=ax
    )

    # Aggiunge una linea verticale per l'ottimo
    if optimal_value is not None:
        ax.axvline(x=optimal_value, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal Value ({optimal_value:,.2f})')

    title_str = (
        f'Performance Analysis for Instance: {instance_name}\n'
        f'({num_facilities} Facilities, {num_customers} Customers)'
    )
    ax.set_title(title_str, fontsize=16, weight='bold')


    ax.set_xlabel('Objective Function Value (Cost)', fontsize=12)
    ax.set_ylabel('Algorithm / Bound', fontsize=12)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x)))

    # Aggiunge le etichette con i valori sulle barre
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.2f}', padding=5, fontsize=10, rotation=0)


    ax.legend(title='Value Type', bbox_to_anchor=(1.1, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # [left, bottom, right, top]

    # --- LOGICA DI SALVATAGGIO NELLE SOTTOCARTELLE ---
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Salva il grafico
    plot_filename = os.path.join(output_dir, f'{instance_name.replace(".txt", "")}_summary.png')
    plt.savefig(plot_filename)
    plt.close(fig)  # Chiudi la figura per liberare memoria
    print(f"Plot saved to {plot_filename}")


def find_instance_subdir(instance_name, root_dir='data'):
    """
    Trova la sottocartella di un'istanza (es. 'small', 'medio') per salvare il grafico.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        if instance_name in filenames:
            # Calcola il percorso relativo della cartella
            relative_dir = os.path.relpath(dirpath, root_dir)
            if relative_dir == '.':
                return ''  # Nessuna sottocartella
            return relative_dir
    return ''  # Default se non trovato


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots from UFL results CSV file.")
    parser.add_argument('csv_file', type=str, help='Path to the ufl_results_detailed.csv file.')
    parser.add_argument('--instance', type=str, default=None,
                        help='Generate plot for a single specific instance name. If not provided, plots for all instances are generated.')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: '{args.csv_file}' not found. Please provide a valid path.")
        exit()

    if args.instance:
        print(f"\nGenerating plot for specific instance: {args.instance}...")
        instance_data = df[df['instance'] == args.instance]
        if not instance_data.empty:
            n_fac = int(instance_data['num_facilities'].iloc[0])
            n_cust = int(instance_data['num_customers'].iloc[0])
            subdir = find_instance_subdir(args.instance, root_dir='data')
            output_directory = os.path.join('reports', subdir)
            plot_instance_summary(df, args.instance, n_fac, n_cust, output_dir=output_directory)
        else:
            print(f"No data found for instance {args.instance} in CSV file.")
    else:
        for instance in df['instance'].unique():
            print(f"\nGenerating plot for {instance}...")
            instance_data = df[df['instance'] == instance]
            n_fac = int(instance_data['num_facilities'].iloc[0])
            n_cust = int(instance_data['num_customers'].iloc[0])
            subdir = find_instance_subdir(instance, root_dir='data')
            output_directory = os.path.join('reports', subdir)
            plot_instance_summary(df, instance, n_fac, n_cust, output_dir=output_directory)