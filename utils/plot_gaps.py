import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def plot_gap_analysis_boxplot(df, category_name, output_dir='reports'):
    """Crea un box plot per confrontare le distribuzioni dei gap."""

    df_gaps = df[df['optimality_gap_%'] != '-'].copy()
    df_gaps['optimality_gap_%'] = pd.to_numeric(df_gaps['optimality_gap_%'])

    if df_gaps.empty:
        print(f"No gap data to plot for category '{category_name}'.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Ordiniamo gli algoritmi per una visualizzazione logica
    algo_order = ['weak_rl', 'greedy', 'strong_rl', 'erlenkotter']

    sns.boxplot(
        data=df_gaps,
        x='algorithm',
        y='optimality_gap_%',
        order=algo_order,  # Applica l'ordine
        palette='viridis',
        ax=ax
    )
    # Per un'alternativa visiva, puoi usare: sns.violinplot(...)

    ax.set_title(f'Gap Distribution for "{category_name}" Instances', fontsize=16, weight='bold')
    ax.set_xlabel('Algorithm / Bound', fontsize=12)
    ax.set_ylabel('Gap (%)', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'gap_boxplot_{category_name}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)

    print(f"Gap box plot for '{category_name}' saved to {plot_filename}")


def plot_gap_scalability(df, output_dir='reports_gaps'):
    """Crea uno scatter plot per analizzare la scalabilità dei gap."""

    df_gaps = df[df['optimality_gap_%'] != '-'].copy()
    df_gaps['optimality_gap_%'] = pd.to_numeric(df_gaps['optimality_gap_%'])
    df_gaps['instance_size'] = df_gaps['num_facilities'] * df_gaps['num_customers']

    if df_gaps.empty:
        print("No gap data for scalability plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.scatterplot(
        data=df_gaps,
        x='instance_size',
        y='optimality_gap_%',
        hue='algorithm',
        style='algorithm',  # Forme diverse per ogni algoritmo
        s=100,  # Dimensione dei punti
        ax=ax
    )

    ax.set_title('Scalability of Gaps vs. Instance Size', fontsize=16, weight='bold')
    ax.set_xlabel('Instance Size (Facilities x Customers)', fontsize=12)
    ax.set_ylabel('Gap (%)', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    ax.legend(title='Algorithm / Bound', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, 'scalability_analysis_gaps.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)

    print(f"Gap scalability plot saved to {plot_filename}")


def plot_gap_analysis(df, category_name, output_dir='reports_gaps'):
    """
    Crea un grafico a barre per confrontare i gap di ottimalità/dualità
    per un gruppo di istanze.
    """
    # Filtra i dati rilevanti: solo quelli per cui abbiamo calcolato un gap
    df_gaps = df[df['optimality_gap_%'] != '-'].copy()
    df_gaps['optimality_gap_%'] = pd.to_numeric(df_gaps['optimality_gap_%'])

    # Assegniamo un tipo a ogni algoritmo per colorare le barre in modo diverso
    def get_gap_type(algo):
        if algo == 'greedy':
            return 'Optimality Gap (Upper Bound)'
        else:  # strong_rl, weak_rl, erlenkotter
            return 'Duality Gap (Lower Bound)'

    df_gaps['gap_type'] = df_gaps['algorithm'].apply(get_gap_type)

    if df_gaps.empty:
        print(f"No gap data to plot for category '{category_name}'.")
        return

    # Creiamo il grafico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.barplot(
        data=df_gaps,
        x='instance',
        y='optimality_gap_%',
        hue='algorithm',
        palette='viridis',  # Usiamo una palette di colori diversa
        ax=ax
    )

    ax.set_title(f'Gap Analysis for "{category_name}" Instances', fontsize=16, weight='bold')
    ax.set_xlabel('Instance Name', fontsize=12)
    ax.set_ylabel('Gap (%)', fontsize=12)

    # Aggiungi una linea orizzontale a y=0 per riferimento
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Formatta l'asse Y per mostrare il simbolo %
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

    plt.xticks(rotation=45, ha='right')

    ax.legend(title='Algorithm / Bound', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Salviamo il grafico
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'gap_analysis_{category_name}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)

    print(f"Gap analysis plot for '{category_name}' saved to {plot_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate gap analysis plots from UFL results CSV file.")
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the ufl_results_detailed.csv file.'
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: '{args.csv_file}' not found. Please provide a valid path.")
        exit()

    # Categorizzazione delle istanze (identica a plot_scalability.py)
    df['instance_size'] = df['num_facilities'] * df['num_customers']


    def assign_category(size):
        if size <= 1500:  # Soglia per 'small'
            return 'small'
        elif size <= 2600:  # Soglia per 'medio'
            return 'medio'
        else:
            return 'big'


    instance_sizes = df.groupby('instance')['instance_size'].first()
    instance_to_category = instance_sizes.apply(assign_category).to_dict()
    df['category'] = df['instance'].map(instance_to_category)

    # Genera un grafico dei gap per ogni categoria
    for category, group_df in df.groupby('category'):
        print(f"\nGenerating gap plot for '{category}' category...")
        plot_gap_analysis_boxplot(group_df, category, output_dir='reports_gaps')
        plot_gap_analysis(group_df, category, output_dir='reports_gaps')

    plot_gap_scalability(df)