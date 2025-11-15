import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def plot_performance_comparison(df, category_name, output_dir='reports_scalability'):
    """
    Crea un grafico a barre raggruppato per confrontare i tempi di esecuzione
    per un gruppo di istanze di dimensione simile.
    """
    df_successful = df[df['objective'] != 'Error'].copy()
    df_successful['time_sec'] = pd.to_numeric(df_successful['time_sec'])

    # Creiamo il grafico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.barplot(
        data=df_successful,
        x='instance',
        y='time_sec',
        hue='algorithm',
        ax=ax
    )

    ax.set_yscale('log')  # La scala logaritmica

    ax.set_title(f'Performance Comparison for "{category_name}" Instances', fontsize=16, weight='bold')
    ax.set_xlabel('Instance Name', fontsize=12)
    ax.set_ylabel('Computational Time (seconds, log scale)', fontsize=12)

    # Ruota le etichette dell'asse X se sono troppe e si sovrappongono
    plt.xticks(rotation=45, ha='right')

    ax.legend(title='Algorithm', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Salviamo il grafico
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'performance_comparison_{category_name}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)

    print(f"Performance comparison plot for '{category_name}' saved to {plot_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots from UFL results CSV file.")
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

    # Aggiungiamo una colonna 'category' basata sulla dimensione dell'istanza
    df['instance_size'] = df['num_facilities'] * df['num_customers']


    def assign_category(size):
        if size <= 1500:
            return 'small'
        elif size <= 10000:
            return 'medio'
        else:
            return 'big'


    # Applichiamo la categorizzazione a ogni riga (lo facciamo sul df completo una volta sola)
    # Prendiamo la dimensione per ogni istanza unica per evitare duplicati
    instance_sizes = df.groupby('instance')['instance_size'].first()
    instance_to_category = instance_sizes.apply(assign_category).to_dict()
    df['category'] = df['instance'].map(instance_to_category)

    # Ordiniamo gli algoritmi per una legenda e colori consistenti
    algo_order = ['strong', 'weak', 'strong_rl', 'weak_rl', 'erlenkotter', 'greedy']
    df['algorithm'] = pd.Categorical(df['algorithm'], categories=algo_order, ordered=True)

    # Genera un grafico a barre per ogni categoria
    for category, group_df in df.groupby('category'):
        print(f"\nGenerating performance plot for '{category}' category...")
        plot_performance_comparison(group_df, category, output_dir='reports_scalability')