import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# Konfiguracja eksperymentu
instances = ["A", "B"]
neighborhoods = ["ESWAP", "VSWAP"]

def save_plot(df, inst, swap, metric_col, ylabel, filename):
    plt.figure(figsize=(10, 6))
    
    # Usuwamy ewentualne wartości NaN
    data = df.dropna(subset=['objective', metric_col])
    
    # Obliczanie korelacji
    corr, _ = pearsonr(data['objective'], data[metric_col])
    
    # Rysowanie wykresu punktowego z linią regresji
    sns.regplot(
        data=data, 
        x='objective', 
        y=metric_col, 
        scatter_kws={'alpha':0.5, 's':15, 'color': 'royalblue'},
        line_kws={'color': 'red', 'label': f'Korelacja: {corr:.4f}'}
    )
    
    plt.title(f"Test Globalnej Wypukłości - Instancja {inst} ({swap})\nMiara: {ylabel}")
    plt.xlabel("Wartość funkcji celu (Profit - Length)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Zapisywanie
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Zapisano: {filename} (r = {corr:.4f})")

def main():
    sns.set_theme(style="whitegrid")
    
    for inst in instances:
        for swap in neighborhoods:
            csv_file = f"convexity_{inst}_{swap}.csv"
            
            if not os.path.exists(csv_file):
                print(f"Błąd: Brak pliku {csv_file}")
                continue
            
            df = pd.read_csv(csv_file)
            
            # 1. Podobieństwo wierzchołków do Najlepszego
            save_plot(df, inst, swap, 'nodes_to_best', 
                      "Liczba wspólnych wierzchołków (do Best)", 
                      f"plot_{inst}_{swap}_nodes_best.png")
            
            # 2. Podobieństwo krawędzi do Najlepszego
            save_plot(df, inst, swap, 'edges_to_best', 
                      "Liczba wspólnych krawędzi (do Best)", 
                      f"plot_{inst}_{swap}_edges_best.png")
            
            # 3. Średnie podobieństwo wierzchołków do pozostałych
            save_plot(df, inst, swap, 'nodes_avg', 
                      "Średnie podobieństwo wierzchołków (do innych)", 
                      f"plot_{inst}_{swap}_nodes_avg.png")
            
            # 4. Średnie podobieństwo krawędzi do pozostałych
            save_plot(df, inst, swap, 'edges_avg', 
                      "Średnie podobieństwo krawędzi (do innych)", 
                      f"plot_{inst}_{swap}_edges_avg.png")

if __name__ == "__main__":
    main()