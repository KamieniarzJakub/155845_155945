import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import csv

def load_coords_profits(filename):
    coords = []
    profits = []
    if not os.path.exists(filename):
        print(f"Błąd: Nie znaleziono pliku danych instancji {filename}")
        return [], []
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 3:
                continue
            x, y, p = float(row[0]), float(row[1]), float(row[2])
            coords.append((x, y))
            profits.append(p)
    return coords, profits

def load_cycle(filepath):
    cycle = []
    if not os.path.exists(filepath):
        print(f"Błąd: Nie znaleziono pliku rozwiązania {filepath}")
        return []
    with open(filepath) as f:
        lines = f.readlines()
        start = False
        for line in lines:
            line = line.strip()
            if line.lower() == "cycle:":
                start = True
                continue
            # Dodano obsługę przecinków lub spacji, jeśli indeksy są w jednej linii
            if start:
                parts = line.replace(',', ' ').split()
                for p in parts:
                    if p.isdigit():
                        cycle.append(int(p))
    return cycle

def plot_cycle(coords, profits, cycle, title, savepath):
    if not coords or not cycle:
        return

    visited = set(cycle)
    all_indices = set(range(len(coords)))
    unvisited = list(all_indices - visited)

    # Współrzędne cyklu (zamknięcie pętli powrotem do punktu startowego)
    x = [coords[v][0] for v in cycle] + [coords[cycle[0]][0]]
    y = [coords[v][1] for v in cycle] + [coords[cycle[0]][1]]

    norm = mcolors.Normalize(vmin=min(profits), vmax=max(profits))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(12, 8))

    # Rysowanie krawędzi ścieżki
    ax.plot(x, y, '-', color='gray', linewidth=1.2, alpha=0.5, zorder=1)

    # Punkty odwiedzone (okręgi)
    sc_visited = ax.scatter(
        [coords[v][0] for v in cycle],
        [coords[v][1] for v in cycle],
        c=[profits[v] for v in cycle],
        cmap=cmap, norm=norm,
        s=80, edgecolors='black', linewidth=0.8,
        marker='o', zorder=3, label='Odwiedzone'
    )

    # Punkty nieodwiedzone (trójkąty)
    if unvisited:
        ax.scatter(
            [coords[v][0] for v in unvisited],
            [coords[v][1] for v in unvisited],
            c=[profits[v] for v in unvisited],
            cmap=cmap, norm=norm,
            s=40, edgecolors='black', linewidth=0.5,
            marker='^', alpha=0.4, zorder=2, label='Nieodwiedzone'
        )

    # Opcjonalne: Etykiety punktów (wyłączone dla czytelności przy dużych instancjach, 
    # odkomentuj jeśli potrzebujesz indeksów na mapie)
    # for v in cycle:
    #     ax.text(coords[v][0], coords[v][1], str(v), fontsize=7, zorder=4)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.6)

    cbar = fig.colorbar(sc_visited, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Profit')
    
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)

def visualize_best_files():
    # Pliki znajdują się w głównym katalogu
    for inst in ["A", "B"]:
        coords_file = f"TSPA.csv" if inst == "A" else f"TSPB.csv"
        solution_file = f"best_{inst}.txt"
        
        print(f"Przetwarzanie instancji {inst}...")
        
        coords, profits = load_coords_profits(coords_file)
        cycle = load_cycle(solution_file)
        
        if coords and cycle:
            savepath = f"visual_best_{inst}.png"
            plot_cycle(coords, profits, cycle, f"Najlepsze rozwiązanie - Instancja {inst}", savepath)
            print(f"Sukces! Wykres zapisany jako: {savepath}")
        else:
            print(f"Pominięto instancję {inst} z powodu braku plików.")

if __name__ == "__main__":
    visualize_best_files()