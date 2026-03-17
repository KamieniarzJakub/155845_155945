import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import csv
import math

# ------------------------------
# Funkcja do wczytania współrzędnych i zysków
# ------------------------------
def load_coords_profits(filename):
    coords = []
    profits = []
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 3:
                continue
            x, y, p = float(row[0]), float(row[1]), float(row[2])
            coords.append((x, y))
            profits.append(p)
    return coords, profits

# ------------------------------
# Funkcja do wczytania cyklu z pliku txt
# ------------------------------
def load_cycle(filepath):
    cycle = []
    with open(filepath) as f:
        lines = f.readlines()
        start = False
        for line in lines:
            line = line.strip()
            if line == "cycle:":
                start = True
                continue
            if start and line.isdigit():
                cycle.append(int(line))
    return cycle

# ------------------------------
# Funkcja do rysowania cyklu z wielkością wierzchołków
# ------------------------------
def plot_cycle(coords, profits, cycle, title, savepath):
    x = [coords[v][0] for v in cycle] + [coords[cycle[0]][0]]
    y = [coords[v][1] for v in cycle] + [coords[cycle[0]][1]]

    # Normalizacja zysków do 0-1 dla mapy kolorów
    norm = mcolors.Normalize(vmin=min(profits), vmax=max(profits))
    cmap = cm.viridis
    colors = [cmap(norm(profits[v])) for v in cycle]

    fig, ax = plt.subplots(figsize=(10,6))  # jawny obiekt Axes

    # Rysowanie cyklu
    ax.plot(x, y, '-o', markersize=5, color='blue', zorder=1)
    # Rysowanie wierzchołków kolorowych wg zysków
    sc = ax.scatter([coords[v][0] for v in cycle],
                    [coords[v][1] for v in cycle],
                    c=[profits[v] for v in cycle], cmap=cmap, norm=norm,
                    s=50, zorder=2)
    for i, v in enumerate(cycle):
        ax.text(coords[v][0], coords[v][1], str(v), fontsize=8, color='black', zorder=3)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True)

    # Dodanie legendy po prawej stronie
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Profit')

    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
    
# ------------------------------
# Główna funkcja do wizualizacji wszystkich metod
# ------------------------------
def visualize_all(inst_tag):
    base_path = f"output/{inst_tag}"
    coords_file = f"TSP{inst_tag}.csv"
    coords, profits = load_coords_profits(coords_file)

    methods_path = os.path.join(base_path, "solutions")
    plot_dir = os.path.join(base_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for fname in os.listdir(methods_path):
        if fname.endswith(".txt"):
            method = fname.replace(".txt", "")
            cycle = load_cycle(os.path.join(methods_path, fname))
            savepath = os.path.join(plot_dir, f"{method}.png")
            plot_cycle(coords, profits, cycle, f"{method} - {inst_tag}", savepath)
            print(f"Zapisano wykres: {savepath}")

# ------------------------------
# Wywołanie dla obu instancji
# ------------------------------
if __name__ == "__main__":
    for inst in ["A","B"]:
        visualize_all(inst)