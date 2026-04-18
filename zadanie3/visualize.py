import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import csv

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

def plot_cycle(coords, profits, cycle, title, savepath):

    visited = set(cycle)
    all_indices = set(range(len(coords)))
    unvisited = list(all_indices - visited)

    # visited path coordinates (close the cycle)
    x = [coords[v][0] for v in cycle] + [coords[cycle[0]][0]]
    y = [coords[v][1] for v in cycle] + [coords[cycle[0]][1]]

    # unified color scale
    norm = mcolors.Normalize(vmin=min(profits), vmax=max(profits))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(10, 6))

    # line for visited (path)
    ax.plot(x, y, '-', color='gray', linewidth=1.2, zorder=1)

    # visited points with circle marker
    sc_visited = ax.scatter(
        [coords[v][0] for v in visited],
        [coords[v][1] for v in visited],
        c=[profits[v] for v in visited],
        cmap=cmap, norm=norm,
        s=70, edgecolors='black', linewidth=0.6,
        marker='o', zorder=3
    )

    # unvisited points with square marker
    sc_unvisited = ax.scatter(
        [coords[v][0] for v in unvisited],
        [coords[v][1] for v in unvisited],
        c=[profits[v] for v in unvisited],
        cmap=cmap, norm=norm,
        s=30, edgecolors='black', linewidth=0.6,
        marker='^', alpha=0.85, zorder=2
    )

    # labels for all points
    for v in visited:
        ax.text(coords[v][0], coords[v][1], str(v), fontsize=8, color='black', zorder=4)
    for v in unvisited:
        ax.text(coords[v][0], coords[v][1], str(v), fontsize=7, color='black', zorder=4)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True)

    # colorbar showing meaning of color for BOTH visited and unvisited
    cbar = fig.colorbar(sc_visited, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Profit')

    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)

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

if __name__ == "__main__":
    for inst in ["A", "B"]:
        visualize_all(inst)