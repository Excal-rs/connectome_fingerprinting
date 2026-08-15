#!/usr/bin/env python3
"""
============================================================================
 CONNECTOME FINGERPRINTING — VISUALISER (figures only)
============================================================================
Run:   python visualiser.py

Reads the raw result files produced by `analyzer.py` (from ./results) and
renders every figure into ./figures. It never touches the 12 GB dataset, so
it runs on a fresh clone of the repo in seconds.

Inputs (from ./results):
  results.json, per_network.csv, per_region.csv, intelligence.csv,
  cross_task_grid.csv, similarity_matrix.npy, null_distribution.npy

Outputs (into ./figures):
  accuracy_climb.png, permutation_null.png, similarity_matrix.png,
  similarity_distributions.png, networks.png, region_map.png/.gif,
  intelligence.png, cross_task_grid.png, task_specialisation.png
============================================================================
"""
import os, json
import csv
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

IN = "results"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

N_SUBJECTS = 339
TASKS = ["MOTOR", "WM", "EMOTION", "GAMBLING", "LANGUAGE", "RELATIONAL", "SOCIAL"]

# ---- validated, colour-blind-safe palette (from the dataviz design system) ----
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; GREEN = "#008300"
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Load raw results
# ---------------------------------------------------------------------------
def load_results():
    with open(f"{IN}/results.json") as f:
        return json.load(f)


def load_similarity():
    return np.load(f"{IN}/similarity_matrix.npy").astype(float)


def load_null_distribution():
    return np.load(f"{IN}/null_distribution.npy").astype(float)


def load_per_network():
    rows = []
    with open(f"{IN}/per_network.csv") as f:
        for row in csv.DictReader(f):
            rows.append((row["network"], float(row["accuracy"]), int(row["n_edges"])))
    return rows


def load_per_region():
    with open(f"{IN}/per_region.csv") as f:
        return list(csv.DictReader(f))


def load_intelligence_csv():
    iq, self_id, identified = [], [], []
    with open(f"{IN}/intelligence.csv") as f:
        for row in csv.DictReader(f):
            iq.append(float(row["intelligence_composite"]))
            self_id.append(float(row["self_identifiability"]))
            identified.append(int(row["identified_correctly"]))
    return np.array(iq), np.array(self_id), np.array(identified, dtype=bool)


def load_cross_task_grid():
    return np.loadtxt(f"{IN}/cross_task_grid.csv", delimiter=",", skiprows=1)


# ===========================================================================
# CORE figures — accuracy climb, similarity matrix, distributions
# ===========================================================================
def fig_core(results, S):
    core = results["core"]
    chance, before, fix1, fix2 = core["chance"], core["naive"], core["fix1"], core["fix2"]

    # ---- figure 1: the climb ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = ["Naive start\n(single scan)", "+ Fix 1\n(full-day data)", "+ Fix 2\n(remove generic brain)"]
    vals, colors = [before, fix1, fix2], [ORANGE, BLUE, GREEN]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
    # above the bars (zorder 3), or the line only shows in the gaps between them
    ax.axhline(chance * 100, ls="--", lw=1.5, color=MUTED, zorder=4)
    ax.text(2.48, chance * 100 + 1.2, "random guessing (0.3%)", ha="right", va="bottom",
            fontsize=9, color="black", style="italic", zorder=4)
    ax.set_ylim(0, 100); ax.set_ylabel("Identification accuracy (%)")
    ax.set_title("Identification accuracy at each stage of the method", fontweight="bold", pad=12)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_climb.png", dpi=200); plt.close(fig)

    # ---- figure 2: similarity matrix ----
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(S, cmap=SEQ, aspect="equal", interpolation="nearest")
    ax.set_xlabel("Person's fingerprint — Day 2"); ax.set_ylabel("Person's fingerprint — Day 1")
    ax.set_title("fMRI fingerprint similarity matrix", fontweight="bold", fontsize=12, pad=10)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Fingerprint similarity", color=INK2)
    cb.outline.set_edgecolor(AXIS); cb.ax.tick_params(color=MUTED, labelcolor=MUTED, length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/similarity_matrix.png", dpi=200); plt.close(fig)

    # ---- figure 3: within vs between distributions ----
    within = np.diag(S); between = S[~np.eye(S.shape[0], dtype=bool)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.linspace(min(between.min(), within.min()), max(between.max(), within.max()), 60)
    ax.hist(between, bins=bins, density=True, color=ORANGE, alpha=0.85, label="Different people", zorder=2)
    ax.hist(within, bins=bins, density=True, color=BLUE, alpha=0.85, label="Same person, different day", zorder=3)
    ax.axvline(between.mean(), color=ORANGE, lw=1.5, ls="--"); ax.axvline(within.mean(), color=BLUE, lw=1.5, ls="--")
    ax.set_xlabel("Fingerprint similarity"); ax.set_ylabel("Relative frequency")
    ax.set_title("Within- vs between-subject fingerprint similarity", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper center"); ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{OUT}/similarity_distributions.png", dpi=200); plt.close(fig)


# ===========================================================================
# PERMUTATION TEST — the empirical null distribution
# ===========================================================================
def fig_permutation(results, null_distribution):
    chance = results["core"]["chance"]
    perm = results["permutation"]
    observed_accuracy, p_value = perm["observed_accuracy"], perm["p_value"]

    # no permutation beat the observed accuracy => p sits at the floor the test can resolve
    resolution_floor = 1 / (perm["n_perm"] + 1)
    p_label = (f"$p < {resolution_floor:.4f}$" if p_value <= resolution_floor
               else f"$p = {p_value:.4f}$")

    # ---- figure 4: observed accuracy against the shuffled-identity null ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Histogram of accuracies expected under the null hypothesis
    ax.hist(null_distribution * 100, bins=30, color=ORANGE, alpha=0.75, edgecolor="white")

    # Analytical chance level
    ax.axvline(chance * 100, color=MUTED, linestyle="--", linewidth=2,
               label=f"Analytical chance ({chance*100:.1f}%)")

    # 95th percentile of the empirical null distribution
    null95 = np.percentile(null_distribution, 95)
    ax.axvline(null95 * 100, color=BLUE, linestyle=":", linewidth=2,
               label=f"95th percentile of null ({null95*100:.1f}%)")

    # Observed fingerprinting accuracy
    ax.axvline(observed_accuracy * 100, color=GREEN, linewidth=4,
               label=f"Observed accuracy ({observed_accuracy*100:.1f}%)")

    # Annotate statistical significance
    ax.text(observed_accuracy * 100 - 1.5, ax.get_ylim()[1] * 0.97,
            f"Permutation\n{p_label}", ha="right", va="top",
            fontsize=10, color=GREEN, fontweight="bold")

    ax.set_xlabel("Identification accuracy (%)"); ax.set_ylabel("Frequency")
    ax.set_title("Permutation test of connectome fingerprinting accuracy", fontweight="bold", pad=12)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.legend(frameon=False, loc="lower center")
    fig.tight_layout(); fig.savefig(f"{OUT}/permutation_null.png", dpi=200); plt.close(fig)


# ===========================================================================
# EXTENSION 1 (H3) — which networks fingerprint best
# ===========================================================================
def fig_networks(rows):
    items = sorted(((net, acc) for net, acc, _ in rows), key=lambda kv: kv[1])
    labels = [k for k, _ in items]; vals = [v * 100 for _, v in items]
    colors = [GREEN if v == max(vals) else BLUE for v in vals]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    bars = ax.barh(labels, vals, color=colors, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=11, color=INK)
    ax.set_xlim(0, 100); ax.set_xlabel("Accuracy using only this network's connections (%)")
    ax.set_title("Identification accuracy by brain network", fontweight="bold", pad=12)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/networks.png", dpi=200); plt.close(fig)


# ===========================================================================
# EXTENSION 1b — the same result region by region, as a rotating 3D brain
# ===========================================================================
def fig_region_map(rows, n_frames=72, elevation=14):
    """Draw the 360 regions at their real MNI positions, shaded by how well each identifies people.

    Renders one frame per step of a full turn and writes them as figures/region_map.gif,
    plus a single still (region_map.png) for anywhere a GIF can't animate.
    """
    coordinates = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    accuracies = np.array([float(r["accuracy"]) for r in rows]) * 100
    region_names = np.array([r["region"] for r in rows])

    centred = coordinates - coordinates.mean(0)
    normalised_accuracy = (accuracies - accuracies.min()) / np.ptp(accuracies)
    # a cube this wide holds the brain at every angle, so nothing clips as it turns
    radius = np.linalg.norm(centred, axis=1).max() * 1.05

    # a wireframe hull per hemisphere, faint but enough to read the cloud as a head
    hull_edges = []
    for hemisphere in (centred[coordinates[:, 0] > 0], centred[coordinates[:, 0] < 0]):
        hull = ConvexHull(hemisphere)
        edges = {tuple(sorted(pair)) for face in hull.simplices
                 for pair in ((face[0], face[1]), (face[1], face[2]), (face[0], face[2]))}
        hull_edges += [[hemisphere[a], hemisphere[b]] for a, b in edges]
    hull_edges = np.array(hull_edges)

    fig = plt.figure(figsize=(7.0, 5.5), dpi=100)
    ax = fig.add_subplot(projection="3d")
    # the axes rect runs off-canvas on purpose: the cube's empty corners spill outside
    # the frame instead of shrinking the brain, and only the brain itself stays in view
    ax.set_position([-0.11, -0.02, 1.23, 1.02])

    cax = fig.add_axes([0.30, 0.10, 0.40, 0.022])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(accuracies.min(), accuracies.max()), cmap=SEQ),
                      cax=cax, orientation="horizontal")
    cb.set_label("Identification accuracy using only this region's connections (%)",
                 fontsize=9, color=INK2, labelpad=6)
    cb.outline.set_edgecolor(AXIS); cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED, length=0, labelsize=9)

    fig.text(0.5, 0.94, "Identification accuracy by brain region", ha="center",
             fontsize=15, fontweight="bold", color=INK)
    caption = fig.text(0.035, 0.16, "", fontsize=9.5, color=MUTED, ha="left")

    side_at_azimuth = [(0, "right"), (90, "front"), (180, "left"), (270, "back"), (360, "right")]

    def draw(azimuth):
        ax.clear(); ax.set_axis_off()
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_box_aspect((1, 1, 1), zoom=1.55)
        ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius); ax.set_zlim(-radius, radius)
        facing = min(side_at_azimuth, key=lambda side: abs(azimuth % 360 - side[0]))[1]
        caption.set_text(f"viewing from the {facing}")

        elevation_radians, azimuth_radians = np.radians(elevation), np.radians(azimuth)
        towards_camera = np.array([np.cos(elevation_radians) * np.cos(azimuth_radians),
                                   np.cos(elevation_radians) * np.sin(azimuth_radians),
                                   np.sin(elevation_radians)])
        depth = centred @ towards_camera
        nearness = (depth - depth.min()) / np.ptp(depth)

        ax.add_collection3d(Line3DCollection(hull_edges, colors=AXIS, linewidths=0.8, alpha=0.5))

        dot_colors = SEQ(normalised_accuracy)
        dot_colors[:, 3] = 0.35 + 0.65 * nearness ** 1.2
        dot_sizes = (16 + 300 * normalised_accuracy ** 1.5) * (0.55 + 0.6 * nearness)
        back_to_front = np.argsort(depth)
        ax.scatter(*centred[back_to_front].T, s=dot_sizes[back_to_front], c=dot_colors[back_to_front],
                   edgecolors=SURFACE, linewidths=0.4, depthshade=False)

    frames = []
    for azimuth in np.linspace(0, 360, n_frames, endpoint=False):
        draw(azimuth)
        fig.canvas.draw()
        frames.append(Image.frombytes("RGBA", fig.canvas.get_width_height(),
                                      fig.canvas.buffer_rgba()).convert("RGB"))
    # one shared palette, or the colours shimmer as the frames cycle
    palette = frames[0].quantize(colors=192)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    # 70 ms a frame — 72 frames make a ~5 s turn, slow enough to read region by region
    frames[0].save(f"{OUT}/region_map.gif", save_all=True, append_images=frames[1:],
                   duration=70, loop=0, optimize=True)

    draw(20)   # near-lateral, the most brain-shaped angle of the turn
    fig.savefig(f"{OUT}/region_map.png", dpi=200)
    plt.close(fig)


# ===========================================================================
# EXTENSION 3 — identifiability vs intelligence
# ===========================================================================
def fig_intelligence(results, iq, self_id, identified):
    intel = results["intelligence"]
    rP, pP = intel["pearson_r"], intel["pearson_p"]

    valid = np.isfinite(iq)
    x, y = iq[valid], self_id[valid]
    hit = identified[valid]
    hi = iq[valid & identified]; lo = iq[valid & ~identified]

    # ---- figure 7: scatter + regression, points coloured by hit/miss ----
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax.scatter(x[hit], y[hit], s=42, color=BLUE, alpha=0.75, edgecolor=SURFACE,
               linewidth=0.6, zorder=3, label="identified correctly")
    ax.scatter(x[~hit], y[~hit], s=52, color=ORANGE, alpha=0.9, edgecolor=SURFACE,
               linewidth=0.6, zorder=4, marker="D", label="missed")
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, a + b * xs, color=INK, lw=2, zorder=5)
    ax.text(0.04, 0.95, f"Pearson r = {rP:+.2f}\n(p = {pP:.2g}, n = {valid.sum()})",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=AXIS, lw=1))
    ax.set_xlabel("Cognitive / intelligence composite (z-scored)")
    ax.set_ylabel("Self-identifiability (SDs above impostors)")
    ax.set_title("Self-identifiability vs cognitive composite", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)

    # right panel: IQ of correctly-identified vs missed
    parts = [lo, hi] if lo.size else [hi]
    labels = ([f"Missed\n(n = {lo.size})", f"Identified\n(n = {hi.size})"] if lo.size
              else [f"Identified\n(n = {hi.size})"])
    cols = ([ORANGE, BLUE] if lo.size else [BLUE])
    vp = ax2.violinplot(parts, showmeans=True, showextrema=False)
    for body, c in zip(vp["bodies"], cols):
        body.set_facecolor(c); body.set_alpha(0.35); body.set_edgecolor(c)
    vp["cmeans"].set_color(INK); vp["cmeans"].set_linewidth(2)
    for k, (data, c) in enumerate(zip(parts, cols), start=1):
        jit = k + (np.random.RandomState(0).rand(data.size) - 0.5) * 0.16
        ax2.scatter(jit, data, s=16, color=c, alpha=0.5, zorder=3)
    ax2.set_xticks(range(1, len(labels) + 1)); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Intelligence composite (z)")
    ax2.set_title("Cognitive composite by identification outcome",
                  fontweight="bold", fontsize=11, pad=12)
    ax2.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); ax2.set_axisbelow(True); ax2.tick_params(length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/intelligence.png", dpi=200); plt.close(fig)


# ===========================================================================
# EXTENSION 2 (H2) — cross-task grid + specialisation
# ===========================================================================
def fig_cross_task(results, grid):
    cross = results["cross_task"]

    # ---- figure 5: cross-task heatmap ----
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid * 100, cmap=SEQ, vmin=0, vmax=100)
    ax.set_xticks(range(7)); ax.set_xticklabels(TASKS, rotation=45, ha="right")
    ax.set_yticks(range(7)); ax.set_yticklabels(TASKS)
    ax.set_xlabel("Database task (session B)"); ax.set_ylabel("Query task (session A)")
    ax.set_title("Cross-task identification matrix\n(numbers = % of people correctly identified)",
                 fontweight="bold", fontsize=12, pad=10)
    for i in range(7):
        for j in range(7):
            v = grid[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if v > 45 else INK)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Identification accuracy (%)", color=INK2)
    cb.outline.set_edgecolor(AXIS); cb.ax.tick_params(color=MUTED, labelcolor=MUTED, length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/cross_task_grid.png", dpi=200); plt.close(fig)

    # ---- figure 6: accuracy vs scan length, labelled by task ----
    diag = np.array([cross["diag"][t] for t in TASKS])
    lengths = np.array([cross["lengths"][t] for t in TASKS])
    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.scatter(lengths, diag * 100, s=90, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for t in range(7):
        label = TASKS[t] if TASKS[t] == "WM" else TASKS[t].title()
        ax.annotate(label, (lengths[t], diag[t] * 100),
                    textcoords="offset points", xytext=(8, 4), fontsize=10, color=INK2)
    ax.set_xlabel("Scan length (timepoints) — i.e. how much data")
    ax.set_ylabel("Within-task identification accuracy (%)")
    ax.set_title("Within-task identification accuracy vs scan length",
                 fontweight="bold", fontsize=11.5, pad=12)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.margins(0.15)
    fig.tight_layout(); fig.savefig(f"{OUT}/task_specialisation.png", dpi=200); plt.close(fig)


# ===========================================================================
if __name__ == "__main__":
    results = load_results()
    S = load_similarity()
    iq, self_id, identified = load_intelligence_csv()

    fig_core(results, S)
    fig_permutation(results, load_null_distribution())
    fig_networks(load_per_network())
    fig_region_map(load_per_region())
    fig_intelligence(results, iq, self_id, identified)
    fig_cross_task(results, load_cross_task_grid())

    print(f"Done. Figures written to ./{OUT}/")
    print("  accuracy_climb, permutation_null, similarity_matrix, similarity_distributions,")
    print("  networks, region_map, intelligence, cross_task_grid, task_specialisation  (.png)")
    print("  region_map.gif — the 360-region map as a rotating brain")
