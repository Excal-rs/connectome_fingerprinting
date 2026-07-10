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
  results.json, per_network.csv, intelligence.csv,
  cross_task_grid.csv, similarity_matrix.npy

Outputs (into ./figures):
  accuracy_climb.png, similarity_matrix.png, similarity_distributions.png,
  networks.png, intelligence.png, cross_task_grid.png, task_specialisation.png
============================================================================
"""
import os, json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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


def load_per_network():
    rows = []
    with open(f"{IN}/per_network.csv") as f:
        for row in csv.DictReader(f):
            rows.append((row["network"], float(row["accuracy"]), int(row["n_edges"])))
    return rows


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
    chance, before, fix12, fix123 = core["chance"], core["naive"], core["fix12"], core["fix123"]

    # ---- figure 1: the climb ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = ["Naive start\n(single scan)", "+ Fix 1 & 2\n(full-day data)", "+ Fix 3\n(remove generic brain)"]
    vals, colors = [before, fix12, fix123], [ORANGE, BLUE, GREEN]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
    ax.axhline(chance * 100, ls="--", lw=1.5, color=MUTED, zorder=2)
    ax.text(2.48, chance * 100 + 1.2, "random guessing (0.3%)", ha="right", va="bottom",
            fontsize=9, color=MUTED, style="italic")
    ax.set_ylim(0, 100); ax.set_ylabel("Identification accuracy (%)")
    ax.set_title("From 31% to 92%: three fixes to brain fingerprinting", fontweight="bold", pad=12)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_climb.png", dpi=200); plt.close(fig)

    # ---- figure 2: similarity matrix ----
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(S, cmap=SEQ, aspect="equal", interpolation="nearest")
    ax.set_xlabel("Person's fingerprint — Day 2"); ax.set_ylabel("Person's fingerprint — Day 1")
    ax.set_title("Each person matches themselves across days\n(bright diagonal = correct matches)",
                 fontweight="bold", fontsize=12, pad=10); ax.tick_params(length=0)
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
    ax.set_title("Why it works: your two days look alike; two people don't", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper center"); ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{OUT}/similarity_distributions.png", dpi=200); plt.close(fig)


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
    ax.set_title("Higher-order networks fingerprint best", fontweight="bold", pad=12)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(f"{OUT}/networks.png", dpi=200); plt.close(fig)


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
    ax.set_title("Are more able people more identifiable?", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)

    # right panel: IQ of correctly-identified vs missed
    parts = [lo, hi] if lo.size else [hi]
    labels = (["Missed", "Identified"] if lo.size else ["Identified"])
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
    ax2.set_title(f"Miss rate: {(~hit).sum()}/{valid.sum()}", fontweight="bold", fontsize=11, pad=12)
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
    ax.set_title("Identity survives a change of task\n(numbers = % of people correctly identified)",
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
    ax.set_title("Identifiability isn't just 'more data'\nHigher-order tasks lead; working-memory lags despite the most data",
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
    fig_networks(load_per_network())
    fig_intelligence(results, iq, self_id, identified)
    fig_cross_task(results, load_cross_task_grid())

    print(f"Done. Figures written to ./{OUT}/")
    print("  accuracy_climb, similarity_matrix, similarity_distributions,")
    print("  networks, intelligence, cross_task_grid, task_specialisation  (.png)")
