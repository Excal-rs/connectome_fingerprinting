# -*- coding: utf-8 -*-
"""
Connectome Fingerprinting — STEP-BY-STEP version for Spyder
=============================================================
This is the same analysis as the original analyzer.py. Nothing about the
math or the results has changed — only the *organisation*:

  - Every stage now runs as its own Spyder "cell" (separated by `# %%`).
  - Each cell leaves its result sitting in a plain variable, so you can
    look at it in the Variable Explorer right after running that cell
    (arrays, and pandas DataFrames for anything table-shaped).
  - Comments explain what each block does and why.

HOW TO USE IN SPYDER
  - Click inside a cell (the block between two `# %%` lines) and press
    Ctrl+Enter to run just that cell. Shift+Enter runs it and jumps to
    the next one.
  - Run the cells top to bottom, in order — later cells reuse variables
    created by earlier ones (e.g. the per-network cell in CELL 10 needs
    day1_personal / day2_personal, which are created in CELL 8).
  - You can still press F5 to run the whole file start to finish, same
    as running the original script.

Outputs written into ./results (identical filenames to the original, so
visualiser.py still works unchanged):
  results.json, per_network.csv, intelligence.csv, cross_task_grid.csv,
  similarity_matrix.npy
"""

# %% CELL 0 — Imports and working directory
# ---------------------------------------------------------------------------
import os
import json
import urllib.request
import tarfile
import numpy as np
import pandas as pd
from scipy import stats

# Set the working folder FIRST, before anything else touches the filesystem.
os.chdir("C:/Users/aroma/nma")
print("Working directory is now:", os.getcwd())


# %% CELL 1 — Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data"          # raw HCP data lives here (downloaded if missing)
RESULTS_DIR = "results"    # everything we compute gets written here
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SUBJECTS = 339                                  # subjects in the Neuromatch HCP set
N_PARCELS = 360                                   # brain summarised into 360 regions (Glasser atlas)
UPPER_TRI_IDX = np.triu_indices(N_PARCELS, k=1)   # the 64,620 unique region-pairs = one "fingerprint"
subject_ids = list(range(N_SUBJECTS))

TASKS = ["MOTOR", "WM", "EMOTION", "GAMBLING", "LANGUAGE", "RELATIONAL", "SOCIAL"]

OSF_URLS = {"hcp_rest": "https://osf.io/bqp7m/download",
            "hcp_task": "https://osf.io/s4h8j/download"}

print(f"{N_SUBJECTS} subjects x {len(UPPER_TRI_IDX[0])} region-pairs per fingerprint")


# %% CELL 2 — Download + extract the HCP data (skips this if ./data already has it)
# ---------------------------------------------------------------------------
def ensure(name):
    """Return path to ./data/<name>, downloading + extracting from OSF if missing."""
    folder = os.path.join(DATA_DIR, name)
    if os.path.isdir(folder):
        return folder
    os.makedirs(DATA_DIR, exist_ok=True)
    archive = folder + ".tgz"
    if not os.path.isfile(archive):
        print(f"Downloading {name} from OSF (this can take a few minutes)...")
        urllib.request.urlretrieve(OSF_URLS[name], archive)
    print(f"Extracting {name}...")
    with tarfile.open(archive) as t:
        t.extractall(DATA_DIR)
    return folder

REST_DIR = ensure("hcp_rest")
TASK_DIR = ensure("hcp_task")
print("Rest data folder:", REST_DIR)
print("Task data folder:", TASK_DIR)


# %% CELL 3 — Brain-network label for each of the 360 regions
# ---------------------------------------------------------------------------
# regions.npy column 1 = which large-scale network (e.g. "Visual", "Default")
# each of the 360 Glasser parcels belongs to. Look at NETWORKS in the
# Variable Explorer — it's a 360-length array of network name strings.
NETWORKS = np.array([str(x) for x in np.load(f"{TASK_DIR}/regions.npy", allow_pickle=True)[:, 1]])
print("Networks found:", sorted(set(NETWORKS)))


# %% CELL 4 — Loader helpers (small, reused for every subject/task/run)
# ---------------------------------------------------------------------------
def load_rest_run(subject, run):
    """Load one resting-state scan (360 regions x time) for one subject."""
    return np.load(f"{REST_DIR}/subjects/{subject}/timeseries/bold{run+1}_Atlas_MSMAll_Glasser360Cortical.npy")

def load_task_run(subject, task, run):
    """Load one task scan. Each task has two runs; bold-file numbers are 5+2*task+run."""
    k = 5 + 2 * task + run
    return np.load(f"{TASK_DIR}/subjects/{subject}/timeseries/bold{k}_Atlas_MSMAll_Glasser360Cortical.npy")

def rest_day(subject, day):
    """Glue a day's two rest runs into one long recording (~2400 timepoints)."""
    runs = [0, 1] if day == 0 else [2, 3]
    return np.concatenate([load_rest_run(subject, r) for r in runs], axis=1)


# %% CELL 5 — The core method (four small building-block functions)
# ---------------------------------------------------------------------------
# These four are kept as functions on purpose: each one gets called dozens
# or hundreds of times (once per subject, once per task, etc.), so writing
# them out inline everywhere would just create duplicated code. Everything
# downstream of here calls these by name.

def fingerprint(scan):
    """(360 x time) recording -> one fingerprint (64,620 numbers).
    Correlation = how much each pair of regions rises and falls together over time."""
    scan = scan - scan.mean(axis=1, keepdims=True)
    grid = np.corrcoef(scan)
    grid = np.arctanh(np.clip(grid, -0.999999, 0.999999))   # Fisher-z rescaling
    return grid[UPPER_TRI_IDX]

def build(load_fn):
    """Stack everyone's fingerprints into one table (people x 64,620)."""
    return np.vstack([fingerprint(load_fn(s)) for s in subject_ids])

def remove_backbone(F):
    """Subtract the group-average fingerprint (the 'generic brain'), leaving
    each person's personal deviation. Unsupervised, per-session, leak-free."""
    return F - F.mean(axis=0, keepdims=True)

def similarity(A, B):
    """339x339 grid: correlation between every A-fingerprint and every B-fingerprint."""
    Az = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    Bz = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    return Az @ Bz.T / A.shape[1]

def accuracy(S):
    """Fraction of people correctly identified (best-match guess), averaged both directions."""
    t = np.arange(S.shape[0])
    return ((S.argmax(1) == t).mean() + (S.argmax(0) == t).mean()) / 2


# %% CELL 6 — CORE experiment, step 1: the naive attempt (one short scan per session)
# ---------------------------------------------------------------------------
rest_run0 = build(lambda s: load_rest_run(s, 0))   # everyone's first short rest scan
rest_run2 = build(lambda s: load_rest_run(s, 2))   # everyone's third short rest scan (a different day)
S_naive = similarity(rest_run0, rest_run2)
acc_naive = accuracy(S_naive)
chance_level = 1 / N_SUBJECTS

print(f"chance level          : {chance_level:.3%}")
print(f"naive (single scan)   : {acc_naive:.1%}")


# %% CELL 7 — CORE experiment, step 2: Fix 1 & 2 — use a full day of data
# ---------------------------------------------------------------------------
day1 = build(lambda s: rest_day(s, 0))   # both rest runs of day 1, concatenated
day2 = build(lambda s: rest_day(s, 1))   # both rest runs of day 2, concatenated
S_fullday = similarity(day1, day2)
acc_fullday = accuracy(S_fullday)

print(f"+ Fix 1 & 2 (full day) : {acc_fullday:.1%}")


# %% CELL 8 — CORE experiment, step 3: Fix 3 — remove the "generic brain" backbone
# ---------------------------------------------------------------------------
day1_personal = remove_backbone(day1)
day2_personal = remove_backbone(day2)
S_final = similarity(day1_personal, day2_personal)
acc_final = accuracy(S_final)

print(f"+ Fix 3 (backbone)     : {acc_final:.1%}")

# the similarity matrix is the one heavy artifact the figures need later
np.save(f"{RESULTS_DIR}/similarity_matrix.npy", S_final.astype(np.float32))


# %% CELL 9 — CORE experiment: summary table
# ---------------------------------------------------------------------------
core_results = {"chance": chance_level, "naive": acc_naive,
                 "fix12": acc_fullday, "fix123": acc_final}
core_results_df = pd.DataFrame(core_results.items(), columns=["stage", "accuracy"])
print(core_results_df)


# %% CELL 10 — EXTENSION 1 (H3): identifiability by brain network
# ---------------------------------------------------------------------------
network_of_region_a = NETWORKS[UPPER_TRI_IDX[0]]
network_of_region_b = NETWORKS[UPPER_TRI_IDX[1]]

network_scores = {}
network_edge_counts = {}
for net in sorted(set(NETWORKS)):
    inside_network = (network_of_region_a == net) & (network_of_region_b == net)
    if inside_network.sum() < 50:      # skip networks with too few within-network edges
        continue
    network_scores[net] = accuracy(similarity(day1_personal[:, inside_network],
                                               day2_personal[:, inside_network]))
    network_edge_counts[net] = int(inside_network.sum())

per_network_df = pd.DataFrame({
    "network": list(network_scores.keys()),
    "accuracy": list(network_scores.values()),
    "n_edges": [network_edge_counts[n] for n in network_scores]
}).sort_values("accuracy", ascending=False).reset_index(drop=True)

print(per_network_df)
per_network_df.to_csv(f"{RESULTS_DIR}/per_network.csv", index=False)


# %% CELL 11 — EXTENSION 3: load behavioural data for the intelligence composite
# ---------------------------------------------------------------------------
# There is no IQ field in this anonymised release, so we build a general-
# cognitive composite from three tasks that load heavily on fluid
# intelligence in the HCP literature: working-memory 2-back accuracy,
# relational-reasoning accuracy, and the adaptive language-math difficulty
# reached.
behavior_dir = os.path.join(DATA_DIR, "hcp", "behavior")
wm_df = pd.read_csv(f"{behavior_dir}/wm.csv")
relational_df = pd.read_csv(f"{behavior_dir}/relational.csv")
language_df = pd.read_csv(f"{behavior_dir}/language.csv")


# %% CELL 12 — EXTENSION 3: per-subject subtest scores
# ---------------------------------------------------------------------------
def by_subject(series):
    return series.reindex(range(N_SUBJECTS)).to_numpy(dtype=float)

two_back_acc = by_subject(wm_df[wm_df.ConditionName.str.startswith("2BK")].groupby("Subject").ACC.mean())
relational_acc = by_subject(relational_df[relational_df.ConditionName == "REL"].groupby("Subject").ACC.mean())
math_difficulty = by_subject(language_df[language_df.ConditionName == "MATH"].groupby("Subject").AVG_DIFFICULTY_LEVEL.mean())

subtests = {"working_memory_2back": two_back_acc,
            "relational_reasoning": relational_acc,
            "language_math_difficulty": math_difficulty}
subtests_df = pd.DataFrame(subtests)
print(subtests_df.describe())


# %% CELL 13 — EXTENSION 3: build the standardised composite score
# ---------------------------------------------------------------------------
def z(x):
    return (x - np.nanmean(x)) / np.nanstd(x)

intelligence_composite = np.nanmean(
    np.vstack([z(two_back_acc), z(relational_acc), z(math_difficulty)]), axis=0)


# %% CELL 14 — EXTENSION 3: how identifiable is each person, from S_final?
# ---------------------------------------------------------------------------
N = S_final.shape[0]
off_diagonal = ~np.eye(N, dtype=bool)

within_match = np.diag(S_final)   # each person's similarity to themselves (day1 vs day2)

# how far above the "impostor" crowd each person's true match sits, in SDs
row_impostor_mean = np.array([S_final[i, off_diagonal[i]].mean() for i in range(N)])
row_impostor_std = np.array([S_final[i, off_diagonal[i]].std() for i in range(N)])
col_impostor_mean = np.array([S_final[off_diagonal[:, j], j].mean() for j in range(N)])
col_impostor_std = np.array([S_final[off_diagonal[:, j], j].std() for j in range(N)])

self_identifiability = 0.5 * (
    (within_match - row_impostor_mean) / row_impostor_std +
    (within_match - col_impostor_mean) / col_impostor_std
)

# were they the correct top-match, in both directions?
subj_index = np.arange(N)
correct_rate = 0.5 * ((S_final.argmax(1) == subj_index).astype(float) +
                       (S_final.argmax(0) == subj_index).astype(float))
identified_correctly = correct_rate == 1.0   # correct in both directions

identifiability_df = pd.DataFrame({
    "subject": subject_ids,
    "self_identifiability": self_identifiability,
    "identified_correctly": identified_correctly
})
print(identifiability_df.head())


# %% CELL 15 — EXTENSION 3: correlate identifiability with intelligence
# ---------------------------------------------------------------------------
valid = np.isfinite(intelligence_composite)

pearson_r, pearson_p = stats.pearsonr(intelligence_composite[valid], self_identifiability[valid])
spearman_rho, spearman_p = stats.spearmanr(intelligence_composite[valid], self_identifiability[valid])
pointbiserial_r, pointbiserial_p = stats.pointbiserialr(
    identified_correctly[valid].astype(float), intelligence_composite[valid])

iq_identified = intelligence_composite[valid & identified_correctly]
iq_missed = intelligence_composite[valid & ~identified_correctly]

print(f"self-identifiability vs IQ : Pearson r = {pearson_r:+.3f} (p={pearson_p:.3g}) | "
      f"Spearman rho = {spearman_rho:+.3f} (p={spearman_p:.3g})")
print(f"identified-correctly vs IQ : point-biserial r = {pointbiserial_r:+.3f} (p={pointbiserial_p:.3g})")
print(f"mean IQ  identified={iq_identified.mean():+.3f} (n={iq_identified.size})  "
      f"missed={iq_missed.mean() if iq_missed.size else float('nan'):+.3f} (n={iq_missed.size})")

subtest_correlations = {}
for name, scores in subtests.items():
    v = valid & np.isfinite(scores)
    r, p = stats.pearsonr(scores[v], self_identifiability[v])
    subtest_correlations[name] = {"r": float(r), "p": float(p)}
subtest_correlations_df = pd.DataFrame(subtest_correlations).T
print(subtest_correlations_df)


# %% CELL 16 — EXTENSION 3: save results
# ---------------------------------------------------------------------------
intelligence_full_df = pd.DataFrame({
    "subject": subject_ids,
    "intelligence_composite": intelligence_composite,
    "self_identifiability": self_identifiability,
    "identified_correctly": identified_correctly.astype(int)
})
intelligence_full_df.to_csv(f"{RESULTS_DIR}/intelligence.csv", index=False)

intelligence_results = {
    "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
    "spearman_rho": float(spearman_rho), "spearman_p": float(spearman_p),
    "pointbiserial_r": float(pointbiserial_r), "pointbiserial_p": float(pointbiserial_p),
    "n": int(valid.sum()), "n_identified": int(identified_correctly[valid].sum()),
    "n_missed": int((~identified_correctly[valid]).sum()),
    "mean_iq_identified": float(iq_identified.mean()),
    "mean_iq_missed": float(iq_missed.mean()) if iq_missed.size else None,
    "subtests": subtest_correlations
}


# %% CELL 17 — EXTENSION 2 (H2): build backbone-removed fingerprints for every task
# ---------------------------------------------------------------------------
# This is the slow cell — 14 builds of 339 fingerprints each (one per task per run).
runA_by_task = {}
runB_by_task = {}
for t, task_name in enumerate(TASKS):
    print(f"Building fingerprints for {task_name}...")
    runA_by_task[task_name] = remove_backbone(build(lambda s, t=t: load_task_run(s, t, 0))).astype(np.float32)
    runB_by_task[task_name] = remove_backbone(build(lambda s, t=t: load_task_run(s, t, 1))).astype(np.float32)


# %% CELL 18 — EXTENSION 2: the 7x7 cross-task identification grid
# ---------------------------------------------------------------------------
cross_task_grid = np.array([
    [accuracy(similarity(runA_by_task[ti], runB_by_task[tj])) for tj in TASKS]
    for ti in TASKS
])
cross_task_grid_df = pd.DataFrame(cross_task_grid, index=TASKS, columns=TASKS)
print(cross_task_grid_df.round(3))

same_task_accuracy = float(np.mean(np.diag(cross_task_grid)))
diff_task_accuracy = float(cross_task_grid[~np.eye(7, dtype=bool)].mean())
print(f"same task (diagonal)     : {same_task_accuracy:.1%}")
print(f"different task (off-diag): {diff_task_accuracy:.1%}  vs chance {1/N_SUBJECTS:.2%}")

# saved without the row-name index, to match the original file format exactly
cross_task_grid_df.to_csv(f"{RESULTS_DIR}/cross_task_grid.csv", index=False)


# %% CELL 19 — EXTENSION 2: specialisation — is within-task accuracy just "more data"?
# ---------------------------------------------------------------------------
within_task_accuracy = np.diag(cross_task_grid)
scan_lengths = np.array([load_task_run(0, t, 0).shape[1] for t in range(7)])  # timepoints per task

length_vs_accuracy_r = float(np.corrcoef(scan_lengths, within_task_accuracy)[0, 1])
print(f"within-task accuracy vs scan length: correlation r = {length_vs_accuracy_r:.2f}")
print("(weak -> accuracy is NOT just 'more data'; it tracks the type of processing)")

specialisation_df = pd.DataFrame({
    "task": TASKS,
    "accuracy": within_task_accuracy,
    "timepoints": scan_lengths
}).sort_values("accuracy", ascending=False).reset_index(drop=True)
print(specialisation_df)

cross_task_results = {
    "grid": cross_task_grid.tolist(), "same_task": same_task_accuracy, "cross_task": diff_task_accuracy,
    "diag": {TASKS[t]: float(within_task_accuracy[t]) for t in range(7)},
    "lengths": {TASKS[t]: int(scan_lengths[t]) for t in range(7)},
    "accuracy_vs_length_r": length_vs_accuracy_r
}


# %% CELL 20 — Save everything into one results.json (mirrors the original output)
# ---------------------------------------------------------------------------
all_results = {
    "core": core_results,
    "networks": {k: float(v) for k, v in network_scores.items()},
    "intelligence": intelligence_results,
    "cross_task": cross_task_results
}
with open(f"{RESULTS_DIR}/results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nDone. Raw results written to ./{RESULTS_DIR}/")
print("  results.json, per_network.csv, intelligence.csv, cross_task_grid.csv, similarity_matrix.npy")
print("Next: run visualiser.py to render the figures.")
