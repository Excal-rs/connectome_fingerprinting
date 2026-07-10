#!/usr/bin/env python3
"""
============================================================================
 CONNECTOME FINGERPRINTING — ANALYZER (computation only)
============================================================================
Run:   python analyzer.py

Runs the full experiment and writes RAW RESULT FILES only (no figures).
The companion script `visualiser.py` turns these into the figures, so it can
run on a clone of the repo without re-downloading the 12 GB dataset.

What it does, start to finish:
  * downloads the HCP data if it isn't already in ./data (rest ~1.6 GB, task ~1.4 GB)
  * CORE experiment: brain "fingerprinting" from 31%  ->  92% via three fixes
  * EXTENSION 1 (H3): which brain networks make people most identifiable
  * EXTENSION 2 (H2): does identity survive a change of task (cross-task grid)
  * EXTENSION 3: does a person's intelligence affect how identifiable they are

Outputs (into ./results):
  results.json          all scalar results + metadata for every analysis
  per_network.csv       identification accuracy per brain network
  intelligence.csv      per-subject intelligence composite + identifiability
  cross_task_grid.csv   7x7 cross-task identification accuracy grid
  similarity_matrix.npy 339x339 day1-vs-day2 similarity matrix (backbone removed)

See README.md for the full plain-English explanation of every step.
============================================================================
"""
import os, json, urllib.request, tarfile
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA = "data"
OUT = "results"
os.makedirs(OUT, exist_ok=True)
N_SUBJECTS = 339              # subjects in the Neuromatch HCP set
N_PARCELS = 360               # brain summarised into 360 regions (Glasser atlas)
IU = np.triu_indices(N_PARCELS, k=1)   # the 64,620 unique region-pairs = a fingerprint
people = list(range(N_SUBJECTS))
TASKS = ["MOTOR", "WM", "EMOTION", "GAMBLING", "LANGUAGE", "RELATIONAL", "SOCIAL"]

OSF = {"hcp_rest": "https://osf.io/bqp7m/download",
       "hcp_task": "https://osf.io/s4h8j/download"}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def ensure(name):
    """Return path to ./data/<name>, downloading + extracting from OSF if missing."""
    d = os.path.join(DATA, name)
    if os.path.isdir(d):
        return d
    os.makedirs(DATA, exist_ok=True)
    tgz = d + ".tgz"
    if not os.path.isfile(tgz):
        print(f"Downloading {name} from OSF (this can take a few minutes)...")
        urllib.request.urlretrieve(OSF[name], tgz)
    print(f"Extracting {name}...")
    with tarfile.open(tgz) as t:
        t.extractall(DATA)
    return d


REST = ensure("hcp_rest")
TASK = ensure("hcp_task")
# brain-network label for each of the 360 regions (column 1 of regions.npy)
NETWORKS = np.array([str(x) for x in np.load(f"{TASK}/regions.npy", allow_pickle=True)[:, 1]])


def load_rest_run(subject, run):
    return np.load(f"{REST}/subjects/{subject}/timeseries/bold{run+1}_Atlas_MSMAll_Glasser360Cortical.npy")


def load_task_run(subject, task, run):
    # each task has two runs; their bold-file numbers are 5 + 2*task + run
    k = 5 + 2 * task + run
    return np.load(f"{TASK}/subjects/{subject}/timeseries/bold{k}_Atlas_MSMAll_Glasser360Cortical.npy")


def rest_day(subject, day):
    # combine a day's two rest runs into one long recording (~2400 timepoints)
    runs = [0, 1] if day == 0 else [2, 3]
    return np.concatenate([load_rest_run(subject, r) for r in runs], axis=1)


def load_intelligence():
    """Per-subject cognitive ('intelligence') score from in-scanner task performance.

    There is no IQ field in this anonymised release, so we build a general-cognitive
    composite from three tasks that load heavily on fluid intelligence in the HCP
    literature: working-memory 2-back accuracy, relational-reasoning accuracy, and
    the adaptive language-math difficulty reached. Each is z-scored and averaged, so
    the composite is a standardised score (mean 0, higher = better cognitive
    performance). Returns (composite, subtests-dict), each a length-N_SUBJECTS array.
    """
    import pandas as pd
    B = os.path.join(DATA, "hcp", "behavior")

    def by_subject(series):
        return series.reindex(range(N_SUBJECTS)).to_numpy(dtype=float)

    def z(x):
        return (x - np.nanmean(x)) / np.nanstd(x)

    wm = pd.read_csv(f"{B}/wm.csv")
    rel = pd.read_csv(f"{B}/relational.csv")
    lang = pd.read_csv(f"{B}/language.csv")

    two_back = by_subject(wm[wm.ConditionName.str.startswith("2BK")].groupby("Subject").ACC.mean())
    relational = by_subject(rel[rel.ConditionName == "REL"].groupby("Subject").ACC.mean())
    math_diff = by_subject(lang[lang.ConditionName == "MATH"].groupby("Subject").AVG_DIFFICULTY_LEVEL.mean())

    subtests = {"working_memory_2back": two_back, "relational_reasoning": relational,
                "language_math_difficulty": math_diff}
    composite = np.nanmean(np.vstack([z(two_back), z(relational), z(math_diff)]), axis=0)
    return composite, subtests


# ---------------------------------------------------------------------------
# The method (four small functions)
# ---------------------------------------------------------------------------
def fingerprint(scan):
    """(360 x time) recording -> one fingerprint (64,620 numbers).
    Correlation = how much each pair of regions rises and falls together over time."""
    scan = scan - scan.mean(axis=1, keepdims=True)
    grid = np.corrcoef(scan)
    grid = np.arctanh(np.clip(grid, -0.999999, 0.999999))   # Fisher-z rescaling
    return grid[IU]


def build(load_fn):
    """Stack everyone's fingerprints into one table (people x 64,620)."""
    return np.vstack([fingerprint(load_fn(s)) for s in people])


def remove_backbone(F):
    """Subtract the group-average fingerprint (the 'generic brain'), leaving each
    person's personal deviation. Unsupervised, per-session, leak-free."""
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


# ===========================================================================
# CORE EXPERIMENT — 31% -> 92%
# ===========================================================================
def run_core():
    print("\n=== CORE: from a naive attempt to a strong result ===")
    chance = 1 / N_SUBJECTS

    # Naive: a single short scan per session
    before = accuracy(similarity(build(lambda s: load_rest_run(s, 0)),
                                 build(lambda s: load_rest_run(s, 2))))

    # Fix 1 & 2: a full day of data (both runs glued; phase-encoding balanced)
    day1 = build(lambda s: rest_day(s, 0))
    day2 = build(lambda s: rest_day(s, 1))
    fix12 = accuracy(similarity(day1, day2))

    # Fix 3: also remove the generic brain
    d1, d2 = remove_backbone(day1), remove_backbone(day2)
    S = similarity(d1, d2)
    fix123 = accuracy(S)

    print(f"  chance                 : {chance:.3%}")
    print(f"  naive (single scan)    : {before:.1%}")
    print(f"  + Fix 1 & 2 (full day) : {fix12:.1%}")
    print(f"  + Fix 3 (backbone)     : {fix123:.1%}")

    # the similarity matrix is the one heavy artifact the figures need
    np.save(f"{OUT}/similarity_matrix.npy", S.astype(np.float32))

    return {"chance": chance, "naive": before, "fix12": fix12, "fix123": fix123}, d1, d2, S


# ===========================================================================
# EXTENSION 1 (H3) — which networks fingerprint best
# ===========================================================================
def run_networks(d1, d2):
    print("\n=== EXTENSION 1 (H3): identifiability by brain network ===")
    eA, eB = NETWORKS[IU[0]], NETWORKS[IU[1]]
    scores, edge_counts = {}, {}
    for net in sorted(set(NETWORKS)):
        inside = (eA == net) & (eB == net)
        if inside.sum() < 50:
            continue
        scores[net] = accuracy(similarity(d1[:, inside], d2[:, inside]))
        edge_counts[net] = int(inside.sum())
    for net, a in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {net:14s} {a:5.1%}  ({edge_counts[net]} connections)")

    with open(f"{OUT}/per_network.csv", "w") as f:
        f.write("network,accuracy,n_edges\n")
        for net, a in sorted(scores.items(), key=lambda kv: -kv[1]):
            f.write(f"{net},{a:.4f},{edge_counts[net]}\n")

    return scores


# ===========================================================================
# EXTENSION 3 — does a person's intelligence affect how identifiable they are?
# ===========================================================================
def run_intelligence(S):
    """Test whether more (cognitively) able people are easier to fingerprint.

    Per person we measure identifiability from the day-1 vs day-2 rest similarity
    matrix in two ways:
      * self-identifiability (continuous): how far a person's own across-day match
        stands above the crowd of impostor matches, in SDs -> a graded 'distinctiveness'
      * identified correctly (binary): were they the top match (averaged both directions)
    Then we correlate each against the cognitive composite (Pearson + Spearman for the
    continuous score, point-biserial for the binary one).
    """
    from scipy import stats
    print("\n=== EXTENSION 3: identifiability vs intelligence ===")
    iq, subtests = load_intelligence()

    N = S.shape[0]
    off = ~np.eye(N, dtype=bool)

    # self-identifiability: within-match distance above the impostor cloud, in SDs.
    # Averaged over both directions (day1->day2 query and day2->day1 query).
    within = np.diag(S)
    row_imp_mean = np.array([S[i, off[i]].mean() for i in range(N)])
    row_imp_std = np.array([S[i, off[i]].std() for i in range(N)])
    col_imp_mean = np.array([S[off[:, j], j].mean() for j in range(N)])
    col_imp_std = np.array([S[off[:, j], j].std() for j in range(N)])
    self_id = 0.5 * ((within - row_imp_mean) / row_imp_std + (within - col_imp_mean) / col_imp_std)

    # identified correctly: top match in each direction, averaged -> {0, 0.5, 1}
    t = np.arange(N)
    correct_rate = 0.5 * ((S.argmax(1) == t).astype(float) + (S.argmax(0) == t).astype(float))
    identified = correct_rate == 1.0   # correct in both directions

    valid = np.isfinite(iq)
    rP, pP = stats.pearsonr(iq[valid], self_id[valid])
    rS, pS = stats.spearmanr(iq[valid], self_id[valid])
    rB, pB = stats.pointbiserialr(identified[valid].astype(float), iq[valid])
    hi = iq[valid & identified]; lo = iq[valid & ~identified]
    print(f"  self-identifiability vs IQ : Pearson r = {rP:+.3f} (p = {pP:.3g}) | "
          f"Spearman rho = {rS:+.3f} (p = {pS:.3g})")
    print(f"  identified-correctly vs IQ : point-biserial r = {rB:+.3f} (p = {pB:.3g})")
    print(f"  mean IQ  identified={hi.mean():+.3f} (n={hi.size})  "
          f"missed={lo.mean() if lo.size else float('nan'):+.3f} (n={lo.size})")
    print("  per-subtest Pearson r with self-identifiability:")
    subtest_r = {}
    for name, sc in subtests.items():
        v = valid & np.isfinite(sc)
        r, p = stats.pearsonr(sc[v], self_id[v])
        subtest_r[name] = {"r": float(r), "p": float(p)}
        print(f"    {name:26s} r = {r:+.3f} (p = {p:.3g})")

    with open(f"{OUT}/intelligence.csv", "w") as f:
        f.write("subject,intelligence_composite,self_identifiability,identified_correctly\n")
        for i in range(N):
            f.write(f"{i},{iq[i]:.4f},{self_id[i]:.4f},{int(identified[i])}\n")

    return {"pearson_r": float(rP), "pearson_p": float(pP),
            "spearman_rho": float(rS), "spearman_p": float(pS),
            "pointbiserial_r": float(rB), "pointbiserial_p": float(pB),
            "n": int(valid.sum()), "n_identified": int(identified[valid].sum()),
            "n_missed": int((~identified[valid]).sum()),
            "mean_iq_identified": float(hi.mean()),
            "mean_iq_missed": float(lo.mean()) if lo.size else None,
            "subtests": subtest_r}


# ===========================================================================
# EXTENSION 2 (H2) — cross-task identification + SPECIALISATION analysis
# ===========================================================================
def run_cross_task():
    print("\n=== EXTENSION 2 (H2): cross-task identification ===")
    runA = [remove_backbone(build(lambda s, t=t: load_task_run(s, t, 0))).astype(np.float32) for t in range(7)]
    runB = [remove_backbone(build(lambda s, t=t: load_task_run(s, t, 1))).astype(np.float32) for t in range(7)]
    grid = np.array([[accuracy(similarity(runA[i], runB[j])) for j in range(7)] for i in range(7)])

    same = float(np.mean(np.diag(grid)))
    diff = float(grid[~np.eye(7, dtype=bool)].mean())
    print(f"  same task (diagonal)     : {same:.1%}")
    print(f"  different task (off-diag): {diff:.1%}  vs chance {1/N_SUBJECTS:.2%}")

    np.savetxt(f"{OUT}/cross_task_grid.csv", grid, delimiter=",",
               header=",".join(TASKS), comments="")

    # ---- SPECIALISATION: is within-task accuracy explained by scan length? ----
    diag = np.diag(grid)
    lengths = np.array([load_task_run(0, t, 0).shape[1] for t in range(7)])  # timepoints per task
    r = float(np.corrcoef(lengths, diag)[0, 1])
    print(f"\n  within-task accuracy vs scan length: correlation r = {r:.2f}")
    print("  (weak -> accuracy is NOT just 'more data'; it tracks the type of processing)")
    for t in np.argsort(-diag):
        print(f"    {TASKS[t]:11s} {diag[t]:5.1%}   ({lengths[t]} timepoints)")

    return {"grid": grid.tolist(), "same_task": same, "cross_task": diff,
            "diag": {TASKS[t]: float(diag[t]) for t in range(7)},
            "lengths": {TASKS[t]: int(lengths[t]) for t in range(7)},
            "accuracy_vs_length_r": r}


# ===========================================================================
if __name__ == "__main__":
    core, d1, d2, S = run_core()
    nets = run_networks(d1, d2)
    intel = run_intelligence(S)
    cross = run_cross_task()

    results = {"core": core,
               "networks": {k: float(v) for k, v in nets.items()},
               "intelligence": intel,
               "cross_task": cross}
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Raw results written to ./{OUT}/")
    print("  results.json, per_network.csv, intelligence.csv, cross_task_grid.csv, similarity_matrix.npy")
    print("Next: run  python visualiser.py  to render the figures.")
