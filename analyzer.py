#!/usr/bin/env python3
"""
============================================================================
 CONNECTOME FINGERPRINTING - ANALYZER (computation only)
============================================================================
Run: `python analyzer.py`
Prerequisites: `python download.py` must have been ran to download HCP database

This will run the experiment and return raw result files with no visualisation.
To visualise results, please see `visualiser.py`.

What it does, start to finish:
  - Brain Fingerprinting
    - Identify people based on their fmri data across multiple dats (91.6% Accuracy)
  - Which brain network is the most identifiable?
    - Runs tests on different higher-order networks to check identifiability of different brain regions
  - Cross task identification
    - Compute identifiability using data from one task to identify a person on another task
  - Intelligience Identifiability Correlation
    - Check is a persons calculated intelligience correlates with identifiability of that person


Outputs (into ./results):
  results.json          all scalar results + metadata for every analysis
  per_network.csv       identification accuracy per brain network
  intelligence.csv      per-subject intelligence composite + identifiability
  cross_task_grid.csv   7x7 cross-task identification accuracy grid
  similarity_matrix.npy 339x339 day1-vs-day2 similarity matrix (backbone removed)

See README.md for the full plain-English explanation of every step.
============================================================================
"""

# Importing External Libraries
import os, json
from enum import IntEnum
from collections.abc import Callable
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants and Configuration
# ---------------------------------------------------------------------------
DATA = "data"
OUT  = "results"
os.makedirs(OUT, exist_ok=True)

N_SUBJECTS = 339  # Number of subjects in the HCP dataset
N_REGIONS  = 360  # Dataset summarises brain into N_REGIONS regions
class Task(IntEnum):
    # Members act as ints (0..6) in arithmetic, but read as names at call sites,
    MOTOR      = 0
    WM         = 1
    EMOTION    = 2
    GAMBLING   = 3
    LANGUAGE   = 4
    RELATIONAL = 5
    SOCIAL     = 6
TASKS = [t.name for t in Task]   # ["MOTOR", "WM", ...] for headers/labels/indexing

IU = np.triu_indices(N_REGIONS, k=1)   # Indicies of the 64,620 unique region-pairs, matrix[IU] can be used to extract only the useful fingerprints
people = list(range(N_SUBJECTS))


# ---------------------------------------------------------------------------
# Data access  (the dataset is fetched separately by download.py)
# ---------------------------------------------------------------------------
def require(name: str) -> str:
    """Return the path to ./data/<name>, or exit with a hint if it isn't present.

    Args:
        name: Sub-directory name to look for inside the data folder.

    Returns:
        The full path to the data directory.

    Raises:
        SystemExit: If the directory doesn't exist yet.
    """
    d = os.path.join(DATA, name)
    if not os.path.isdir(d):
        raise SystemExit(f"Missing {d}. Run `python download.py` first to fetch the HCP dataset.")
    return d


REST = require("hcp_rest")
TASK = require("hcp_task")

regions = np.load(f"{TASK}/regions.npy", allow_pickle=True)
network_labels = regions[:, 1]                      
NETWORKS = network_labels.astype(str) # NETWORKS[I] == Network the `i`th region belongs to


def load_rest_run(subject: int, run: int) -> np.ndarray:
    """Load one resting-state fMRI run for a subject.

    Args:
        subject: Subject ID, selecting the folder under REST/subjects/.
        run: Which rest run, 0-indexed (0-3). Mapped to the 1-indexed
            bold file on disk via `run + 1`.

    Returns:
        A (N_REGIONS, timepoints) float array - the BOLD signal for each of
        the 360 regions across the run's timepoints (typically 1200).
    """
    return np.load(f"{REST}/subjects/{subject}/timeseries/bold{run+1}_Atlas_MSMAll_Glasser360Cortical.npy")


def load_task_run(subject: int, task: Task, run: int) -> np.ndarray:
    """Load one task fMRI run for a subject.

    Args:
        subject: Subject ID, selecting the folder under TASK/subjects/.
        task: Which task to load (a Task enum member, e.g. Task.MOTOR).
        run: Which of the task's two runs, 0-indexed (0 or 1). Each task
            occupies two consecutive bold files, so the file number on disk
            is `5 + 2 * task + run` (bold5 through bold18).

    Returns:
        A (N_REGIONS, timepoints) float array - the BOLD signal for each of
        the 360 regions across the run's timepoints.
    """
    k = 5 + 2 * task + run
    return np.load(f"{TASK}/subjects/{subject}/timeseries/bold{k}_Atlas_MSMAll_Glasser360Cortical.npy")


def rest_day(subject: int, day: int) -> np.ndarray:
    """Combine a day's two rest runs into one long recording.

    Args:
        subject: Subject ID, passed through to load_rest_run.
        day: Which scanning day, 0-indexed. Day 0 uses rest runs 0 and 1,
            day 1 uses runs 2 and 3.

    Returns:
        A (N_REGIONS, timepoints) float array - the day's two runs
        concatenated along the time axis (~2400 timepoints).
    """
    runs = [0, 1] if day == 0 else [2, 3]
    return np.concatenate([load_rest_run(subject, r) for r in runs], axis=1)


def load_intelligence() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a per-subject cognitive ('intelligence') score from in-scanner performance.

    There is no IQ field in this anonymised release, so we build a general-cognitive
    composite from three tasks that load heavily on fluid intelligence in the HCP
    literature: working-memory 2-back accuracy, relational-reasoning accuracy, and
    the adaptive language-math difficulty reached. Each subtest is reduced to one
    number per subject, z-scored, and the three are averaged into a standardised
    composite (mean 0; higher = better cognitive performance).

    Returns:
        A 2-tuple of:
          - composite: a (length - N_SUBJECTS) array, the averaged cognitive score
            (NaN where a subject is missing data),
          - subtests: a dict mapping subtest name -> its own (length - N_SUBJECTS) array.
    """
    behavior_dir = os.path.join(DATA, "hcp", "behavior")

    def align_by_subject(per_subject_score: "pd.Series") -> np.ndarray:
        # Force one slot per subject 0..N-1 (NaN where missing) so subtests stay aligned.
        return per_subject_score.reindex(range(N_SUBJECTS)).to_numpy(dtype=float)

    def zscore(values: np.ndarray) -> np.ndarray:
        # Standardise to SD units, ignoring missing subjects.
        return (values - np.nanmean(values)) / np.nanstd(values)

    wm_data         = pd.read_csv(f"{behavior_dir}/wm.csv")
    relational_data = pd.read_csv(f"{behavior_dir}/relational.csv")
    language_data   = pd.read_csv(f"{behavior_dir}/language.csv")

    # Working memory: mean accuracy on the hard 2-back conditions (ignore 0-back baseline).
    two_back_rows     = wm_data[wm_data.ConditionName.str.startswith("2BK")]
    two_back_accuracy = align_by_subject(two_back_rows.groupby("Subject").ACC.mean())

    # Relational reasoning: mean accuracy on the REL condition (ignore the easier MATCH).
    relational_rows     = relational_data[relational_data.ConditionName == "REL"]
    relational_accuracy = align_by_subject(relational_rows.groupby("Subject").ACC.mean())

    # Language: mean adaptive difficulty reached on the MATH condition (harder = better).
    math_rows       = language_data[language_data.ConditionName == "MATH"]
    math_difficulty = align_by_subject(math_rows.groupby("Subject").AVG_DIFFICULTY_LEVEL.mean())

    subtests = {"working_memory_2back": two_back_accuracy,
                "relational_reasoning": relational_accuracy,
                "language_math_difficulty": math_difficulty}

    # Put the three subtests on a common scale, then average across them per subject.
    standardised = np.vstack([zscore(two_back_accuracy),
                              zscore(relational_accuracy),
                              zscore(math_difficulty)])
    composite = np.nanmean(standardised, axis=0)
    return composite, subtests


# ---------------------------------------------------------------------------
# The method (four small functions)
# ---------------------------------------------------------------------------
def fingerprint(timeseries: np.ndarray) -> np.ndarray:
    """Turn one (regions x time) recording into that session's connectome fingerprint.

    The fingerprint is the brain's correlation structure: for every pair of the 360
    regions, how strongly their activity rises and falls together across the scan.
    That grid is symmetric, so only its upper triangle is kept (see IU), giving a
    flat signature that can be compared against other sessions and people.

    Args:
        timeseries: A (N_REGIONS, timepoints) array of BOLD signal, e.g. from
            load_rest_run / load_task_run.

    Returns:
        A 1-D array of 64,620 Fisher-z-transformed correlations - the unique
        region-pair values from the upper triangle. One person, one session.
    """
    centered = timeseries - timeseries.mean(axis=1, keepdims=True)      # per-region: remove baseline level
    connectome = np.corrcoef(centered)                                  # 360x360 region-by-region correlations
    connectome = np.arctanh(np.clip(connectome, -0.999999, 0.999999))   # Fisher-z rescale (clip keeps arctanh finite)
    return connectome[IU]


def build(load_scan: Callable[[int], np.ndarray]) -> np.ndarray:
    """Stack every subject's fingerprint into one table for a session.

    Args:
        load_scan: A function mapping a subject ID to their (N_REGIONS, timepoints)
            recording, e.g. `lambda s: load_rest_run(s, 0)`. Passing it in is what
            lets the same builder serve rest runs, task runs, or whole days.

    Returns:
        A (N_SUBJECTS, 64,620) array - one row per subject (ordered as in `people`),
        each row the fingerprint from fingerprint().
    """
    return np.vstack([fingerprint(load_scan(subject)) for subject in people])


def remove_backbone(fingerprints: np.ndarray) -> np.ndarray:
    """Subtract the group-average fingerprint (the 'generic brain') from each subject.

    Most of any fingerprint is structure shared by all human brains - nearly identical
    across people and so useless for telling them apart. Removing the per-session
    average leaves only each subject's personal deviation, which sharpens
    identification. It uses no labels and is recomputed per session, so it is
    unsupervised and leak-free.

    Args:
        fingerprints: A (N_SUBJECTS, 64,620) table for one session, as built by build().

    Returns:
        A (N_SUBJECTS, 64,620) array of the same shape, each row now the subject's
        deviation from the group mean (every column mean-centred across subjects).
    """
    return fingerprints - fingerprints.mean(axis=0, keepdims=True)


def similarity(fingerprints_a: np.ndarray, fingerprints_b: np.ndarray) -> np.ndarray:
    """Correlate every fingerprint in one session against every fingerprint in another.

    Typically A is day 1 and B is day 2. Each fingerprint is z-scored first, so a dot
    product between two rows is their Pearson correlation. Identification succeeds when
    a subject's own across-session entry (on the diagonal) is their strongest match.

    Args:
        fingerprints_a: A (N_SUBJECTS, 64,620) table for session A.
        fingerprints_b: A (N_SUBJECTS, 64,620) table for session B (same subject order).

    Returns:
        A (N_SUBJECTS, N_SUBJECTS) correlation grid where entry [i, j] is the
        similarity between subject i in session A and subject j in session B.
    """
    a_z = (fingerprints_a - fingerprints_a.mean(1, keepdims=True)) / fingerprints_a.std(1, keepdims=True)
    b_z = (fingerprints_b - fingerprints_b.mean(1, keepdims=True)) / fingerprints_b.std(1, keepdims=True)
    return a_z @ b_z.T / fingerprints_a.shape[1]


def accuracy(similarity_matrix: np.ndarray) -> float:
    """Fraction of subjects correctly re-identified by best match, averaged both ways.

    Each subject's guess is the row/column with the highest similarity to them; the
    guess is correct when that best match is the same subject (i.e. on the diagonal).
    The two directions (A->B and B->A) are averaged so the score is symmetric.

    Args:
        similarity_matrix: A square (N_SUBJECTS, N_SUBJECTS) grid from similarity(),
            with matching subject order on both axes.

    Returns:
        A single accuracy in [0, 1] - the mean fraction of subjects whose strongest
        match is themselves (1 / N_SUBJECTS would be chance).
    """
    truth  = np.arange(similarity_matrix.shape[0])
    a_to_b = (similarity_matrix.argmax(1) == truth).mean()   # for each session-A subject, is their top B-match themselves?
    b_to_a = (similarity_matrix.argmax(0) == truth).mean()   # same, the other direction
    return float((a_to_b + b_to_a) / 2)


# ===========================================================================
# CORE EXPERIMENT - Naive, Fix 1 + 2, Fix 1 + 2 + 3
# ===========================================================================
def run_core() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the core fingerprinting result, from a naive baseline to the full method.

    Runs identification three ways on resting-state data and prints each accuracy:
      * naive: a single short scan per session (weak baseline)
      * Fix 1 & 2: a whole day of rest (both runs concatenated, phase-encoding balanced)
      * Fix 3: additionally remove the group-average 'backbone' (remove_backbone)
    Also saves the day-1-vs-day-2 similarity matrix, the one heavy artifact the figures need.

    Returns:
        A 4-tuple of:
          * a dict of the four scalar accuracies (chance, naive, fix12, fix123),
          * day1_deviations, day2_deviations: the (N_SUBJECTS, 64,620) backbone-removed
            fingerprint tables for each day,
          * similarity_matrix: the (N_SUBJECTS, N_SUBJECTS) day-1-vs-day-2 grid.
        The latter three are handed on to the extension analyses.
    """
    print("\n=== CORE: from a naive attempt to a strong result ===")
    chance = 1 / N_SUBJECTS

    # Naive: a single short scan per session
    naive_accuracy = accuracy(similarity(build(lambda s: load_rest_run(s, 0)),
                                         build(lambda s: load_rest_run(s, 2))))

    # Fix 1 & 2: a full day of data (both runs glued, phase-encoding balanced)
    day1_fingerprints = build(lambda s: rest_day(s, 0))
    day2_fingerprints = build(lambda s: rest_day(s, 1))
    fullday_accuracy = accuracy(similarity(day1_fingerprints, day2_fingerprints))

    # Fix 3: also remove the generic brain
    day1_deviations = remove_backbone(day1_fingerprints)
    day2_deviations = remove_backbone(day2_fingerprints)
    similarity_matrix = similarity(day1_deviations, day2_deviations)
    backbone_accuracy = accuracy(similarity_matrix)

    print(f"chance                 : {chance:.3%}")
    print(f"naive (single scan)    : {naive_accuracy:.1%}")
    print(f"+ Fix 1 & 2 (full day) : {fullday_accuracy:.1%}")
    print(f"+ Fix 3 (backbone)     : {backbone_accuracy:.1%}")

    # the similarity matrix is the one heavy artifact the figures need
    np.save(f"{OUT}/similarity_matrix.npy", similarity_matrix.astype(np.float32))

    return ({"chance": chance, "naive": naive_accuracy, "fix12": fullday_accuracy, "fix123": backbone_accuracy},
            day1_deviations, day2_deviations, similarity_matrix)


# ===========================================================================
# EXTENSION 1 (H3) - which networks fingerprint best
# ===========================================================================
def run_networks(day1_deviations: np.ndarray, day2_deviations: np.ndarray) -> dict:
    """Measure identification accuracy using only the edges within each brain network.

    Every fingerprint edge links two regions; an edge sits 'inside' a network when both
    of its endpoint regions belong to that network. Restricting the fingerprint to a
    network's internal edges and re-running identification reveals which networks carry
    the most identifying information. Networks with fewer than 50 internal edges are
    skipped as too small to be reliable. Writes per_network.csv.

    Args:
        day1_deviations: (N_SUBJECTS, 64,620) backbone-removed fingerprints, day 1.
        day2_deviations: the same for day 2.

    Returns:
        A dict mapping network name -> identification accuracy, for the networks that
        cleared the 50-edge minimum.
    """
    print("\n=== EXTENSION 1 (H3): identifiability by brain network ===")

    # Label each of the 64,620 fingerprint edges by the network of its two endpoint regions.
    # IU[0]/IU[1] hold the two region indices per edge, in the same order as the fingerprint.
    endpoint_a_network = NETWORKS[IU[0]]
    endpoint_b_network = NETWORKS[IU[1]]

    # For each network, keep only its internal edges and re-run the full identification.
    network_accuracy, network_edge_count = {}, {}
    for network in sorted(set(NETWORKS)):
        within_network = (endpoint_a_network == network) & (endpoint_b_network == network)
        if within_network.sum() < 50:          # too few edges to give a stable estimate
            continue
        network_accuracy[network] = accuracy(similarity(day1_deviations[:, within_network],
                                                        day2_deviations[:, within_network]))
        network_edge_count[network] = int(within_network.sum())

    # Report and persist, most identifiable network first.
    ranked = sorted(network_accuracy.items(), key=lambda kv: -kv[1])
    for network, acc in ranked:
        print(f"  {network:14s} {acc:5.1%}  ({network_edge_count[network]} connections)")

    with open(f"{OUT}/per_network.csv", "w") as csv_file:
        csv_file.write("network,accuracy,n_edges\n")
        for network, acc in ranked:
            csv_file.write(f"{network},{acc:.4f},{network_edge_count[network]}\n")

    return network_accuracy


# ===========================================================================
# EXTENSION 3 - does a person's intelligence affect how identifiable they are?
# ===========================================================================
def compute_self_identifiability(similarity_matrix: np.ndarray) -> np.ndarray:
    """Per-subject distinctiveness: how far each subject's own match sits above impostors.

    For subject i the 'self match' is the diagonal entry (their day-1 vs day-2 similarity).
    The impostors are the off-diagonal entries in that subject's row (querying day 1 against
    everyone's day 2) and column (the reverse). We z-score the self match against the
    impostor cloud in each direction and average the two, giving a graded score in SD units
    (higher = more uniquely identifiable) rather than a plain yes/no.

    Args:
        similarity_matrix: the (N_SUBJECTS, N_SUBJECTS) cross-day grid from run_core.

    Returns:
        A length-N_SUBJECTS array of distinctiveness scores, in standard deviations.
    """
    off_diagonal = ~np.eye(N_SUBJECTS, dtype=bool)    # True everywhere except the self-matches
    self_match = np.diag(similarity_matrix)           # each subject's own across-day similarity

    # Impostor cloud stats, treating each subject first as a day-1 query (their row)...
    row_impostor_mean = np.array([similarity_matrix[i, off_diagonal[i]].mean() for i in range(N_SUBJECTS)])
    row_impostor_std  = np.array([similarity_matrix[i, off_diagonal[i]].std()  for i in range(N_SUBJECTS)])
    # ...then as a day-2 query (their column).
    col_impostor_mean = np.array([similarity_matrix[off_diagonal[:, j], j].mean() for j in range(N_SUBJECTS)])
    col_impostor_std  = np.array([similarity_matrix[off_diagonal[:, j], j].std()  for j in range(N_SUBJECTS)])

    # Self match's distance above the impostors, in SDs, averaged over both directions.
    return 0.5 * ((self_match - row_impostor_mean) / row_impostor_std
                  + (self_match - col_impostor_mean) / col_impostor_std)


def compute_identified(similarity_matrix: np.ndarray) -> np.ndarray:
    """Per-subject boolean: was this subject the top match in BOTH directions?

    Args:
        similarity_matrix: the (N_SUBJECTS, N_SUBJECTS) cross-day grid from run_core.

    Returns:
        A length-N_SUBJECTS boolean array, True where the subject is their own best match
        both when querying day 1 -> day 2 and day 2 -> day 1.
    """
    truth = np.arange(N_SUBJECTS)
    row_correct = similarity_matrix.argmax(1) == truth   # best day-2 match for each day-1 subject
    col_correct = similarity_matrix.argmax(0) == truth   # best day-1 match for each day-2 subject
    return row_correct & col_correct


def run_intelligence(similarity_matrix: np.ndarray) -> dict:
    """Test whether more (cognitively) able people are easier to fingerprint.

    Per subject we measure identifiability from the day-1 vs day-2 rest similarity
    matrix in two ways:
      * self-identifiability (continuous): how far a subject's own across-day match
        stands above the crowd of impostor matches, in SDs -> a graded 'distinctiveness'
      * identified correctly (binary): were they the top match in both directions
    Each is then correlated against the cognitive composite (Pearson + Spearman for the
    continuous score, point-biserial for the binary one), plus a per-subtest Pearson.
    Writes intelligence.csv.

    Args:
        similarity_matrix: the (N_SUBJECTS, N_SUBJECTS) day-1-vs-day-2 grid from run_core.

    Returns:
        A dict of the correlation statistics, sample sizes, group mean IQs, and the
        per-subtest correlations.
    """
    print("\n=== EXTENSION 3: identifiability vs intelligence ===")
    intelligence, subtests = load_intelligence()

    # Two identifiability measures per subject: a graded score and a strict yes/no.
    self_identifiability = compute_self_identifiability(similarity_matrix)
    identified = compute_identified(similarity_matrix)

    # Correlate each measure against the cognitive composite, over subjects that have one.
    # (Pearson + Spearman suit the continuous score; point-biserial suits the binary one.)
    has_intelligence = np.isfinite(intelligence)
    pearson_r, pearson_p = stats.pearsonr(intelligence[has_intelligence], self_identifiability[has_intelligence])
    spearman_rho, spearman_p = stats.spearmanr(intelligence[has_intelligence], self_identifiability[has_intelligence])
    pointbiserial_r, pointbiserial_p = stats.pointbiserialr(identified[has_intelligence].astype(float),
                                                            intelligence[has_intelligence])
    iq_identified = intelligence[has_intelligence & identified]
    iq_missed     = intelligence[has_intelligence & ~identified]

    print(f"  self-identifiability vs IQ : Pearson r = {pearson_r:+.3f} (p = {pearson_p:.3g}) | "
          f"Spearman rho = {spearman_rho:+.3f} (p = {spearman_p:.3g})")
    print(f"  identified-correctly vs IQ : point-biserial r = {pointbiserial_r:+.3f} (p = {pointbiserial_p:.3g})")
    print(f"  mean IQ  identified={iq_identified.mean():+.3f} (n={iq_identified.size})  "
          f"missed={iq_missed.mean() if iq_missed.size else float('nan'):+.3f} (n={iq_missed.size})")

    # Break the continuous relationship down by individual subtest.
    print("  per-subtest Pearson r with self-identifiability:")
    subtest_correlations = {}
    for subtest_name, subtest_scores in subtests.items():
        valid_subtest = has_intelligence & np.isfinite(subtest_scores)
        r, p = stats.pearsonr(subtest_scores[valid_subtest], self_identifiability[valid_subtest])
        subtest_correlations[subtest_name] = {"r": float(r), "p": float(p)}
        print(f"    {subtest_name:26s} r = {r:+.3f} (p = {p:.3g})")

    # One row per subject for downstream plotting.
    with open(f"{OUT}/intelligence.csv", "w") as csv_file:
        csv_file.write("subject,intelligence_composite,self_identifiability,identified_correctly\n")
        for i in range(N_SUBJECTS):
            csv_file.write(f"{i},{intelligence[i]:.4f},{self_identifiability[i]:.4f},{int(identified[i])}\n")

    return {"pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho), "spearman_p": float(spearman_p),
            "pointbiserial_r": float(pointbiserial_r), "pointbiserial_p": float(pointbiserial_p),
            "n": int(has_intelligence.sum()), "n_identified": int(identified[has_intelligence].sum()),
            "n_missed": int((~identified[has_intelligence]).sum()),
            "mean_iq_identified": float(iq_identified.mean()),
            "mean_iq_missed": float(iq_missed.mean()) if iq_missed.size else None,
            "subtests": subtest_correlations}


# ===========================================================================
# EXTENSION 2 (H2) - cross-task identification + SPECIALISATION analysis
# ===========================================================================
def build_task_fingerprints(run: int) -> list[np.ndarray]:
    """Backbone-removed fingerprint table for the given run of every task.

    Args:
        run: which of each task's two runs to load (0 or 1).

    Returns:
        A list of one (N_SUBJECTS, 64,620) float32 table per task, ordered as in Task.
    """
    # t=t binds the current task into each lambda (else every lambda would see the last t).
    return [remove_backbone(build(lambda s, t=t: load_task_run(s, t, run))).astype(np.float32) for t in Task]


def run_cross_task() -> dict:
    """Identify subjects across tasks, and test whether accuracy tracks scan length.

    Builds a backbone-removed fingerprint per subject for each task's two runs, then
    fills a 7x7 grid: entry [i, j] is the accuracy of using task-i run-0 to identify
    subjects in task-j run-1. The diagonal is same-task identification; the off-diagonal
    is cross-task (generalising across mental states). A SPECIALISATION check then
    correlates each task's own-task accuracy against its scan length, to show accuracy
    reflects the type of processing rather than merely how much data was collected.
    Writes cross_task_grid.csv.

    Returns:
        A dict with the full grid, the mean same-task and cross-task accuracies, the
        per-task diagonal accuracies and scan lengths, and the accuracy-vs-length r.
    """
    print("\n=== EXTENSION 2 (H2): cross-task identification ===")

    # Fill the 7x7 grid: identify run-0 of task i among run-1 of task j.
    run0_by_task = build_task_fingerprints(0)
    run1_by_task = build_task_fingerprints(1)
    accuracy_grid = np.array([[accuracy(similarity(run0_by_task[i], run1_by_task[j])) for j in Task] for i in Task])

    # Same task = diagonal; cross task = everything off the diagonal.
    same_task_acc = float(np.mean(np.diag(accuracy_grid)))
    cross_task_acc = float(accuracy_grid[~np.eye(len(Task), dtype=bool)].mean())
    print(f"  same task (diagonal)     : {same_task_acc:.1%}")
    print(f"  different task (off-diag): {cross_task_acc:.1%}  vs chance {1/N_SUBJECTS:.2%}")

    np.savetxt(f"{OUT}/cross_task_grid.csv", accuracy_grid, delimiter=",",
               header=",".join(TASKS), comments="")

    # SPECIALISATION: is a task's own-task accuracy just explained by how long its scan was?
    # A weak correlation argues accuracy reflects the *type* of processing, not sheer data.
    within_task_acc = np.diag(accuracy_grid)
    scan_lengths = np.array([load_task_run(0, t, 0).shape[1] for t in Task])  # timepoints per task
    length_corr = float(np.corrcoef(scan_lengths, within_task_acc)[0, 1])
    print(f"\n  within-task accuracy vs scan length: correlation r = {length_corr:.2f}")
    print("  (weak -> accuracy is NOT just 'more data'; it tracks the type of processing)")
    for task_idx in np.argsort(-within_task_acc):   # best-identified task first
        print(f"    {TASKS[task_idx]:11s} {within_task_acc[task_idx]:5.1%}   ({scan_lengths[task_idx]} timepoints)")

    return {"grid": accuracy_grid.tolist(), "same_task": same_task_acc, "cross_task": cross_task_acc,
            "diag": {TASKS[t]: float(within_task_acc[t]) for t in Task},
            "lengths": {TASKS[t]: int(scan_lengths[t]) for t in Task},
            "accuracy_vs_length_r": length_corr}


# ===========================================================================


if __name__ == "__main__":
    core_results, day1_deviations, day2_deviations, similarity_matrix = run_core()
    network_results = run_networks(day1_deviations, day2_deviations)
    intelligence_results = run_intelligence(similarity_matrix)
    cross_task_results = run_cross_task()

    results = {"core": core_results,
               "networks": {name: float(acc) for name, acc in network_results.items()},
               "intelligence": intelligence_results,
               "cross_task": cross_task_results}
    with open(f"{OUT}/results.json", "w") as results_file:
        json.dump(results, results_file, indent=2)

    print(f"\nDone. Raw results written to ./{OUT}/")
    print("  results.json, per_network.csv, intelligence.csv, cross_task_grid.csv, similarity_matrix.npy")
    print("Next: run  python visualiser.py  to render the figures.")
