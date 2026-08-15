# Connectome Fingerprinting

**Can you identify a person from their brain's wiring pattern?**

We know everyone's brain is unique, but research (*Finn et al. 2015, Nature Neuroscience*) has shown that the way different regions of the brain rise and fall together creates a stable and personal fingerprint. We tested this on **339 people** from the **HCP dataset**.

### Core experiment summary

1. Construct a person's fingerprint on Day 1 from their resting state fMRI data.
2. Construct a fingerprint of everyone in the dataset on another day (Day 2) using the exact same method.
3. Using the fingerprint from Day 1, can we pick out the same person on Day 2?

Guessing at random gives you an analytical chance of **1 in 339 (~0.3%)**. We reached an accuracy of **91.6%**.

We also dug into *which* parts of the brain actually carry identity, how well that identity survives a change of activity, and whether it's easier to identify people who are more intelligent.

---

### Project structure

├── download.py          STAGE 0: fetches the HCP dataset from OSF into data/ (run once)
├── analyzer.py          STAGE 1: loads data, runs every experiment, writes raw results
├── visualiser.py        STAGE 2: reads results/, renders every figure (no dataset access)
├── connectome_fingerprinting.ipynb
│                        the same pipeline, narrated: every step explained as it runs
├── requirements.txt     numpy, scipy, pandas, matplotlib
├── results/             raw result files  (committed, small)
├── figures/             the figures        (committed)
└── data/                the 12 GB HCP dataset  (git-ignored; fetched by download.py)

## How to run it
```
pip install -r requirements.txt

python download.py     # OSF → data/       (fetches the full ~12 GB HCP dataset; skips what's already there)
python analyzer.py     # data → raw results/   (a few minutes of compute)
python visualiser.py   # results/ → figures/   (seconds; no dataset needed)
```

### Or a walkthrough

connectome_fingerprinting.ipynb is the same code told as a story. Every step gets an explanation, followed by the code, and then the exact number or figure it produces, taking you from our naive 31% attempt all the way through to the final extensions. It runs the identical analysis and draws the identical figures inline, but writes nothing to disk so it can't clobber results/ or figures/.

It's **self-contained** and built for [Google Colab](https://colab.research.google.com/): it inlines the downloader and needs no other files from this repo. You can upload the single .ipynb and run it top to bottom. No installations required. Colab already has numpy, scipy, pandas, and matplotlib. *(Note: Colab wipes storage when the runtime disconnects, so the ~12 GB fetch will repeat each session).*

### Input files (what analyzer.py needs in data/)

Fetched from OSF by python download.py, into ./data/:

| Input | Contents |
| --- | --- |
| hcp_rest/ | resting-state scans: 4 per person (two per day), each a 360 regions × time table |
| hcp_task/ | task scans: 7 tasks × 2 runs per person, plus regions.npy (the network label of each region) |
| hcp/behavior/ | in-scanner task performance (used to build the intelligence composite)¹ |

### Output files

**analyzer.py → results/** (raw numbers, no plotting):

| File | Contents |
| --- | --- |
| results.json | every scalar result + metadata for all four analyses |
| per_network.csv | identification accuracy per brain network |
| per_region.csv | identification accuracy + MNI coordinates per brain region (all 360) |
| intelligence.csv | per-subject intelligence composite + identifiability |
| cross_task_grid.csv | 7×7 cross-task identification-accuracy grid |
| similarity_matrix.npy | the 339×339 Day-1-vs-Day-2 similarity matrix |
| null_distribution.npy | the 5,000 shuffled-identity accuracies (empirical null) |

**visualiser.py → figures/** (reads results/ only):
accuracy_climb.png, permutation_null.png, similarity_matrix.png,
similarity_distributions.png, networks.png, region_map.gif (+ a .png still),
intelligence.png, cross_task_grid.png, task_specialisation.png.

### Our Metrics

1. **A fingerprint** = For every *pair* of the 360 regions, how strongly do they correlate? This produces a massive matrix of 360 × 359 / 2 = 64,620 numbers per person.
2. **Similarity** = How alike two fingerprints are (the correlation between the two lists of 64,620 numbers).
3. **Identification** = For each person's Day-1 fingerprint, guess that their most-similar Day-2 fingerprint is them.
4. **Accuracy** = The fraction we guessed correctly, averaged over both directions (Day 1→2 and 2→1).

---

# Test results

## 1. Core result: identification from 31% → 92%

![Accuracy climb from 31% to 91.6%](figures/accuracy_climb.png)

Our first naive attempt scored only **31%**. It was far above analytical chance, but nowhere near the ~90% the literature reaches. We didn't touch the algorithm to fix it; it all came down to our data handling. Two fixes closed the gap:

* **Fix 1: Utilising more data.** This single fix solved *two* problems at once, taking us from 31% to 89.5%.
* **Longer timeseries:** Each day actually consists of *two* scans. A single short scan is just a blurry snapshot. Extracting 64,620 features from a short signal mostly just feeds noise into the model, but gluing both runs together (~2400 timepoints) gives the model a much sharper fingerprint to work with. This did almost all the heavy lifting.
* **Canceling the scanner artifact:** The two scans we concatenated happened to use *opposite* phase-encoding directions. Combining them canceled out what would have otherwise artificially helped the model identify people based on the specific machine used rather than the brain as well as muddying the patterns.


* **Fix 2: Subtracting the "generic brain."** Every human brain is broadly wired the exact same way. That shared baseline structure is statistically very "loud" in the data, but it's useless for telling people apart. Subtracting the group-average fingerprint removes the neural activity that comes from just having a human brain, leaving behind only the individual's unique deviation from the crowd. (This pushed us from 89.5% to 91.6%).

**Null-distribution vs Analytical Chance**

![Observed accuracy against the shuffled-identity null distribution](figures/permutation_null.png)


The analytical chance level (1/339 ≈ 0.3%) assumes observations are independent and identically distributed, which connectome data can violate. So, we established chance *from the data*: we shuffled the Day-2 identities, re-scored, and repeated **5,000 times**.

The empirical null distribution sits exactly where the analytical chance level says it should (mean 0.3%, 95th percentile 0.9%, best-ever shuffle 1.8%). No permutation came anywhere near our observed 91.6%, putting the permutation p-value at **< 0.0002**, the absolute ceiling of what 5,000 permutations can resolve.

The result shows up cleanly as a bright diagonal. Everyone matches themselves across days:

![339×339 similarity matrix with bright diagonal](figures/similarity_matrix.png)

The reason this works is because there is a greater variance of two peoples fingerprints, but it is much tighter when it's the same person on different days/tasks.

![Within- vs between-subject similarity distributions](figures/similarity_distributions.png)

> **Does Fix 2 leak the answer?** No. The average never looks at *who is who*. Day 1's average uses only Day 1 data, and Day 2's average uses only Day 2 data. Nothing leaks between
> the "question" and the "answer." Every fix just cleans the data; the brains do the actual identifying.

## 2. Which networks make you identifiable? (Extension 1)

The 360 regions group into ~12 **networks** (vision, movement, attention, etc.). We wanted to know where identity actually lives, so we reran the identification using **only one network's connections** at a time.


![Identification accuracy per brain network](figures/networks.png)


Identity overwhelmingly lives in the **higher-order association networks** (posterior-multimodal carried 83%, frontoparietal 67%, the systems for flexible, integrative thinking). **Primary sensory/motor networks** carried the least (auditory 14%, somatomotor 35%). The frontoparietal result perfectly reproduces Finn et al.'s headline finding.

### The same question, region by region

A dozen networks is a pretty coarse answer, so we asked it again **360 times**, once per region, using only the 359 connections that touch that specific region. Each region is drawn at its real position in the brain, shaded by the accuracy it reaches alone:

![Rotating 3D map of per-region identification accuracy](figures/region_map.gif)


Accuracy runs from **38% to 92%** across the 360 regions (mean 61%). The absolute strongest are the **anterior insula** (L_AVI 92%, R_AVI 91%) and the neighbouring **frontal operculum** (FOP5 89%) and **temporal pole** (TGd 90%). This forms a bilateral hot spot that the broader network-level view actually smears across three different network labels. The weakest are the **early visual** regions (R_V3 38%, R_V4 and R_V1 42%, L_V3A 43%), which see roughly the same world in everyone.

*(Note: Read these numbers as *contribution*, not isolation. A region's 359 connections still span the whole brain, so the map shows which regions a fingerprint most depends on, not that the anterior insula alone can identify you).*

## 3. Does identity survive a change of task? (Extension 2)

Up to this point, both fingerprints came from a *resting* state. The much stronger claim is that your fingerprint persists across **different mental states**. To test this, we built a fingerprint per person per task and tried to match people **across** completely different tasks.

![7×7 cross-task identification grid](figures/cross_task_grid.png)


* **Same task** (diagonal): **61%**. This is the harder, low-data regime (task runs are short).
* **Different task** (off-diagonal): **26%**. This is still ~90× above the 0.3% chance level.

Identity clearly survives a change of task. (The diagonal sits below our 92% rest result simply because each task run is short; ~200–400 timepoints vs. rest's ~2400. We couldn't use our "Fix 1" here, so it's a data limitation, not a regression).

## 4. The key insight: identifiability tracks *specialisation*, not data volume

The tasks that make people most identifiable are the **higher-order, cognitively rich** ones. The obvious skeptical worry here is: "Maybe the winning tasks just had longer scans?"

They didn't. The correlation between scan length and accuracy is **r = -0.01**, essentially zero. Working-memory has the *most* data of any task yet sits near the bottom, while social and language tasks lead the pack:

![Within-task accuracy vs scan length](figures/task_specialisation.png)


**The Interpretation** (consistent across Extensions 1 and 2): Identity lives in the brain's **higher-order association systems** (social cognition, language, relational reasoning; the frontoparietal and posterior-multimodal networks). It lives there far more than in the **primary sensory/motor systems**, which are highly stereotyped across people. Two completely independent analyses (per-network *and* per-task) point to the exact same conclusion.

## 5. Are smarter people easier to identify? (Extension 3)

A natural question: are cognitively distinctive brains also *identity*-distinctive?

This dataset doesn't have an IQ field, so we built a **general-cognitive composite** from three in-scanner tasks (working-memory 2-back accuracy, relational-reasoning accuracy, and adaptive language-math difficulty), z-scored and averaged. We correlated it against each person's **self-identifiability** (how many standard deviations their own across-day match sits above the impostor crowd).

![Intelligence vs identifiability — flat regression](figures/intelligence.png)

There is **essentially no relationship** (Pearson r = +0.04, p = 0.45). The regression line is flat; the 33 missed people span the full ability range.

This is reassuring for two reasons:

1. The fingerprint picks out *the person*, not a generic "smart brain" signal. An identity biometric shouldn't rank people by ability, and this one doesn't.
2. It sharpens our earlier findings: identity concentrates in the higher-order networks and tasks, but *within* those systems, it's idiosyncratic wiring, not general ability, that makes a person unique.

*(At n = 339 the design has ~80% power to detect even r ≈ 0.15, so this is a real absence, not just a failure to look).*

## References

* **Finn et al. 2015**, *Nature Neuroscience*: connectome fingerprinting; the frontoparietal finding.
* **Glasser et al. 2016**, *Nature*: the 360-region atlas.
* **Amico & Goñi 2018**, *Scientific Reports*: "differential identifiability" (a natural next extension).
* **Noble, Scheinost & Constable 2021**, *NeuroImage*: reliability of connectome-based measures.
