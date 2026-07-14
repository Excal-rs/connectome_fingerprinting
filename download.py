#!/usr/bin/env python3
"""
============================================================================
 CONNECTOME FINGERPRINTING — DOWNLOADER (data only)
============================================================================
Run:   python download.py

Fetches the COMPLETE HCP dataset (~12 GB) from OSF into ./data and unpacks the
archives, so `analyzer.py` can run offline. Safe to re-run: anything already
present is skipped, so an interrupted download just picks up where it stopped.

Downloads (into ./data):
  hcp_rest.tgz        ~1.6 GB  -> data/hcp_rest/    resting-state scans (4 per person)
  hcp_task.tgz        ~1.3 GB  -> data/hcp_task/    7 in-scanner tasks x 2 runs + regions.npy
  hcp_covariates.tgz  ~0.8 MB  -> data/hcp/         behaviour CSVs + pseudo-demographics
  atlas.npz           ~91 KB   -> data/atlas.npz    parcel coords / vertex labels

Source: Neuromatch Academy curated HCP release (Glasser-360 parcellation).
============================================================================
"""
import os, urllib.request, tarfile

DATA = "data"

# (filename, OSF download url, dir it extracts to under ./data — or None if it's a plain file)
FILES = [
    ("hcp_rest.tgz",       "https://osf.io/bqp7m/download", "hcp_rest"),
    ("hcp_task.tgz",       "https://osf.io/s4h8j/download", "hcp_task"),
    ("hcp_covariates.tgz", "https://osf.io/x5p4g/download", "hcp"),
    ("atlas.npz",          "https://osf.io/j5kuc/download", None),
]


def fetch(fname, url, extracted):
    """Download <fname> from OSF into ./data and extract it if it's an archive.
    Skips whatever is already in place, so this is safe to run repeatedly."""
    path = os.path.join(DATA, fname)
    done = os.path.join(DATA, extracted) if extracted else path
    if os.path.exists(done):
        print(f"  {fname:20s} already present -> {done}")
        return

    if not os.path.isfile(path):
        print(f"  {fname:20s} downloading (this can take a few minutes)...")
        urllib.request.urlretrieve(url, path)

    if extracted:
        print(f"  {fname:20s} extracting -> data/{extracted}/")
        with tarfile.open(path) as t:
            t.extractall(DATA, filter="data")


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    print(f"Fetching HCP dataset into ./{DATA}/")
    for fname, url, extracted in FILES:
        fetch(fname, url, extracted)
    print("Done. Next: run  python analyzer.py")
