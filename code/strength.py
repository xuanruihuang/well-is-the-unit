"""Cytotoxicity of a Tahoe-100M sample from cell counts (samewell/notes.md §78): -log2(total cells in the well / median
total cells of the plate's DMSO wells). Shared by the §80 scripts."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import pyarrow.parquet as pq

AGG = (_WS + "/tahoe_agg/"); META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")


def cytotoxicity():
    idx = json.load(open(AGG + "index.json")); cnt = np.load(AGG + "counts.npy"); Ps = len(idx["samples"]); nl = len(idx["lines"])
    tot = cnt[:, 0].reshape(nl, Ps, 2).sum(2).sum(0)
    sm = pq.read_table(META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    dmso = collections.defaultdict(list)
    for k, s in enumerate(idx["samples"]):
        if info[s][0].startswith("DMSO"):
            dmso[info[s][1]].append(k)
    med = {p: np.median(tot[ks]) for p, ks in dmso.items()}
    return {s: float(-np.log2(max(tot[k], 1) / med[info[s][1]])) for k, s in enumerate(idx["samples"])}


def tertile(values):
    """Tertile index (0, 1, 2) of each value within the list."""
    v = np.asarray(values, float); q = np.quantile(v, [1 / 3, 2 / 3]); return np.digitize(v, q)
