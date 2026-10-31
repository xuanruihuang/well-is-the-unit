"""Verified replicate pairs (samewell/notes.md §82). Plate 13's wells do not reproduce their labelled treatments
(§81), so no pair involves plate 13. Pairs: every two wells with the same salt-stripped compound (sw_core.norm) and
dose. Groups:
  sameplate   both wells on one plate (different positions)
  G614        plates 6 and 14, same position (by design; since 10-29 the 6 cross-salt pairs of gallic acid, pyridoxine
              and tofacitinib between plates 6 and 14, which differ in position, are in diffpos)
  samepos     two other plates, same position
  diffpos     two plates, different positions (exact-name replicates of plates 1-3 vs 7-12, and pairs that are
              replicates only after salt normalisation)
  Ada         adagrasib 0.05 uM, on every plate (one compound; reported apart)
Also dvia (SD over lines of the viability log2 odds against the plate's DMSO wells, §78), reproducible between plates
6 and 14 (r 0.74, §81), used as the strength measure of a pair (mean of its two wells)."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import re, json, collections, itertools
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
from sw_step2 import position

AGG = (_WS + "/tahoe_agg/"); META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")


SALT = re.compile(r"\s+(citrate|hydrochloride|dihydrochloride|maleate|mesylate|sulfate|hcl|sodium)$", re.I)


def norm(s):
    """Salt-stripped compound name: removes one trailing parenthesised qualifier and trailing salt words, never the
    whole name (sw_core.norm's greedy pattern reduces a name that starts with a parenthesis to '')."""
    t = s.strip(); u = re.sub(r"\s*\([^()]*(\([^()]*\)[^()]*)*\)\s*$", "", t)
    t = u if u else t; u = SALT.sub("", t)
    return (u if u else t).strip().lower()


def dvia_all():
    idx = json.load(open(AGG + "index.json")); cnt = np.load(AGG + "counts.npy"); Ps = len(idx["samples"]); nl = len(idx["lines"])
    cells = cnt[:, 0].reshape(nl, Ps, 2).sum(2)
    sm = pq.read_table(META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    dmso = collections.defaultdict(list)
    for k, s in enumerate(idx["samples"]):
        if info[s][0].startswith("DMSO"):
            dmso[info[s][1]].append(k)
    out = {}
    for k, s in enumerate(idx["samples"]):
        p = info[s][1]; nD = cells[:, dmso[p]].sum(1); ND = nD.sum(); n = cells[:, k]; N = n.sum()
        out[s] = float(np.std(np.log2((n + 0.5) / (N - n + 0.5)) - np.log2((nD + 0.5) / (ND - nD + 0.5))))
    return out


def pairs(d, dv=None):
    pos, _, _ = position(d); plate = d["plate"]; by = collections.defaultdict(list)
    for j in range(len(d["drug"])):
        if plate[j] != "plate13":
            by[(norm(d["drug"][j]), d["conc"][j])].append(j)
    out = []
    for (comp, conc), js in by.items():
        for a, b in itertools.combinations(sorted(js), 2):
            if comp == "adagrasib":
                g = "Ada"
            elif plate[a] == plate[b]:
                g = "sameplate"
            elif {plate[a], plate[b]} == {"plate6", "plate14"} and pos[a] == pos[b]:
                g = "G614"   # same position only (10-29, §99 A8): cross-salt pairs of plates 6/14 differ in position -> diffpos
            elif pos[a] == pos[b]:
                g = "samepos"
            else:
                g = "diffpos"
            e = dict(a=int(a), b=int(b), comp=comp, cond="%s_%s" % (comp, conc), group=g, pa=plate[a], pb=plate[b],
                     salt=d["drug"][a] != d["drug"][b])
            if dv is not None:
                e["dvia"] = 0.5 * (dv[str(d["sample"][a])] + dv[str(d["sample"][b])])
            out.append(e)
    return out
