"""Routing test on drugs (notes/mainline_routing.md §8): fold-scale program scores for Tahoe-100M, from the cached
effect arrays of tahoe_program.py (48 lines x 1,241 conditions x 6,134 genes; condition = one drug-dose on one plate;
effect per cell half = treated half minus the same-plate DMSO half). Score of program p in line c = mean over the
program's genes expressed in the line (DMSO CP10K >= 0.5, CP10K = expm1 of the line's mean DMSO log1p) of
(effect / s), s = c / (1 + c). Plate of each condition recovered from tahoe_agg's sample order (conditions are the kept
samples in index order). Output: $WORKSPACE/rt/tahoe/tahoe_scores.npz with scores (2, lines, conds, programs), lines,
conds, drug, conc, plate, moa_fine, moa_broad, targets, n_genes (lines, programs). Usage: rt_tahoe_extract.py. Exploratory."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import sys, json, ast
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, (_WS + "/rt"))
from rt_classes import programs

ARR = (_WS + "/tahoe_program/tahoe_arrays.npz")
AGG = (_WS + "/tahoe_agg")
META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata")
OUT = (_WS + "/rt/tahoe/tahoe_scores.npz")
MIN_CP10K, MIN_GENES = 0.5, 8


def main():
    z = np.load(ARR, allow_pickle=True, mmap_mode="r")
    lines, conds, genes, base = [str(x) for x in z["lines"]], [str(x) for x in z["conds"]], [str(x) for x in z["genes"]], np.asarray(z["base"], np.float64)
    # plate per condition: conditions are the kept non-DMSO samples in tahoe_agg index order
    samples = json.load(open(AGG + "/index.json"))["samples"]
    sm = pq.read_table(META + "/sample_metadata.parquet").to_pydict()
    s2 = {s: (d.strip(), dc, p) for s, d, dc, p in zip(sm["sample"], sm["drug"], sm["drugname_drugconc"], sm["plate"])}
    name = lambda s: "%s_%s" % (s2[s][0], ast.literal_eval(s2[s][1])[0][1])
    plate, j = [], 0
    for s in samples:
        if j < len(conds) and not s2[s][0].startswith("DMSO") and name(s) == conds[j]:
            plate.append(s2[s][2]); j += 1
    assert j == len(conds), (j, len(conds))
    dup = {c for c in conds if conds.count(c) > 1}
    print("conditions", len(conds), "duplicate names", len(dup), "plates", sorted(set(plate)), flush=True)
    dm = pq.read_table(META + "/drug_metadata.parquet").to_pydict()
    mf, mb, tg = (dict(zip(dm["drug"], dm[k])) for k in ("moa-fine", "moa-broad", "targets"))
    drug = [c.rsplit("_", 1)[0] for c in conds]; conc = [c.rsplit("_", 1)[1] for c in conds]
    main_p, report_p = programs()
    P = list(main_p.items()) + list(report_p.items())
    gpos = {g: k for k, g in enumerate(genes)}
    scores = np.full((2, len(lines), len(conds), len(P)), np.nan, np.float32)
    ng = np.zeros((len(lines), len(P)), int)
    for c in range(len(lines)):
        cp = np.expm1(base[c]); s = cp / (1 + cp)
        XA = np.asarray(z["XA"][c], np.float64); XB = np.asarray(z["XB"][c], np.float64)
        for k, (nm, st) in enumerate(P):
            ix = np.array(sorted(gpos[g] for g in st if g in gpos and cp[gpos[g]] >= MIN_CP10K), int)
            ng[c, k] = len(ix)
            if len(ix) >= MIN_GENES:
                scores[0, c, :, k] = (XA[:, ix] / s[ix]).mean(1); scores[1, c, :, k] = (XB[:, ix] / s[ix]).mean(1)
        print("line", c, lines[c], flush=True)
    np.savez(OUT, scores=scores, lines=np.array(lines), conds=np.array(conds), drug=np.array(drug), conc=np.array(conc),
             plate=np.array(plate), moa_fine=np.array([str(mf.get(d)) for d in drug]), moa_broad=np.array([str(mb.get(d)) for d in drug]),
             targets=np.array([str(tg.get(d)) for d in drug]), programs=np.array([nm for nm, _ in P]), n_genes=ng)
    print("saved", OUT, "programs with >= 8 genes in all lines", [nm for k, (nm, _) in enumerate(P) if (ng[:, k] >= MIN_GENES).all()], flush=True)


if __name__ == "__main__":
    main()
