"""Replication ceiling for the Rhaister comparison (samewell/notes.md §106, independent review 10-29 item A7; A100).
For the records of rh_metrics.py (held-out line x replicated test drug-dose, 5 splits), the truth of plate a and the
truth of plate b (Rhaister's per-plate cell_eval files: each plate relative to its own control wells, 2,000 genes):
ceiling = pearson_delta(truth_a, truth_b), the score a perfect copy of the other plate's measurement would get.
Reported next to Rhaister's same/other scores, medians, for plates 6/14 and the other pairs.
Output $WORKSPACE/rt/samewell/rhaister/out/rh99.json"""
import json
import numpy as np
import pyarrow.parquet as pq
from rh_metrics import pear, RH, ORIG, OUT


def main():
    m = json.load(open(OUT + "/metrics_same_other.json")); cache = {}
    def plate(p):
        if p not in cache:
            cache[p] = pq.read_table(ORIG + "/cell_eval/plate_plate%d.parquet" % p).to_pandas().set_index(["cell_line", "treatment"])
        return cache[p]
    res = {"rules": __doc__}
    for mode in ("fewshot", "zeroshot"):
        rs = m[mode]["records"]; ceil = []
        for r in rs:
            pa, pb = plate(r["plate_a"]), plate(r["plate_b"]); k = (r["cell"], r["treatment"])
            cols = [c for c in pa.columns if c in pb.columns]
            ya = np.nan_to_num(pa.loc[k, cols].to_numpy(np.float64)); yb = np.nan_to_num(pb.loc[k, cols].to_numpy(np.float64))
            ceil.append(pear(ya, yb))
        for subset, f in (("plates_6_14", lambda r: {r["plate_a"], r["plate_b"]} == {6, 14}), ("other_pairs", lambda r: {r["plate_a"], r["plate_b"]} != {6, 14})):
            ix = [i for i, r in enumerate(rs) if f(r)]
            md = lambda k: round(float(np.nanmedian([rs[i][k] for i in ix])), 4)
            res["%s|%s" % (mode, subset)] = dict(n=len(ix), rhaister_same=md("rhaister|same|pearson_delta"), rhaister_other=md("rhaister|other|pearson_delta"),
                                                 others_mean_same=md("others_mean|same|pearson_delta"), others_mean_other=md("others_mean|other|pearson_delta"),
                                                 ceiling_truth_a_vs_b=round(float(np.nanmedian([ceil[i] for i in ix])), 4))
            print(mode, subset, json.dumps(res["%s|%s" % (mode, subset)]), flush=True)
    json.dump(res, open(OUT + "/rh99.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
