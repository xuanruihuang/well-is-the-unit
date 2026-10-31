"""Reviewer questions 1 and 5 (samewell/notes.md §96; runs locally on figdata89.json from figexport89.py).
1  Sensitivity of verification to its threshold: a pair is verified if, by expression or by viability, each well ranks
   its partner among the top x of the partner plate's treated wells (x = 5%, 10% as in the paper, 20%). For each x and
   group (different positions, one plate, plates 6/14): pairs and compounds verified, medians over pairs of the
   same-well and replicate-well predictions (pair medians over lines, D', as Fig. 1c), and the mean over pairs of their
   difference.
5  Per-compound breakdown of Fig. 1c: for verified pairs and pairs without detectable viability effect, at different
   positions and on one plate, per compound the number of pairs and the medians over its pairs of the two predictions.
Usage: python thr96.py <figdata89.json> <out.json>"""
import sys, json, collections
import numpy as np


def main():
    d = json.load(open(sys.argv[1])); P, C = d["pairs"], d["check83_rows"]
    assert len(P) == len(C) and all(p["group"] == c["group"] and p["salt"] == c["salt"] and p["cls"] == c["cls"] for p, c in zip(P, C))
    res = {"rules": __doc__, "threshold": {}, "per_compound": {}}
    for x in (0.95, 0.90, 0.80):
        for g in ("diffpos", "sameplate", "G614"):
            rs = [p for p, c in zip(P, C) if p["group"] == g and (min(c["expr_pct"]) >= x or min(c["via_pct"]) >= x)]
            e = dict(pairs=len(rs), compounds=len({p["comp"] for p in rs}))
            if rs:
                e.update(same=round(float(np.median([p["same"] for p in rs])), 4), other=round(float(np.median([p["other"] for p in rs])), 4),
                         diff_mean=round(float(np.mean([p["same"] - p["other"] for p in rs])), 4))
            res["threshold"]["top %d%% | %s" % (round(100 * (1 - x)), g)] = e; print("top %d%%" % round(100 * (1 - x)), g, e)
    groups = [("Verified, different positions", lambda p: p["group"] == "diffpos" and p["cls"] == "confirmed"),
              ("Verified, one plate", lambda p: p["group"] == "sameplate" and p["cls"] == "confirmed"),
              ("No viability effect, different positions", lambda p: p["group"] == "diffpos" and p["nodvia"]),
              ("No viability effect, one plate", lambda p: p["group"] == "sameplate" and p["nodvia"])]
    for name, f in groups:
        by = collections.defaultdict(list)
        for p in P:
            if f(p):
                by[p["comp"]].append(p)
        res["per_compound"][name] = [dict(compound=c, pairs=len(v), same=round(float(np.median([p["same"] for p in v])), 2),
                                          other=round(float(np.median([p["other"] for p in v])), 2)) for c, v in sorted(by.items())]
        print(name, res["per_compound"][name])
    vd = [p for p in P if p["group"] == "diffpos" and p["cls"] == "confirmed"]
    tr = [p["same"] - p["other"] for p in vd if p["comp"] == "trametinib"]; ot = [p["same"] - p["other"] for p in vd if p["comp"] != "trametinib"]
    res["trametinib"] = dict(pairs=len(tr), diff_mean=round(float(np.mean(tr)), 4), other_pairs=len(ot), other_diff_mean=round(float(np.mean(ot)), 4),
                             min_pair_diff=round(float(min(tr + ot)), 4))
    print("trametinib", res["trametinib"])
    json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
