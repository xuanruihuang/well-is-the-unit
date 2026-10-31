"""Summary of the §91 BioPert runs (samewell/notes.md §91; A100). Joins bp91_train.py records of the BioPert model
(per seed) with the ECFP model's records of the same reference (same target well, line and gene set) and summarises by
group and label class: medians over records of same, other, fingerprint, copy-A549 (same / other), line centroid and
ceiling; mean over compounds of same - other, same - fingerprint and other - fingerprint with 95% intervals from 1,000
bootstrap resamples of compounds (seed 0). Output $WORKSPACE/rt/samewell/bp91/summary.json. Usage: bp91_sum.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, json, collections
import numpy as np

ROOT = (_WS + "/rt/samewell/bp91/")
SUBSETS = [("G614 all", "G614", None), ("G614 confirmed", "G614", ("cls84", "confirmed")), ("diffpos all", "diffpos", None),
           ("diffpos confirmed", "diffpos", ("cls84", "confirmed")), ("diffpos nodvia", "diffpos", ("cls85", "nodvia")),
           ("sameplate confirmed", "sameplate", ("cls84", "confirmed")), ("sameplate nodvia", "sameplate", ("cls85", "nodvia")),
           ("Ada confirmed", "Ada", ("cls84", "confirmed"))]


def regroup():
    """Pair groups from the current verified.pairs (10-29: plates-6/14 cross-salt pairs moved to diffpos); records keep
    their sample indices t and o (pb indices, as bp91_data.py), so no retraining is needed."""
    import verified as V, bp91_data as B
    d = B.light_d(); si = {s: k for k, s in enumerate(json.load(open(B.PB + "/index.json"))["samples"])}
    pb = lambda j: si[str(d["sample"][j])]
    return {(min(pb(p["a"]), pb(p["b"])), max(pb(p["a"]), pb(p["b"]))): p["group"] for p in V.pairs(d)}


def main():
    rb = np.random.default_rng(0); res = {"rules": __doc__}; GR = regroup()
    for ref in ("dprime", "dmso"):
        fpf = ROOT + ref + "/ecfp_42_records.json"
        fp = {(r["t"], r["target"], r["genes"]): r["fp"] for r in json.load(open(fpf))} if os.path.exists(fpf) else {}
        for seed in (42, 1, 2):
            f = ROOT + "%s/bp_%d_records.json" % (ref, seed)
            if not os.path.exists(f):
                continue
            recs = json.load(open(f)); std = json.load(open(ROOT + "%s/bp_%d.json" % (ref, seed)))
            moved = collections.Counter()
            for r in recs:
                g = GR[(min(r["t"], r["o"]), max(r["t"], r["o"]))]; moved[(r["group"], g)] += r["group"] != g; r["group"] = g
            print(ref, seed, "regrouped records", {"%s->%s" % k: v for k, v in moved.items() if v}, flush=True)
            res["%s seed %d standard" % (ref, seed)] = dict(test_pearson=std["test_pearson"], best_epoch=std["best_epoch"], n=std["n"])
            for gs in ("expr", "all"):
                for name, g, c in SUBSETS:
                    rs = [dict(r, fp=fp.get((r["t"], r["target"], gs), np.nan)) for r in recs if r["genes"] == gs and r["group"] == g and (c is None or r[c[0]] == c[1])]
                    if not rs:
                        continue
                    comps = sorted({r["comp"] for r in rs})
                    out = dict(n_records=len(rs), n_pairs=len({r["pair"] for r in rs}), n_comp=len(comps))
                    for k in ("same", "other", "fp", "copy_same", "copy_other", "linemean", "ceiling"):
                        out[k] = round(float(np.nanmedian([r[k] for r in rs])), 3)
                    for k, f_ in (("same-other", lambda r: r["same"] - r["other"]), ("same-fp", lambda r: r["same"] - r["fp"]), ("other-fp", lambda r: r["other"] - r["fp"])):
                        byc = collections.defaultdict(list)
                        for r in rs:
                            byc[r["comp"]].append(f_(r))
                        cm = np.array([np.nanmean(byc[x]) for x in comps])
                        if np.isnan(cm).all():
                            continue
                        bs = [np.nanmean(cm[rb.integers(0, len(cm), len(cm))]) for _ in range(1000)]
                        out[k] = [round(float(np.nanmean(cm)), 3)] + [round(float(q), 3) for q in np.nanquantile(bs, [0.025, 0.975])]
                    res["%s seed %d %s | %s" % (ref, seed, gs, name)] = out
                    if seed == 42 or gs == "expr":
                        print(ref, seed, gs, name, json.dumps(out), flush=True)
    json.dump(res, open(ROOT + "summary.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
