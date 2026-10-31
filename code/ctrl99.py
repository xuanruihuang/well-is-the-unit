"""Robustness of the control-well pairs (samewell/notes.md §104, independent review 10-29 item A6; A100).
Per plate (plate 13 apart) and line: R = profile(control well 1) - profile(control well 2); same-pair prediction =
mean of the other lines' R on that plate; other-plates prediction = mean of the other lines' R on another plate
(averaged over the other plates); Pearson on the line's genes (base >= 0.2, cached genes mapped to the pseudobulk
genes). Profiles:
  log1p-mean   mean over cells of log1p(1e4 c / n) (sw_v1a.control_profiles; the paper's numbers)
  pb-logcpm    raw counts summed over the line's cells in the well (pb/counts_lsh.npy), log(1 + 1e6 c / N): no
               per-cell nonlinearity, so depth per cell cannot create a shared shift through log1p
  pb-matched   as pb-logcpm after binomial thinning of the deeper well's summed counts to the shallower well's total
               (per line and plate; seed 0): the two wells of a pair then have the same depth for every line
Tests: per-plate medians and the number of plates with same > other; the 13 x 13 matrix M[i, j] = median over lines of
plate i of Pearson(mean of the other lines' R on plate j, line's R on plate i), with a permutation test of the
diagonal against random matchings (100,000 permutations of the predictor plates; statistic = mean diagonal);
bootstrap over plates of the per-plate difference (diagonal - row mean of the off-diagonal).
Plate 13: its control wells are inferred at B11/B12 (sample numbers), not H11/H12; its labels do not reproduce
(§81), so this is reported apart: same-pair prediction, and the other plates' pairs as predictors of plate 13's R
(different position) against the other plates' pairs as predictors of each other (same position H11/H12).
Output $WORKSPACE/rt/samewell/ctrl99.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
import sw_v1a

OUT = (_WS + "/rt/samewell/"); PB = OUT + "pb/"; META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")


def main():
    d = C.load(); base = d["base"]; L = len(d["lines"])
    idx = json.load(open(PB + "index.json")); X = np.load(PB + "counts_lsh.npy", mmap_mode="r"); cnt = np.load(PB + "cnt_lsh.npy")
    ns = len(idx["samples"]); pgene = {g: k for k, g in enumerate(idx["genes"])}
    gmap = np.array([pgene.get(g, -1) for g in d["genes"]]); print("cached genes found in pb", int((gmap >= 0).sum()), "of", len(gmap), flush=True)
    lpos = [idx["lines"].index(ln) if ln in idx["lines"] else -1 for ln in d["lines"]]; print("lines found", sum(x >= 0 for x in lpos), flush=True)
    sm = pq.read_table(META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    plates = sorted({p for _, p in info.values()}, key=lambda p: int(p[5:]))
    spos = {s: k for k, s in enumerate(idx["samples"])}
    ctrl = {pl: sorted([s for s in idx["samples"] if info[s][1] == pl and info[s][0].startswith("DMSO")], key=lambda s: int(s[4:])) for pl in plates}
    res = {"rules": __doc__, "control_samples": ctrl}
    rng = np.random.default_rng(0)
    def pbcounts(i, s):
        c = lpos[i]; k = spos[s]; rows = [(c * ns + k) * 2 + h for h in range(2)]
        return np.asarray(X[rows[0]], np.float64) + np.asarray(X[rows[1]], np.float64), cnt[rows, 0].sum()
    ok = gmap >= 0
    prof_lm, _ = sw_v1a.control_profiles(d)
    R = {"log1p-mean": {}, "pb-logcpm": {}, "pb-matched": {}}
    for pl in plates:
        if len(ctrl[pl]) != 2:
            continue
        for i in range(L):
            w = prof_lm[(i, pl)]
            if len(w) == 2 and all(x is not None for x in w):
                R["log1p-mean"][(i, pl)] = (w[0] - w[1]).astype(np.float64)
            if lpos[i] < 0:
                continue
            (c1, n1), (c2, n2) = pbcounts(i, ctrl[pl][0]), pbcounts(i, ctrl[pl][1])
            if min(n1, n2) < 10:
                continue
            g1, g2 = np.zeros(len(gmap)), np.zeros(len(gmap)); g1[ok] = c1[gmap[ok]]; g2[ok] = c2[gmap[ok]]
            T1, T2 = c1.sum(), c2.sum()
            lc = lambda c, T: np.log1p(1e6 * c / T)
            R["pb-logcpm"][(i, pl)] = lc(g1, T1) - lc(g2, T2)
            if T1 > T2:
                t1 = rng.binomial(c1.astype(np.int64), T2 / T1); h1 = np.zeros(len(gmap)); h1[ok] = t1[gmap[ok]]; R["pb-matched"][(i, pl)] = lc(h1, t1.sum()) - lc(g2, T2)
            else:
                t2 = rng.binomial(c2.astype(np.int64), T1 / T2); h2 = np.zeros(len(gmap)); h2[ok] = t2[gmap[ok]]; R["pb-matched"][(i, pl)] = lc(g1, T1) - lc(h2, t2.sum())
        print("plate", pl, flush=True)
    Gl = [np.where((base[t] >= 0.2) & ok)[0] for t in range(L)]
    main_plates = [pl for pl in plates if pl != "plate13" and len(ctrl[pl]) == 2]
    res["versions"] = {}
    for name, Rv in R.items():
        mean_other = {}
        for pl in plates:
            have = [i for i in range(L) if (i, pl) in Rv]
            S = np.sum([Rv[(i, pl)] for i in have], 0) if have else None
            mean_other[pl] = (S, have)
        def pred(t, pl):
            S, have = mean_other[pl]
            if S is None:
                return None
            n = len(have) - (t in have)
            return (S - (Rv[(t, pl)] if t in have else 0)) / n
        M = np.full((len(main_plates), len(main_plates)), np.nan); rec_s, rec_o = [], []
        for a, pa in enumerate(main_plates):
            ts = [t for t in range(L) if (t, pa) in Rv]
            cols = collections.defaultdict(list)
            for t in ts:
                y = Rv[(t, pa)][Gl[t]]
                for b, pb_ in enumerate(main_plates):
                    p = pred(t, pb_)
                    if p is not None:
                        cols[b].append(C.pear(p[Gl[t]], y))
                rec_s.append(cols[a][-1]); rec_o.append(np.mean([cols[b][-1] for b in range(len(main_plates)) if b != a]))
            for b in cols:
                M[a, b] = np.median(cols[b])
        diag = np.diag(M); off = np.array([np.nanmean(np.delete(M[a], a)) for a in range(len(M))])
        perm = np.array([np.mean(M[np.arange(len(M)), rng.permutation(len(M))]) for _ in range(100000)])
        boot = [np.mean((diag - off)[k]) for k in (rng.integers(0, len(M), len(M)) for _ in range(5000))]
        e = dict(records=len(rec_s), same_median=round(float(np.median(rec_s)), 4), other_median=round(float(np.median(rec_o)), 4),
                 advantage_record_mean=round(float(np.mean(np.array(rec_s) - np.array(rec_o))), 4),
                 plates_same_gt_other=int(sum(diag > off)), plates=len(M), per_plate=[[round(float(x), 3), round(float(y), 3)] for x, y in zip(diag, off)],
                 diagonal_mean=round(float(diag.mean()), 4), permutation_p=float((1 + (perm >= diag.mean()).sum()) / (1 + len(perm))),
                 plate_bootstrap_diff=[round(float(np.mean(diag - off)), 4)] + [round(float(q), 4) for q in np.percentile(boot, [2.5, 97.5])])
        # plate 13 apart
        if len(ctrl.get("plate13", [])) == 2:
            t13 = [t for t in range(L) if (t, "plate13") in Rv]; s13, o13 = [], []
            for t in t13:
                y = Rv[(t, "plate13")][Gl[t]]; p = pred(t, "plate13")
                s13.append(C.pear(p[Gl[t]], y)); o13.append(np.mean([C.pear(pred(t, q)[Gl[t]], y) for q in main_plates if pred(t, q) is not None]))
            e["plate13"] = dict(records=len(t13), same_median=round(float(np.median(s13)), 4), other_plates_median=round(float(np.median(o13)), 4),
                                control_samples=ctrl["plate13"])
        res["versions"][name] = e; print(name, json.dumps({k: v for k, v in e.items() if k != "per_plate"}), flush=True)
    json.dump(res, open(OUT + "ctrl99.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
