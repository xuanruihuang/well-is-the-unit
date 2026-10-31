"""Separating the well from plate x drug (samewell/notes.md §101, independent review 10-29 item A4; A100).
Responses: treated-well reference D' (as core82/ver99); scores per record (pair x line, both wells as target,
averaged) as in the paper; compound-equal means and sign tests over compounds (ver99.summ).
(1) Product: verified pairs at different positions split into same-name and salt-name pairs.
(2) Dose direction: for a pair of compound k at dose c, each line's dose direction g_t = (mean response of k's wells at
    the higher other dose) - (at the lower other dose), wells of the pair and plate 13 excluded. Every response of line
    t in either well is replaced by its residual on g_t (least squares, line genes); scores recomputed. Also the share
    of the line's between-well difference (y_a - y_b) explained by g_t (median R^2). A first-order dosing or potency
    difference between the wells lies along g_t; a well shift does not.
(3) Plates 6/14 spatial structure: per same-position pair, c(x) = mean over lines of (y_6 - y_14) on the common genes;
    Pearson between c(x) and c(x') for positions at Chebyshev distance 1, 2-3 and >= 4; and between the two wells of
    the three compounds with two wells per plate (same compound, different positions) against different compounds at
    the same distances.
(4) Adagrasib 0.05 uM: 14 wells on 13 plates (two on plate 7). Per well w and line t: same = Pearson(mean of the other
    lines in w, y_tw); other(w') likewise with well w'. Advantage per well = mean over lines of same - mean over w' of
    other; sign test over the 14 wells. The pair of plate-7 wells (one plate, separately dosed wells) against the
    cross-plate pairs: their replicate-well scores, record medians. A shift that belongs to plate x drug would make
    the plate-7 pair agree better than cross-plate pairs; a well shift would not.
Output $WORKSPACE/rt/samewell/well99.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import sw_core as C
import verified as V
from sw_step2 import position
from ver99 import sign_p

OUT = (_WS + "/rt/samewell/")


def main():
    d = C.load(); D0 = d["D"]; base = d["base"]; plate = np.array(d["plate"]); L, J = D0.shape[:2]
    fin = np.isfinite(D0).all(2); Dz = np.where(fin[:, :, None], D0, 0).astype(np.float64); X = np.zeros_like(Dz)
    for p in sorted(set(plate)):
        js = np.where(plate == p)[0]; S = Dz[:, js].sum(1); n = fin[:, js].sum(1)
        for j in js:
            X[:, j] = Dz[:, j] - (S - Dz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]; Gl = [np.where(base[t] >= 0.2)[0] for t in range(L)]
    comp = np.array([V.norm(x) for x in d["drug"]]); conc = np.array([float(c) for c in d["conc"]]); pos, row, col = position(d)
    lab84 = {(x["a"], x["b"]): x["cls"] for x in json.load(open(OUT + "labels84.json"))["pairs"]}
    P = [p for p in V.pairs(d) if (fin[:, p["a"]] & fin[:, p["b"]]).sum() >= 10]
    for p in P:
        p["ver"] = lab84.get((p["a"], p["b"])) == "confirmed"
    def rec_scores(ps, Y=None):
        """Y: optional dict (t, j) -> replaced response on the line's genes."""
        Ss = {}; out = []
        for p in ps:
            a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]
            for t in ls:
                Gt = Gl[t]
                sv, ov = [], []
                for tg, rp in ((a, b), (b, a)):
                    y = X[t, tg, Gt] if Y is None else Y[(t, tg)][Gt]
                    def mo(j):
                        if Y is None:
                            return (X[fin[:, j], j][:, Gt].sum(0) - X[t, j, Gt]) / (fin[:, j].sum() - 1)
                        return np.mean([Y[(i, j)][Gt] for i in ls if i != t], 0)
                    sv.append(C.pear(mo(tg), y)); ov.append(C.pear(mo(rp), y))
                out.append((p["comp"], float(np.mean(sv)), float(np.mean(ov))))
        return out
    def summ(rec, ps):
        if not rec:
            return dict(pairs=len(ps))
        by = collections.defaultdict(list)
        for c, s, o in rec:
            by[c].append(s - o)
        cm = [np.mean(v) for v in by.values()]; k = int(sum(x > 0 for x in cm))
        return dict(pairs=len(ps), compounds=len(by), same=round(float(np.median([r[1] for r in rec])), 4), other=round(float(np.median([r[2] for r in rec])), 4),
                    diff_record_mean=round(float(np.mean([r[1] - r[2] for r in rec])), 4), diff_compound_mean=round(float(np.mean(cm)), 4), compounds_positive=k, sign_p=sign_p(k, len(cm)))
    res = {"rules": __doc__}
    # (1) product
    vd = [p for p in P if p["group"] == "diffpos" and p["ver"]]
    for nm, ps in (("same name", [p for p in vd if not p["salt"]]), ("salt name", [p for p in vd if p["salt"]])):
        res["product | " + nm] = e = summ(rec_scores(ps), ps); print("product", nm, json.dumps(e), flush=True)
    # (2) dose direction
    ok = plate != "plate13"
    def gdir(p, t):
        k, c = p["comp"], conc[p["a"]]
        others = sorted({conc[j] for j in range(J) if comp[j] == k and ok[j] and conc[j] != c})
        if len(others) < 2:
            return None
        def m(cc):
            js = [j for j in range(J) if comp[j] == k and ok[j] and conc[j] == cc and fin[t, j] and j not in (p["a"], p["b"])]
            return X[t, js].mean(0) if js else None
        hi, lo = m(others[-1]), m(others[0])
        if hi is None or lo is None:
            return None
        return hi - lo
    sets = {"diffpos verified": vd, "G614 verified": [p for p in P if p["group"] == "G614" and p["ver"] and pos[p["a"]] == pos[p["b"]]],
            "sameplate verified": [p for p in P if p["group"] == "sameplate" and p["ver"]]}
    res["dose"] = {}
    for sn, ps in sets.items():
        Y = {}; Y0 = {}; r2 = []; keep = []
        for p in ps:
            a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]; gs = {t: gdir(p, t) for t in ls}
            if any(g is None for g in gs.values()):
                continue
            keep.append(p)
            for t in ls:
                g = gs[t] - gs[t].mean()
                for j in (a, b):
                    y = X[t, j]; yc = y - y.mean(); beta = (yc @ g) / max(g @ g, 1e-12); Y[(t, j)] = y - beta * g; Y0[(t, j)] = y
                df = X[t, a] - X[t, b]; dfc = df - df.mean(); b2 = (dfc @ g) / max(g @ g, 1e-12)
                r2.append(1 - ((dfc - b2 * g) @ (dfc - b2 * g)) / max(dfc @ dfc, 1e-12))
        base_s = summ(rec_scores(keep, Y0), keep); proj = summ(rec_scores(keep, Y), keep)   # same predictor lines before and after
        res["dose"][sn] = dict(pairs_with_two_other_doses=len(keep), of=len(ps), before=base_s, after=proj, median_R2_between_well_difference=round(float(np.median(r2)), 4) if r2 else None)
        print("dose", sn, json.dumps(res["dose"][sn]), flush=True)
    # (3) 6/14 spatial structure
    g6 = [p for p in P if p["group"] == "G614" and pos[p["a"]] == pos[p["b"]]]
    cx = {}
    for p in g6:
        a, b = (p["a"], p["b"]) if plate[p["a"]] == "plate6" else (p["b"], p["a"]); ls = np.where(fin[:, a] & fin[:, b])[0]
        v = (X[ls, a][:, G0] - X[ls, b][:, G0]).mean(0); cx[int(pos[a])] = (v - v.mean(), p["comp"])
    ks = sorted(cx); by = collections.defaultdict(list); same_comp = collections.defaultdict(list)
    for i, x in enumerate(ks):
        for y in ks[i + 1:]:
            dist = max(abs(x // 12 - y // 12), abs(x % 12 - y % 12)); bin_ = "1" if dist == 1 else "2-3" if dist <= 3 else ">=4"
            r = C.pear(cx[x][0], cx[y][0])
            (same_comp if cx[x][1] == cx[y][1] else by)[bin_].append(r)
    res["spatial_614"] = {"different compounds": {k: dict(n=len(v), median=round(float(np.median(v)), 4)) for k, v in sorted(by.items())},
                          "same compound, two wells": {k: [round(float(r), 3) for r in v] for k, v in sorted(same_comp.items())}, "positions": len(ks)}
    # permutation (10-29 review: position pairs share wells and are not independent): shuffle the profiles over the
    # positions, 1,000 times; statistic = median r at distance 1 minus median r at distance >= 4 (different compounds)
    Rm = {(x, y): C.pear(cx[x][0], cx[y][0]) for i, x in enumerate(ks) for y in ks[i + 1:]}
    def stat(perm):
        d1, d4 = [], []
        for i, x in enumerate(ks):
            for y in ks[i + 1:]:
                px, py = perm[x], perm[y]
                if cx[px][1] == cx[py][1]:
                    continue
                dist = max(abs(x // 12 - y // 12), abs(x % 12 - y % 12))
                if dist == 1 or dist >= 4:
                    (d1 if dist == 1 else d4).append(Rm[(px, py)] if (px, py) in Rm else Rm[(py, px)])
        return float(np.median(d1) - np.median(d4))
    obs = stat({x: x for x in ks}); rp = np.random.default_rng(0)
    null = [stat(dict(zip(ks, rp.permutation(ks)))) for _ in range(1000)]
    res["spatial_614"]["permutation"] = dict(observed=round(obs, 4), null_q95=round(float(np.quantile(null, 0.95)), 4), null_max=round(float(max(null)), 4),
                                             p=float((1 + sum(v >= obs for v in null)) / 1001))
    print("spatial", json.dumps(res["spatial_614"]), flush=True)
    # (4) adagrasib
    aw = [j for j in range(J) if comp[j] == "adagrasib" and conc[j] == 0.05 and ok[j]]
    res["adagrasib_wells"] = [(str(plate[j]), int(pos[j])) for j in aw]
    per_well, pair_other = [], {}
    for w in aw:
        ls = [t for t in range(L) if fin[t, w]]; diffs = []
        for t in ls:
            Gt = Gl[t]; y = X[t, w, Gt]
            mo = lambda j: (X[fin[:, j], j][:, Gt].sum(0) - (X[t, j, Gt] if fin[t, j] else 0)) / (fin[:, j].sum() - fin[t, j])
            s = C.pear(mo(w), y); os_ = []
            for w2 in aw:
                if w2 != w:
                    o = C.pear(mo(w2), y); os_.append(o); pair_other.setdefault((min(w, w2), max(w, w2)), []).append(o)
            diffs.append(s - np.mean(os_))
        per_well.append(float(np.mean(diffs)))
    k = int(sum(x > 0 for x in per_well)); p7 = [j for j in aw if plate[j] == "plate7"]
    key7 = (min(p7), max(p7)) if len(p7) == 2 else None
    cross = [np.median(v) for kk, v in pair_other.items() if plate[kk[0]] != plate[kk[1]]]
    res["adagrasib"] = dict(wells=len(aw), advantage_per_well_mean=round(float(np.mean(per_well)), 4), wells_positive=k, sign_p=sign_p(k, len(per_well)),
                            plate7_pair_other_median=round(float(np.median(pair_other[key7])), 4) if key7 else None,
                            cross_plate_pairs_other_median_of_pair_medians=round(float(np.median(cross)), 4),
                            cross_plate_pair_medians_q10_q90=[round(float(q), 4) for q in np.quantile(cross, [0.1, 0.9])],
                            plate7_rank_among_pairs=int(sum(c > np.median(pair_other[key7]) for c in cross)) if key7 else None, n_cross_pairs=len(cross))
    print("adagrasib", json.dumps(res["adagrasib"]), flush=True)
    json.dump(res, open(OUT + "well99.json", "w"), indent=1, default=int); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
