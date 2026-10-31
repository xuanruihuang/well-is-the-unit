"""Core numbers on verified replicates only (samewell/notes.md §82; A100). Pairs from verified.py (no plate 13).
Responses: D (well minus the plate's DMSO wells, cached) and D' (each half minus the line's mean of the same half over
the plate's other treated wells, leaving the well out; plate 13 stays in the plate-13 reference only, it pairs with
nothing). Per (pair, line):
  cross      Pearson of the line's two wells (line genes), D and D'
  ceiling    Spearman-Brown of the two halves of a well, mean of the two wells (D')
  adv        same-well advantage: Pearson(mean of the other lines in the target well, target) minus the same with the
             replicate well's other lines, both wells as target, line genes; D and D'
  W, Q, sw   variance components on the common genes G0 centred (D'): W = within-well half product, Q = cross-well
             half product (drug part), sw = S_same - S_cross (shared well part); shares pooled over records
  plate      same-plate check (D and D'): the predictor from another well of the target's plate (other compound) minus
             from a well of another plate; 20 random wells each (plate 13 excluded)
  delta      (diffpos only) Pearson(mean over the other lines of D'_a - D'_b, the target's D'_a - D'_b); null: the
             other lines of a random diffpos pair of another compound (200)
Per pair: permutation T with the pair as unit (one coin per line; 1,000), D', G0 centred (Gram matrices).
Summaries per group: medians or pooled ratios; 95% intervals over compounds (1,000) and two-way compounds x plates
(plates present in every pair of the group fixed; empty resamples dropped); leave-one-plate-out range; dvia tertiles
within the group. Optional argument: a label class of labels84.json (confirmed, inert, ...), restricting the pairs
to that class (§84). Output $WORKSPACE/rt/samewell/core82[_<class>].json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import sys, json, collections
import numpy as np
import sw_core as C
import verified as V

OUT = (_WS + "/rt/samewell/"); NB = 1000; NR = 20


def main():
    d = C.load(); base = d["base"]; plate = np.array(d["plate"]); L, J = d["D"].shape[:2]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]; comp_all = np.array([V.norm(x) for x in d["drug"]])
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    XA = np.stack([np.asarray(z["XA"][i], np.float32) for i in range(L)]).astype(np.float64)
    XB = np.stack([np.asarray(z["XB"][i], np.float32) for i in range(L)]).astype(np.float64)
    fin = np.isfinite(XA).all(2) & np.isfinite(XB).all(2)
    D = np.where(fin[:, :, None], (XA + XB) / 2, 0)
    for H in (XA, XB):
        Hz = np.where(fin[:, :, None], H, 0)
        for p in sorted(set(plate)):
            js = np.where(plate == p)[0]; S = Hz[:, js].sum(1); n = fin[:, js].sum(1)
            for j in js:
                H[:, j] -= (S - Hz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]
    X = np.where(fin[:, :, None], (XA + XB) / 2, 0)
    VAR = {"D": D, "Dprime": X}; Ssum = {k: v.sum(0) for k, v in VAR.items()}; nf = fin.sum(0)
    dv = V.dvia_all(); P = [p for p in V.pairs(d, dv) if fin[:, p["a"]].sum() >= 10]
    sub = sys.argv[1] if len(sys.argv) > 1 else None
    if sub:
        cls_, fn_ = (sub.split("@") + ["labels84"])[:2]
        keep = {(x["a"], x["b"]) for x in json.load(open(OUT + fn_ + ".json"))["pairs"] if x["cls"] == cls_}
        sub = cls_
        P = [p for p in P if (p["a"], p["b"]) in keep]
    rng = np.random.default_rng(0); recs = []; pairT = {}; Yd = {}
    pool_other = np.where(plate != "plate13")[0]
    sbf = lambda c: 2 * c / (1 + c)
    for pi, p in enumerate(P):
        a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]; n = len(ls)
        if n < 10:
            continue
        p["ls"] = ls
        Ya = X[ls, a][:, G0]; Yb = X[ls, b][:, G0]; Ya = Ya - Ya.mean(1, keepdims=True); Yb = Yb - Yb.mean(1, keepdims=True)
        Gaa, Gbb, Gab = Ya @ Ya.T, Yb @ Yb.T, Ya @ Yb.T; Gba = Gab.T
        if p["group"] == "diffpos":
            Yd[pi] = (ls, Ya - Yb)
        for k, l in enumerate(ls):
            ha = [h[l, a, G0] - h[l, a, G0].mean() for h in (XA, XB)]; hb = [h[l, b, G0] - h[l, b, G0].mean() for h in (XA, XB)]
            W = 0.5 * (ha[0] @ ha[1] + hb[0] @ hb[1]); Q = 0.25 * sum(u @ v for u in ha for v in hb)
            o = np.arange(n) != k
            sw = 0.5 * (Gaa[k, o].mean() + Gbb[k, o].mean()) - 0.5 * (Gab[k, o].mean() + Gab[o, k].mean())
            gl = np.where(base[l] >= 0.2)[0]; r = dict(pi=pi, line=int(l), W=W, Q=Q, sw=sw)
            for key, Vx in VAR.items():
                mo = lambda j: (Ssum[key][j, gl] - Vx[l, j, gl]) / (nf[j] - 1)
                r["cross_" + key] = C.pear(Vx[l, a, gl], Vx[l, b, gl])
                adv, pl = [], []
                for tg, rp in ((a, b), (b, a)):
                    y = Vx[l, tg, gl]; adv.append(C.pear(mo(tg), y) - C.pear(mo(rp), y))
                if key == "Dprime" or key == "D":
                    for tg, rp in ((a, b), (b, a)):
                        y = Vx[l, tg, gl]
                        same = pool_other[(plate[pool_other] == plate[tg]) & (comp_all[pool_other] != p["comp"]) & fin[l, pool_other]]
                        else_ = pool_other[(plate[pool_other] != plate[tg]) & (plate[pool_other] != plate[rp]) & fin[l, pool_other]]
                        ws = rng.choice(same, NR, replace=False); we = rng.choice(else_, NR, replace=False)
                        pl.append(np.mean([C.pear(mo(j), y) for j in ws]) - np.mean([C.pear(mo(j), y) for j in we]))
                r["adv_" + key] = float(np.mean(adv)); r["plate_" + key] = float(np.mean(pl))
            r["ceiling"] = float(np.mean([sbf(C.pear(XA[l, w, gl], XB[l, w, gl])) for w in (a, b)]))
            recs.append(r)
        def pairstat(sig):
            vals = []
            for (Gtt, Gto, G11, G12, G21, G22) in ((Gaa, Gab, Gaa, Gab, Gba, Gbb), (Gbb, Gba, Gbb, Gba, Gab, Gaa)):
                s = sig.astype(bool); S0, S1 = ~s[:, None] & ~s[None, :], ~s[:, None] & s[None, :]
                S2, S3 = s[:, None] & ~s[None, :], s[:, None] & s[None, :]
                MA = np.where(S0, G11, 0) + np.where(S1, G12, 0) + np.where(S2, G21, 0) + np.where(S3, G22, 0)
                MB = np.where(S0, G22, 0) + np.where(S1, G21, 0) + np.where(S2, G12, 0) + np.where(S3, G11, 0)
                inA = np.where(s[None, :], Gto, Gtt); inB = np.where(s[None, :], Gtt, Gto); dg = np.diag(Gtt)
                for M, inn in ((MA, inA), (MB, inB)):
                    num = inn.sum(1) - np.diag(inn); nrm = M.sum() - 2 * M.sum(1) + np.diag(M)
                    vals.append(num / np.sqrt(np.maximum(nrm, 1e-12) * dg))
            return float(np.mean(np.concatenate([vals[0] - vals[1], vals[2] - vals[3]])))
        pairT[pi] = (pairstat(np.zeros(n, bool)), np.array([pairstat(rng.random(n) < 0.5) for _ in range(NB)]))
        if pi % 50 == 0:
            print("pair", pi, len(P), flush=True)
    # delta test
    dp = list(Yd); delta = {}
    for pi in dp:
        ls, Dl = Yd[pi]; n = len(ls); Sd = Dl.sum(0); others = [q for q in dp if P[q]["comp"] != P[pi]["comp"]]
        for k, l in enumerate(ls):
            obs = C.pear((Sd - Dl[k]) / (n - 1), Dl[k]); nul = []
            for q in rng.choice(others, 200):
                lq, Dq = Yd[q]; m = lq != l; nul.append(C.pear(Dq[m].mean(0), Dl[k]))
            delta[(pi, int(l))] = (obs, float(np.median(nul)), float(np.quantile(nul, 0.95)))
    plates = sorted(set(plate), key=lambda x: int(x[5:])); pix = {q: k for k, q in enumerate(plates)}
    Kp = np.stack([np.bincount(rng.integers(0, len(plates), len(plates)), minlength=len(plates)) for _ in range(NB)]).astype(float)
    res = {"rules": __doc__, "pairs": {g: [sum(1 for p in P if p["group"] == g and "ls" in p), len({p["comp"] for p in P if p["group"] == g and "ls" in p})]
                                       for g in ("diffpos", "sameplate", "G614", "samepos", "Ada")}}
    print("pairs", json.dumps(res["pairs"]), flush=True)
    def summarise(rs, tag):
        comps = sorted({P[r["pi"]]["comp"] for r in rs}); ci = np.array([comps.index(P[r["pi"]]["comp"]) for r in rs])
        Kc = np.stack([np.bincount(rng.integers(0, len(comps), len(comps)), minlength=len(comps)) for _ in range(NB)]).astype(float)
        pa = np.array([pix[P[r["pi"]]["pa"]] for r in rs]); pb = np.array([pix[P[r["pi"]]["pb"]] for r in rs])
        fixed = [k for k in range(len(plates)) if np.all((pa == k) | (pb == k))]; Kq = Kp.copy(); Kq[:, fixed] = 1
        w2 = Kc[:, ci] * Kq[:, pa] * np.where(pa == pb, 1.0, Kq[:, pb]); wc = Kc[:, ci]
        def iv(w, num, den):
            keep = w.sum(1) > 0; v = (w[keep] @ num) / (w[keep] @ den); return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]
        def loo(num, den):
            out = [float(num[(pa != k) & (pb != k)].sum() / den[(pa != k) & (pb != k)].sum()) for k in sorted((set(pa) | set(pb)) - set(fixed)) if ((pa != k) & (pb != k)).any()]
            return [round(min(out), 3), round(max(out), 3)] if out else None
        e = dict(n_records=len(rs), n_pairs=len({r["pi"] for r in rs}), n_compounds=len(comps), fixed_plates=[plates[k] for k in fixed],
                 dvia_median=round(float(np.median([P[r["pi"]]["dvia"] for r in rs])), 3))
        for key in ("cross_D", "cross_Dprime", "ceiling"):
            e[key] = round(float(np.nanmedian([r[key] for r in rs])), 3)
        one = np.ones(len(rs))
        for key in ("adv_D", "adv_Dprime", "plate_D", "plate_Dprime"):
            v = np.array([r[key] for r in rs]); ok = np.isfinite(v); v0 = np.where(ok, v, 0)
            e[key] = [round(float(v[ok].mean()), 3)] + iv(wc, v0, ok * one) + iv(w2, v0, ok * one)
            if key == "adv_Dprime":
                e[key + "_loo"] = loo(v0, ok * one)
        Wv = np.array([r["W"] for r in rs]); Qv = np.array([r["Q"] for r in rs]); Sv = np.array([r["sw"] for r in rs])
        e["drug_share"] = [round(float(Qv.sum() / Wv.sum()), 3)] + iv(wc, Qv, Wv) + iv(w2, Qv, Wv)
        e["shared_well_share"] = [round(float(Sv.sum() / Wv.sum()), 3)] + iv(wc, Sv, Wv)
        e["own_well_share"] = round(float(1 - (Qv.sum() + Sv.sum()) / Wv.sum()), 3)
        ps = sorted({r["pi"] for r in rs if r["pi"] in pairT})
        if ps:
            obs = np.mean([pairT[q][0] for q in ps]); perm = np.mean([pairT[q][1] for q in ps], 0)
            e["perm"] = dict(observed=round(float(obs), 3), perm_q=[round(float(x), 3) for x in np.percentile(perm, [2.5, 97.5])], perm_max=round(float(perm.max()), 3),
                             p=float((1 + (perm >= obs).sum()) / (1 + NB)))
        dl = [delta[(r["pi"], r["line"])] for r in rs if (r["pi"], r["line"]) in delta]
        if dl:
            e["delta"] = dict(n=len(dl), observed_median=round(float(np.median([x[0] for x in dl])), 3), null_median=round(float(np.median([x[1] for x in dl])), 3),
                              share_above_null95=round(float(np.mean([x[0] > x[2] for x in dl])), 3))
        res[tag] = e; print(tag, json.dumps(e), flush=True)
    for g in ("diffpos", "sameplate", "G614", "samepos", "Ada"):
        rs = [r for r in recs if P[r["pi"]]["group"] == g]
        if len(rs) < 20:
            continue
        summarise(rs, g)
        if g in ("diffpos", "G614"):
            pv = sorted({r["pi"] for r in rs}); q = np.quantile([P[x]["dvia"] for x in pv], [1 / 3, 2 / 3])
            for t in range(3):
                st = [r for r in rs if int(np.digitize(P[r["pi"]]["dvia"], q)) == t]
                if len(st) >= 20:
                    summarise(st, "%s | dvia %d" % (g, t))
        if g == "diffpos":
            for nm, f in (("salt only", lambda r: P[r["pi"]]["salt"]), ("exact names only", lambda r: not P[r["pi"]]["salt"])):
                st = [r for r in rs if f(r)]
                if len(st) >= 20:
                    summarise(st, "diffpos | " + nm)
    json.dump(res, open(OUT + "core82%s.json" % ("_" + sub if sub else ""), "w"), indent=1)
    json.dump([dict(r, group=P[r["pi"]]["group"], comp=P[r["pi"]]["comp"], dvia=P[r["pi"]]["dvia"]) for r in recs], open(OUT + "core82%s_records.json" % ("_" + sub if sub else ""), "w"))
    print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
