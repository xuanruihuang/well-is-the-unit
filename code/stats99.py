"""Uncertainty and estimator checks for the shares and the advantage (samewell/notes.md §102, independent review 10-29
item A5; A100). Records as str92.py (treated-well reference D', common genes G0 centred, per pair x line: within-well
half products Wa, Wb, cross-well half product Q, common well part sw; advantage = same - replicate-well score on the
line's genes, both wells as target, averaged). Plates 6/14: same-position pairs only (§99 A8).
Per set:
  pooled          well share 1 - sum Q / sum W, common part sum sw / sum W (the paper's estimators), and the
                  strength-robust version 1 - sum Q / sum sqrt(Wa Wb) (record level, as str92)
  pair level      the same with sqrt taken after summing over the pair's lines: 1 - sum Q / sum_pairs sqrt(SWa SWb)
                  (review: record-level sqrt of noisy W is biased)
  intervals       95% percentile intervals, 2,000 resamples, not truncated: (i) over compounds; (ii) two-way
                  (compounds and plates resampled independently, a record weighted by the product of the counts of its
                  compound and of its two plates)
  equal weight    per-compound share (sums within the compound), mean and median over compounds; compound-equal
                  advantage (mean over compounds of the compound's record mean) and sign test over compounds
  without trametinib  pooled shares and advantage
  per compound    pairs, records, well share, advantage
Output $WORKSPACE/rt/samewell/stats99.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import sw_core as C
import verified as V
from sw_step2 import position
from ver99 import sign_p

OUT = (_WS + "/rt/samewell/")


def main():
    d = C.load(); base = d["base"]; plate = np.array(d["plate"]); L, J = d["D"].shape[:2]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    XA = np.stack([np.asarray(z["XA"][i], np.float32) for i in range(L)]).astype(np.float64)
    XB = np.stack([np.asarray(z["XB"][i], np.float32) for i in range(L)]).astype(np.float64)
    fin = np.isfinite(XA).all(2) & np.isfinite(XB).all(2)
    for H in (XA, XB):
        Hz = np.where(fin[:, :, None], H, 0)
        for p in sorted(set(plate)):
            js = np.where(plate == p)[0]; S = Hz[:, js].sum(1); n = fin[:, js].sum(1)
            for j in js:
                H[:, j] -= (S - Hz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]
    X = np.where(fin[:, :, None], (XA + XB) / 2, 0); Ss = X.sum(0); nf = fin.sum(0)
    pos, _, _ = position(d)
    lab = {}
    for fn in ("labels84", "labels85"):
        for x in json.load(open(OUT + fn + ".json"))["pairs"]:
            lab[(fn, min(x["a"], x["b"]), max(x["a"], x["b"]))] = x["cls"]
    pairs = [p for p in V.pairs(d) if (fin[:, p["a"]] & fin[:, p["b"]]).sum() >= 10]
    for p in pairs:
        a, b = min(p["a"], p["b"]), max(p["a"], p["b"]); p["cls84"] = lab.get(("labels84", a, b)); p["cls85"] = lab.get(("labels85", a, b))
    sets = {"diffpos verified": lambda p: p["group"] == "diffpos" and p["cls84"] == "confirmed",
            "diffpos nodvia": lambda p: p["group"] == "diffpos" and p["cls85"] == "nodvia",
            "sameplate verified": lambda p: p["group"] == "sameplate" and p["cls84"] == "confirmed",
            "sameplate nodvia": lambda p: p["group"] == "sameplate" and p["cls85"] == "nodvia",
            "G614 all": lambda p: p["group"] == "G614" and pos[p["a"]] == pos[p["b"]],
            "G614 verified": lambda p: p["group"] == "G614" and pos[p["a"]] == pos[p["b"]] and p["cls84"] == "confirmed"}
    want = [pi for pi, p in enumerate(pairs) if any(f(p) for f in sets.values())]
    recs = []
    for pi in want:
        p = pairs[pi]; a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]; n = len(ls)
        Ya = X[ls, a][:, G0]; Yb = X[ls, b][:, G0]; Ya = Ya - Ya.mean(1, keepdims=True); Yb = Yb - Yb.mean(1, keepdims=True)
        Gaa, Gbb, Gab = Ya @ Ya.T, Yb @ Yb.T, Ya @ Yb.T
        for k, l in enumerate(ls):
            ha = [h[l, a, G0] - h[l, a, G0].mean() for h in (XA, XB)]; hb = [h[l, b, G0] - h[l, b, G0].mean() for h in (XA, XB)]
            Wa, Wb = ha[0] @ ha[1], hb[0] @ hb[1]; Q = 0.25 * sum(u @ v for u in ha for v in hb)
            o = np.arange(n) != k; Saa, Sbb = Gaa[k, o].mean(), Gbb[k, o].mean(); Sab = 0.5 * (Gab[k, o].mean() + Gab[o, k].mean())
            gl = np.where(base[l] >= 0.2)[0]; adv = []
            for tg, rp in ((a, b), (b, a)):
                y = X[l, tg, gl]; mo = lambda j: (Ss[j, gl] - X[l, j, gl]) / (nf[j] - 1)
                adv.append(C.pear(mo(tg), y) - C.pear(mo(rp), y))
            recs.append(dict(pi=pi, comp=p["comp"], pa=str(plate[a]), pb=str(plate[b]), Wa=float(Wa), Wb=float(Wb), W=float(0.5 * (Wa + Wb)),
                             Wg=float(np.sqrt(max(Wa, 0) * max(Wb, 0))), Q=float(Q), sw=float(0.5 * (Saa + Sbb) - Sab), adv=float(np.mean(adv))))
    print("records", len(recs), flush=True)
    res = {"rules": __doc__}; rng = np.random.default_rng(0); B = 2000
    def block(rs):
        A = {k: np.array([r[k] for r in rs]) for k in ("Wa", "Wb", "W", "Wg", "Q", "sw", "adv")}
        comps = sorted({r["comp"] for r in rs}); ci = np.array([comps.index(r["comp"]) for r in rs])
        pls = sorted({r["pa"] for r in rs} | {r["pb"] for r in rs}); pa = np.array([pls.index(r["pa"]) for r in rs]); pb = np.array([pls.index(r["pb"]) for r in rs])
        pis = sorted({r["pi"] for r in rs}); pidx = np.array([pis.index(r["pi"]) for r in rs])
        def pair_sums(w):
            out = {}
            for k in ("Wa", "Wb", "Q"):
                out[k] = np.bincount(pidx, weights=w * A[k], minlength=len(pis))
            return out
        def stats(w):
            sW = (w * A["W"]).sum(); ps = pair_sums(w)
            return dict(well_share=1 - (w * A["Q"]).sum() / sW, common=(w * A["sw"]).sum() / sW, beyond_strength_record=1 - (w * A["Q"]).sum() / (w * A["Wg"]).sum(),
                        beyond_strength_pair=1 - ps["Q"].sum() / np.sqrt(np.maximum(ps["Wa"], 0) * np.maximum(ps["Wb"], 0)).sum(),
                        advantage=(w * A["adv"]).sum() / w.sum())
        point = stats(np.ones(len(rs)))
        bc = [stats(np.bincount(rng.integers(0, len(comps), len(comps)), minlength=len(comps))[ci].astype(float)) for _ in range(B)]
        bt = []
        for _ in range(B):
            kc = np.bincount(rng.integers(0, len(comps), len(comps)), minlength=len(comps)); kp = np.bincount(rng.integers(0, len(pls), len(pls)), minlength=len(pls))
            w = (kc[ci] * kp[pa] * np.where(pa == pb, 1, kp[pb])).astype(float)   # a one-plate pair counts its plate once
            if w.sum() > 0:
                bt.append(stats(w))
        iv = lambda bs, k: [round(float(q), 4) for q in np.nanpercentile([b[k] for b in bs], [2.5, 97.5])]
        e = dict(pairs=len(pis), compounds=len(comps), plates=len(pls), records=len(rs))
        for k in point:
            e[k] = round(float(point[k]), 4); e[k + "_ci_compounds"] = iv(bc, k); e[k + "_ci_compounds_plates"] = iv(bt, k)
        pc = {}
        for c in comps:
            m = ci == comps.index(c)
            pc[c] = dict(pairs=len({r["pi"] for r, mm in zip(rs, m) if mm}), records=int(m.sum()), well_share=round(float(1 - A["Q"][m].sum() / A["W"][m].sum()), 4),
                         advantage=round(float(A["adv"][m].mean()), 4))
        ws = [v["well_share"] for v in pc.values()]; ad = [v["advantage"] for v in pc.values()]; kpos = int(sum(x > 0 for x in ad))
        e.update(well_share_compound_mean=round(float(np.mean(ws)), 4), well_share_compound_median=round(float(np.median(ws)), 4),
                 advantage_compound_mean=round(float(np.mean(ad)), 4), compounds_positive=kpos, sign_p=sign_p(kpos, len(ad)), per_compound=pc)
        return e
    for name, f in sets.items():
        rs = [r for r in recs if f(pairs[r["pi"]])]
        res[name] = block(rs); print(name, json.dumps({k: v for k, v in res[name].items() if k != "per_compound"}), flush=True)
        if any(r["comp"] == "trametinib" for r in rs):
            rt = [r for r in rs if r["comp"] != "trametinib"]
            res[name + " without trametinib"] = block(rt); print(name, "w/o trametinib", json.dumps({k: v for k, v in res[name + " without trametinib"].items() if k != "per_compound"}), flush=True)
    json.dump(res, open(OUT + "stats99.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
