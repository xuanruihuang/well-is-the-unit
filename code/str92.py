"""Can a difference in drug strength between the two wells explain the same-well advantage and the well share?
(samewell/notes.md §93; A100). Review point: if a drug acts more strongly in one well than in the other (dose, batch,
cell number) and there is no well shift, the target line and the other lines are both more reliable in the stronger
well, so the same-well prediction gains and W - Q > 0. Under that alternative, per (pair, line), with half products on
the common genes G0 centred and responses relative to the plate's other treated wells (D', as core82):
  Q = sqrt(Wa Wb) and S_ab = sqrt(S_aa S_bb)
(Wa, Wb: within-well half products; Q: cross-well half product; S_xy: mean product of the line's response in well x with
the other lines' responses in well y). Replacing the arithmetic means of the two wells by geometric means therefore gives
shares that a strength difference alone cannot produce:
  well share beyond strength      = sum(sqrt(Wa Wb) - Q) / sum(sqrt(Wa Wb))
  common part beyond strength     = sum(S* - S_ab) / sum(sqrt(Wa Wb)),  S* = signed sqrt(S_aa S_bb) (smaller of the two
                                    if their signs differ)
and the most that a strength difference can contribute to the usual share is sum(W - sqrt(Wa Wb)) / sum(W), an upper
bound because well shifts also differ between wells.
Correlation advantage: under the alternative the mean advantage of a record is 0.5 rho (sqrt(rel_ta) - sqrt(rel_tb))
(sqrt(rel_ma) - sqrt(rel_mb)), with rel the Spearman-Brown split-half reliability of the target (t) and of the other
lines' mean (m) in each well and |rho| <= 1; the bound with rho = 1 is compared with the observed advantage.
Expression-inert subset: both wells' line-averaged response length (G0, centred) below the 90th percentile of the
length of a plate's control-well difference divided by sqrt(2) (one well's worth of shift and noise), in addition to
no detectable viability effect (labels85 nodvia). Optional argument 'dmso': the control-well reference (cached halves) instead of D' (§93 review). Subsets added for plates 6/14:
verified pairs and pairs with a strong response (length >= 3 in either well). Output $WORKSPACE/rt/samewell/str92.json
(str92_dmso.json). Usage: str92.py [dmso]"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import sw_core as C
import verified as V
from sw_v1a import control_profiles

OUT = (_WS + "/rt/samewell/")
sbf = lambda c: 2 * c / (1 + c)


REF = "dprime"


def main():
    global REF
    import sys
    REF = sys.argv[1] if len(sys.argv) > 1 else "dprime"
    d = C.load(); base = d["base"]; plate = np.array(d["plate"]); L, J = d["D"].shape[:2]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    XA = np.stack([np.asarray(z["XA"][i], np.float32) for i in range(L)]).astype(np.float64)
    XB = np.stack([np.asarray(z["XB"][i], np.float32) for i in range(L)]).astype(np.float64)
    fin = np.isfinite(XA).all(2) & np.isfinite(XB).all(2)
    if REF == "dmso":
        XA[~fin] = 0; XB[~fin] = 0
    for H in ((XA, XB) if REF == "dprime" else ()):
        Hz = np.where(fin[:, :, None], H, 0)
        for p in sorted(set(plate)):
            js = np.where(plate == p)[0]; S = Hz[:, js].sum(1); n = fin[:, js].sum(1)
            for j in js:
                H[:, j] -= (S - Hz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]
    X = np.where(fin[:, :, None], (XA + XB) / 2, 0); Ss = X.sum(0); nf = fin.sum(0)
    # well lengths and the control-difference threshold
    def length(v):
        v = v[G0] - v[G0].mean(); return float(np.linalg.norm(v))
    wl = np.array([length(X[fin[:, j], j].mean(0)) if fin[:, j].any() else np.nan for j in range(J)])
    prof, plates = control_profiles(d); cl = []
    for pl in plates:
        if pl == "plate13":
            continue
        diffs = [w[0] - w[1] for (i, p_), w in prof.items() if p_ == pl and len(w) == 2 and w[0] is not None and w[1] is not None]
        if diffs:
            cl.append(length(np.mean(diffs, 0)) / np.sqrt(2))
    thr = float(np.quantile(cl, 0.9))
    lab = {}
    for fn in ("labels84", "labels85"):
        for x in json.load(open(OUT + fn + ".json"))["pairs"]:
            lab[(fn, min(x["a"], x["b"]), max(x["a"], x["b"]))] = x["cls"]
    pairs = [p for p in V.pairs(d) if fin[:, p["a"]].sum() >= 10]
    for p in pairs:
        a, b = min(p["a"], p["b"]), max(p["a"], p["b"])
        p["cls84"] = lab.get(("labels84", a, b)); p["cls85"] = lab.get(("labels85", a, b))
        p["inert_expr"] = bool(wl[p["a"]] <= thr and wl[p["b"]] <= thr)
    recs = []
    for pi, p in enumerate(pairs):
        a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]; n = len(ls)
        if n < 10:
            continue
        Ya = X[ls, a][:, G0]; Yb = X[ls, b][:, G0]; Ya = Ya - Ya.mean(1, keepdims=True); Yb = Yb - Yb.mean(1, keepdims=True)
        Gaa, Gbb, Gab = Ya @ Ya.T, Yb @ Yb.T, Ya @ Yb.T
        for k, l in enumerate(ls):
            ha = [h[l, a, G0] - h[l, a, G0].mean() for h in (XA, XB)]; hb = [h[l, b, G0] - h[l, b, G0].mean() for h in (XA, XB)]
            Wa, Wb = ha[0] @ ha[1], hb[0] @ hb[1]; Q = 0.25 * sum(u @ v for u in ha for v in hb)
            o = np.arange(n) != k; Saa, Sbb = Gaa[k, o].mean(), Gbb[k, o].mean(); Sab = 0.5 * (Gab[k, o].mean() + Gab[o, k].mean())
            Sst = np.sign(Saa) * np.sqrt(Saa * Sbb) if Saa * Sbb > 0 else min(Saa, Sbb)
            gl = np.where(base[l] >= 0.2)[0]; oth = ls[ls != l]
            def rel(H1, H2, idx):
                return float(np.clip(sbf(C.pear(H1, H2)), 0, 1))
            adv, bnd, sa, ot = [], [], [], []
            rt = {w: rel(XA[l, w, gl], XB[l, w, gl], None) for w in (a, b)}
            rm = {w: rel(XA[oth, w][:, gl].mean(0), XB[oth, w][:, gl].mean(0), None) for w in (a, b)}
            for tg, rp in ((a, b), (b, a)):
                y = X[l, tg, gl]; mo = lambda j: (Ss[j, gl] - X[l, j, gl]) / (nf[j] - 1)
                s_, o_ = C.pear(mo(tg), y), C.pear(mo(rp), y); adv.append(s_ - o_); sa.append(s_); ot.append(o_)
            bound = 0.5 * abs((np.sqrt(rt[a]) - np.sqrt(rt[b])) * (np.sqrt(rm[a]) - np.sqrt(rm[b])))
            recs.append(dict(pi=pi, comp=p["comp"], W=0.5 * (Wa + Wb), Wg=float(np.sqrt(max(Wa, 0) * max(Wb, 0))), Q=Q,
                             sw=0.5 * (Saa + Sbb) - Sab, swg=Sst - Sab, adv=float(np.mean(adv)), bound=float(bound), same=float(np.mean(sa)), other=float(np.mean(ot))))
    res = {"rules": __doc__, "control_threshold": round(thr, 4), "control_lengths": [round(x, 3) for x in cl]}
    subsets = [("diffpos confirmed", lambda p: p["group"] == "diffpos" and p["cls84"] == "confirmed"),
               ("diffpos nodvia", lambda p: p["group"] == "diffpos" and p["cls85"] == "nodvia"),
               ("diffpos nodvia + expression-inert", lambda p: p["group"] == "diffpos" and p["cls85"] == "nodvia" and p["inert_expr"]),
               ("sameplate confirmed", lambda p: p["group"] == "sameplate" and p["cls84"] == "confirmed"),
               ("sameplate nodvia", lambda p: p["group"] == "sameplate" and p["cls85"] == "nodvia"),
               ("sameplate nodvia + expression-inert", lambda p: p["group"] == "sameplate" and p["cls85"] == "nodvia" and p["inert_expr"]),
               ("G614", lambda p: p["group"] == "G614"),
               ("G614 confirmed", lambda p: p["group"] == "G614" and p["cls84"] == "confirmed"),
               ("G614 strong", lambda p: p["group"] == "G614" and max(wl[p["a"]], wl[p["b"]]) >= 3)]
    rng = np.random.default_rng(0)
    for name, f in subsets:
        rs = [r for r in recs if f(pairs[r["pi"]])]
        if not rs:
            res[name] = dict(n_pairs=0); print(name, "none", flush=True); continue
        A = {k: np.array([r[k] for r in rs]) for k in ("W", "Wg", "Q", "sw", "swg", "adv", "bound")}
        comps = sorted({r["comp"] for r in rs}); ci = np.array([comps.index(r["comp"]) for r in rs])
        K = np.stack([np.bincount(rng.integers(0, len(comps), len(comps)), minlength=len(comps)) for _ in range(1000)])[:, ci]
        def ivr(num, den):
            v = (K @ num) / (K @ den); return [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]
        pm = collections.defaultdict(list)
        for r in rs:
            pm[r["pi"]].append((r["same"], r["other"]))
        e = dict(n_pairs=len({r["pi"] for r in rs}), n_comp=len(comps),
                 same=round(float(np.median([np.median([x[0] for x in v]) for v in pm.values()])), 4),
                 other=round(float(np.median([np.median([x[1] for x in v]) for v in pm.values()])), 4),
                 drug_share=round(float(A["Q"].sum() / A["W"].sum()), 4), well_share=round(float(1 - A["Q"].sum() / A["W"].sum()), 4),
                 common_share=round(float(A["sw"].sum() / A["W"].sum()), 4),
                 strength_max_contribution=round(float((A["W"] - A["Wg"]).sum() / A["W"].sum()), 4),
                 well_share_beyond_strength=round(float(1 - A["Q"].sum() / A["Wg"].sum()), 4), well_share_beyond_strength_ci=ivr(A["Wg"] - A["Q"], A["Wg"]),
                 common_beyond_strength=round(float(A["swg"].sum() / A["Wg"].sum()), 4), common_beyond_strength_ci=ivr(A["swg"], A["Wg"]),
                 advantage=round(float(A["adv"].mean()), 4), strength_bound_on_advantage=round(float(A["bound"].mean()), 4),
                 advantage_minus_bound=round(float((A["adv"] - A["bound"]).mean()), 4))
        if "inert" in name:
            e["compounds"] = comps
        res[name] = e; print(name, json.dumps(e), flush=True)
    res["reference"] = REF
    json.dump(res, open(OUT + ("str92.json" if REF == "dprime" else "str92_dmso.json"), "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
