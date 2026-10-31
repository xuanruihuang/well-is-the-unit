"""Plate 13 restricted to strongly acting drugs (samewell/notes.md §92; A100). Review point: the partner percentile cannot
tell a wrong label from a drug without effect, since weak drugs rank near chance even when correctly labelled (the
one-plate replicates, mostly weak drugs, also reach a median of 44). Here only pairs in which at least one well has a
strong line-averaged response are compared.
Signature of a well (as check83 A): line-mean D' response over genes with median baseline >= 0.2, centred; response
length = its Euclidean norm; partner percentile = rank of Pearson(a, b) among Pearson(a, w) for all other treated wells w
of b's plate, both directions. Pairs: verified.pairs (G614; diffpos split into same-name and salt-name; sameplate) and
the 45 plate-13 pairs (sw_core.groups G13: same exact name and dose, one well on plate 13, adagrasib apart). Strong: length >= 3 in
either well (also 2.5 and 4). Reported per group: pairs, median percentile (mean of directions), pairs whose partner is in
the top 10% in both directions; for plate 13 also the plate-13 well's length when the other well's length >= 3.
Output $WORKSPACE/rt/samewell/p13strong92.json. Usage: p13strong92.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import sw_core as C
import verified as V

OUT = (_WS + "/rt/samewell/")


def main():
    d = C.load(); D = d["D"]; base = d["base"]; plate = np.array(d["plate"]); L, J = D.shape[:2]
    fin = np.isfinite(D).all(2); Dz = np.where(fin[:, :, None], D, 0).astype(np.float32); Dp = np.zeros_like(Dz)
    for p in sorted(set(plate)):
        js = np.where(plate == p)[0]; S = Dz[:, js].sum(1, dtype=np.float64); n = fin[:, js].sum(1)
        for j in js:
            Dp[:, j] = Dz[:, j] - ((S - Dz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]).astype(np.float32)
    Dp[~fin] = 0
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    M = np.stack([Dp[fin[:, j], j][:, G0].mean(0) if fin[:, j].any() else np.zeros(len(G0)) for j in range(J)]).astype(np.float64)
    M -= M.mean(1, keepdims=True); amp = np.linalg.norm(M, axis=1); Mn = M / np.maximum(amp[:, None], 1e-12)
    def pct(u, v):
        ws = np.where((plate == plate[v]) & (np.arange(J) != v) & (np.arange(J) != u))[0]
        return float(np.mean(Mn[ws] @ Mn[u] < Mn[u] @ Mn[v]))
    P = []
    for p in V.pairs(d):
        g = p["group"]
        if g == "diffpos":
            g = "diffpos salt" if p["salt"] else "diffpos same name"
        P.append((g, p["a"], p["b"], p["cond"]))
    for c, a, b, g in C.groups(d)[0]:                         # exact-name pairs with one well on plate 13 (adagrasib apart)
        if g == "G13":
            a, b = (a, b) if plate[b] == "plate13" else (b, a)
            P.append(("plate13", a, b, c))
    rows = [dict(group=g, cond=c, a=int(a), b=int(b), amp=[float(amp[a]), float(amp[b])], pct=[pct(a, b), pct(b, a)]) for g, a, b, c in P if fin[:, a].sum() >= 10 and fin[:, b].sum() >= 10]
    res = {"rules": __doc__, "rows": rows}
    for thr in (2.5, 3.0, 4.0):
        for g in ("G614", "diffpos same name", "diffpos salt", "sameplate", "Ada", "plate13"):
            rs = [r for r in rows if r["group"] == g and max(r["amp"]) >= thr]
            if not rs:
                continue
            e = dict(n=len(rs), median_pct=round(float(np.median([np.mean(r["pct"]) for r in rs])), 4), top10_both=int(sum(min(r["pct"]) >= 0.9 for r in rs)))
            res["%s | strong %.1f" % (g, thr)] = e; print(g, thr, json.dumps(e), flush=True)
    for g in ("G614", "diffpos same name", "diffpos salt", "sameplate", "Ada", "plate13"):
        rs = [r for r in rows if r["group"] == g]
        res["%s | all" % g] = dict(n=len(rs), median_pct=round(float(np.median([np.mean(r["pct"]) for r in rs])), 4), top10_both=int(sum(min(r["pct"]) >= 0.9 for r in rs)))
        print(g, "all", json.dumps(res["%s | all" % g]), flush=True)
    res["strong rows, sameplate and diffpos salt"] = [r for r in rows if r["group"] in ("sameplate", "diffpos salt") and max(r["amp"]) >= 3]
    for r in res["strong rows, sameplate and diffpos salt"]:
        print("strong", r["group"], r["cond"], [round(x, 2) for x in r["amp"]], [round(x, 2) for x in r["pct"]], flush=True)
    # Fisher exact tests at length >= 3: each group against the same-name pairs across plates outside plate 13
    # (plates 6/14 and plates 1-3 against 7-12), whose labels are expected to be right
    from scipy.stats import fisher_exact
    st = lambda g: [r for r in rows if r["group"] in g and max(r["amp"]) >= 3]
    ref = st(("G614", "diffpos same name")); kr = sum(min(r["pct"]) >= 0.9 for r in ref)
    for g in ("plate13", "sameplate", "diffpos salt"):
        rs = st((g,)); k = sum(min(r["pct"]) >= 0.9 for r in rs)
        e = dict(recognised=[int(k), len(rs)], reference=[int(kr), len(ref)], p_two_sided=float(fisher_exact([[k, len(rs) - k], [kr, len(ref) - kr]])[1]))
        res["fisher strong 3.0 | %s" % g] = e; print("fisher", g, json.dumps(e), flush=True)
    sp = [r for r in rows if r["group"] == "sameplate" and max(r["amp"]) >= 3]
    res["sameplate strong exact names"] = sorted({r["cond"] for r in sp})
    s13 = [r for r in rows if r["group"] == "plate13" and r["amp"][0] >= 3]
    res["plate13 original >= 3"] = dict(n=len(s13), plate13_length=[round(r["amp"][1], 2) for r in s13], original_length=[round(r["amp"][0], 2) for r in s13],
                                        plate13_median_length_all=round(float(np.median(amp[plate == "plate13"])), 2))
    print("plate13 original>=3", json.dumps(res["plate13 original >= 3"]), flush=True)
    json.dump(res, open(OUT + "p13strong92.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
