"""Third-review checks (samewell/notes.md §83; A100).
A Label verification of the analysed pairs (verified.py, the exact set of core82), by two readouts:
  expression  line-mean D' signature (common genes G0, centred): Pearson(a, b) and its percentile among the other
              treated wells of b's plate (and b against a's plate); both directions
  viability   each line's viability log2 odds against the plate's DMSO wells: Pearson(a, b) and its percentile, as above
  A pair is 'confirmed' if both directions reach the 90th percentile in either readout; 'inert' if neither well has a
  viability effect (dvia of both wells below the median dvia of all treated wells) and it is not confirmed;
  'discordant' if one well has dvia above the 80th percentile and the other below the median. Per group x class:
  same-well and replicate-well predictions (D', line genes, both wells as target; medians) and the advantage.
B Control-well pairs without plate 13: per plate and line, H11 - H12 (no drug); predicted by the other lines' same
  difference on the same plate and on the other plates; medians (as sw_v1a negctrl).
C Cell-cycle sign-flip test on control-well pairs without plate 13 (from cc_signflip.json): share of plates p < 0.05.
D Ambient RNA on verified pairs (as sw_step2 run_ambient): R^2 of a line's difference between the wells explained by
  (pooled baseline - line baseline), by the wells' mean difference, by both; per group.
E Plate-13 best matches (plate13_81.json): distribution of sample-number offsets, and r_best by original strength.
Output $WORKSPACE/rt/samewell/check83.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
import verified as V
import sw_v1a

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
    M -= M.mean(1, keepdims=True); Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    # viability vectors per cached well
    idx = json.load(open(V.AGG + "index.json")); cnt = np.load(V.AGG + "counts.npy"); Ps = len(idx["samples"]); nl = len(idx["lines"])
    cells = cnt[:, 0].reshape(nl, Ps, 2).sum(2); sm = pq.read_table(V.META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    spos = {s: k for k, s in enumerate(idx["samples"])}; dmso = collections.defaultdict(list)
    for k, s in enumerate(idx["samples"]):
        if info[s][0].startswith("DMSO"):
            dmso[info[s][1]].append(k)
    def via(s):
        k = spos[s]; p = info[s][1]; nD = cells[:, dmso[p]].sum(1); ND = nD.sum(); n_ = cells[:, k]; N = n_.sum()
        return np.log2((n_ + 0.5) / (N - n_ + 0.5)) - np.log2((nD + 0.5) / (ND - nD + 0.5))
    Vv = np.stack([via(str(s)) for s in d["sample"]]); Vv = Vv - Vv.mean(1, keepdims=True); Vn = Vv / np.maximum(np.linalg.norm(Vv, axis=1, keepdims=True), 1e-12)
    dv = V.dvia_all(); dvw = np.array([dv[str(s)] for s in d["sample"]]); med_dv = np.median(dvw[plate != "plate13"]); q80 = np.quantile(dvw[plate != "plate13"], 0.8)
    P = [p for p in V.pairs(d, dv) if fin[:, p["a"]].sum() >= 10]
    def pct(Z, u, v):
        ws = np.where((plate == plate[v]) & (np.arange(J) != v) & (np.arange(J) != u) & (plate != "plate13"))[0]
        return float(np.mean(Z[ws] @ Z[u] < Z[u] @ Z[v]))
    res = {"rules": __doc__}; rows = []
    VAR = Dp; Ss = VAR.sum(0, dtype=np.float64); nf = fin.sum(0)
    for pi, p in enumerate(P):
        a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]
        if len(ls) < 10:
            continue
        pe = [pct(Mn, a, b), pct(Mn, b, a)]; pv = [pct(Vn, a, b), pct(Vn, b, a)]
        conf = min(pe) >= 0.9 or min(pv) >= 0.9
        cls = "confirmed" if conf else ("inert" if max(dvw[a], dvw[b]) < med_dv else ("discordant" if max(dvw[a], dvw[b]) > q80 and min(dvw[a], dvw[b]) < med_dv else "unresolved"))
        same, other = [], []
        for t in ls:
            Gt = np.where(base[t] >= 0.2)[0]
            for tg, rp in ((a, b), (b, a)):
                y = VAR[t, tg, Gt].astype(np.float64); mo = lambda j: (Ss[j, Gt] - VAR[t, j, Gt]) / (nf[j] - 1)
                same.append(C.pear(mo(tg), y)); other.append(C.pear(mo(rp), y))
        rows.append(dict(group=p["group"], comp=p["comp"], salt=p["salt"], cls=cls, expr_pct=pe, via_pct=pv, expr_r=float(Mn[a] @ Mn[b]), via_r=float(Vn[a] @ Vn[b]),
                         dvia=[float(dvw[a]), float(dvw[b])], same=float(np.nanmedian(same)), other=float(np.nanmedian(other)), adv=float(np.nanmean(np.array(same) - np.array(other)))))
    res["A"] = {}
    for g in ("diffpos", "sameplate", "G614", "Ada"):
        for sub, f in (("all", lambda r: True), ("salt", lambda r: r["salt"]), ("exact", lambda r: not r["salt"]), ("confirmed", lambda r: r["cls"] == "confirmed"),
                       ("inert", lambda r: r["cls"] == "inert"), ("discordant", lambda r: r["cls"] == "discordant"), ("unresolved", lambda r: r["cls"] == "unresolved"),
                       ("salt confirmed", lambda r: r["salt"] and r["cls"] == "confirmed")):
            rs = [r for r in rows if r["group"] == g and f(r)]
            if not rs:
                continue
            e = dict(n_pairs=len(rs), n_comp=len({r["comp"] for r in rs}), same=round(float(np.median([r["same"] for r in rs])), 4), other=round(float(np.median([r["other"] for r in rs])), 4),
                     adv_mean=round(float(np.mean([r["adv"] for r in rs])), 4), expr_r=round(float(np.median([r["expr_r"] for r in rs])), 4), via_r=round(float(np.median([r["via_r"] for r in rs])), 4),
                     expr_pct=round(float(np.median([np.mean(r["expr_pct"]) for r in rs])), 4), via_pct=round(float(np.median([np.mean(r["via_pct"]) for r in rs])), 4))
            res["A"]["%s | %s" % (g, sub)] = e; print("A", g, sub, json.dumps(e), flush=True)
    res["A_rows"] = rows
    # B control pairs without plate 13
    prof, plates = sw_v1a.control_profiles(d); Rc = {}
    for pl in plates:
        if pl == "plate13":
            continue
        for i in range(L):
            w = prof[(i, pl)]
            if len(w) == 2 and w[0] is not None and w[1] is not None:
                Rc[(i, pl)] = (w[0] - w[1]).astype(np.float64)
    ps = [pl for pl in plates if pl != "plate13"]; same_c, oth_c = [], []
    for pl in ps:
        have = [i for i in range(L) if (i, pl) in Rc]
        for t in have:
            Gt = np.where(base[t] >= 0.2)[0]
            same_c.append(C.pear(np.mean([Rc[(i, pl)] for i in have if i != t], 0)[Gt], Rc[(t, pl)][Gt]))
            oo = []
            for q in ps:
                if q == pl:
                    continue
                hq = [i for i in range(L) if (i, q) in Rc and i != t]
                if len(hq) >= 10:
                    oo.append(C.pear(np.mean([Rc[(i, q)] for i in hq], 0)[Gt], Rc[(t, pl)][Gt]))
            oth_c.append(float(np.mean(oo)))
    res["B"] = dict(n=len(same_c), plates=len(ps), same=round(float(np.median(same_c)), 4), other=round(float(np.median(oth_c)), 4),
                    excess_mean=round(float(np.mean(np.array(same_c) - np.array(oth_c))), 4))
    print("B", json.dumps(res["B"]), flush=True)
    cc = json.load(open(OUT + "cc_signflip.json"))["pseudo"]
    pv = [v["p"] for k, v in cc.items() if k != "plate13"]
    res["C"] = dict(n_plates=len(pv), n_p_lt_005=int(sum(x < 0.05 for x in pv)), n_p_lt_001=int(sum(x < 0.01 for x in pv)), plate13_p=cc.get("plate13", {}).get("p"))
    print("C", json.dumps(res["C"]), flush=True)
    # D ambient
    pooled = d["pooled"]; Mw = np.stack([D[fin[:, j], j].mean(0) if fin[:, j].any() else np.zeros(D.shape[2]) for j in range(J)])
    res["D"] = {}
    for g in ("diffpos", "sameplate", "G614"):
        ra, rs_, rb = [], [], []
        for p in P:
            if p["group"] != g:
                continue
            a, b = p["a"], p["b"]; dd = Mw[a] - Mw[b]
            for t in np.where(fin[:, a] & fin[:, b])[0]:
                dl = (D[t, a] - D[t, b]).astype(np.float64); amb = pooled - base[t]
                def r2(*cols):
                    X = np.c_[np.ones(len(dl)), np.column_stack(cols)]; e = dl - X @ np.linalg.lstsq(X, dl, rcond=None)[0]
                    return 1 - (e @ e) / ((dl - dl.mean()) @ (dl - dl.mean()))
                ra.append(r2(amb)); rs_.append(r2(dd)); rb.append(r2(amb, dd))
        res["D"][g] = dict(n=len(ra), R2_ambient=round(float(np.median(ra)), 4), R2_shared=round(float(np.median(rs_)), 4), R2_both=round(float(np.median(rb)), 4))
        print("D", g, json.dumps(res["D"][g]), flush=True)
    # E plate-13 offsets
    p13 = json.load(open(OUT + "plate13_81.json"))["rows"]
    off = collections.Counter(r["offset_samples"] for r in p13); best = collections.Counter(r["best"] for r in p13)
    strong = [r for r in p13 if r["amp_orig"] >= 3]; weak = [r for r in p13 if r["amp_orig"] < 2.2]
    res["E"] = dict(n=len(p13), top_offsets=off.most_common(6), top_best_wells=best.most_common(6),
                    r_best_strong_median=round(float(np.median([r["r_best"] for r in strong])), 4), r_best_weak_median=round(float(np.median([r["r_best"] for r in weak])), 4),
                    r_labelled_strong=[r["r_labelled"] for r in strong])
    print("E", json.dumps(res["E"]), flush=True)
    json.dump(res, open(OUT + "check83.json", "w"), indent=1, default=float); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
