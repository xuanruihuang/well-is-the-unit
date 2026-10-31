"""Verification rule: false-positive rate, cross-readout verification, plates-6/14 cross-salt pairs (samewell/notes.md
§99, independent review 10-29 items A1, A2, A8; A100).
Responses: treated-well reference D' built separately for each half of the cells (as ref88.py); X = mean of halves.
Signatures (as labels84.py): expression = line-mean response over the common genes G0, centred; viability = each
line's viability log2 odds against the plate's DMSO wells, centred. pct(u, v) = share of the other treated wells of
v's plate (plate 13 excluded) whose similarity to u is below sim(u, v). A pair passes a readout if pct(a, b) and
pct(b, a) are both >= 0.9; 'either' = expression or viability (the paper's rule).
A1 Decoys: two treated wells of different compounds (salt-stripped names), plate 13 excluded, 5,000 per class:
   one plate (different positions; also adjacent = Chebyshev distance 1, far = distance >= 4), different plates (not
   the pair 6/14, different positions), plates 6 and 14 (different positions). Pass rates per readout, also for decoys
   whose two wells both lack a viability effect (dvia < 0.289). Expected chance passes if every label of a group
   were wrong = pass rate x pairs in the group.
A2 Verification by viability only, by expression only, and cross-fitted expression (signature from one half of the
   cells, scores from the other half; both directions). Scores as in the paper: per record (pair x line), the mean
   over the two wells as target of Pearson(mean of the other lines in the well, target line) on the line's genes
   (base >= 0.2), for the same well and for the replicate well.
A8 Plates 6/14: same-position pairs only; cross-salt pairs at different positions reported apart.
Summary per set: pairs, compounds, record medians, record-mean difference, compound-equal mean difference, sign
test over compounds (two-sided binomial).
Output $WORKSPACE/rt/samewell/ver99.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections, math
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
import verified as V
from sw_step2 import position

OUT = (_WS + "/rt/samewell/"); THR = 0.289


def sign_p(k, n):
    if n == 0:
        return None
    t = min(k, n - k); return min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def main():
    d = C.load(); base = d["base"]; plate = np.array(d["plate"]); L, J = d["D"].shape[:2]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    XA0 = np.stack([np.asarray(z["XA"][i], np.float32) for i in range(L)]).astype(np.float64)
    XB0 = np.stack([np.asarray(z["XB"][i], np.float32) for i in range(L)]).astype(np.float64)
    fin = np.isfinite(XA0).all(2) & np.isfinite(XB0).all(2)
    hs = []
    for H0 in (XA0, XB0):
        Hz = np.where(fin[:, :, None], H0, 0); H = np.zeros_like(Hz)
        for p in sorted(set(plate)):
            js = np.where(plate == p)[0]; S = Hz[:, js].sum(1); n = fin[:, js].sum(1)
            for j in js:
                H[:, j] = Hz[:, j] - (S - Hz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]
        hs.append(H)
    del XA0, XB0
    XA, XB = hs; X = (XA + XB) / 2

    def esig(H):
        M = np.stack([H[fin[:, j], j][:, G0].mean(0) if fin[:, j].any() else np.zeros(len(G0)) for j in range(J)])
        M -= M.mean(1, keepdims=True); return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    ME, MEA, MEB = esig(X), esig(XA), esig(XB)
    idx = json.load(open(V.AGG + "index.json")); cnt = np.load(V.AGG + "counts.npy"); Ps = len(idx["samples"]); nl = len(idx["lines"])
    cells = cnt[:, 0].reshape(nl, Ps, 2).sum(2); sm = pq.read_table(V.META).to_pydict()
    info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    spos = {s: k for k, s in enumerate(idx["samples"])}; dmso = collections.defaultdict(list)
    for k, s in enumerate(idx["samples"]):
        if info[s][0].startswith("DMSO"):
            dmso[info[s][1]].append(k)
    def via(s):
        k = spos[s]; p = info[s][1]; nD = cells[:, dmso[p]].sum(1); ND = nD.sum(); n_ = cells[:, k]; N = n_.sum()
        return np.log2((n_ + 0.5) / (N - n_ + 0.5)) - np.log2((nD + 0.5) / (ND - nD + 0.5))
    Vv = np.stack([via(str(s)) for s in d["sample"]]); Vv -= Vv.mean(1, keepdims=True); MV = Vv / np.maximum(np.linalg.norm(Vv, axis=1, keepdims=True), 1e-12)
    dv = V.dvia_all(); dvw = np.array([dv[str(s)] for s in d["sample"]])
    ok = plate != "plate13"; peers = {p: np.where((plate == p) & ok)[0] for p in set(plate)}

    def pct(Z, u, v):
        ws = peers[plate[v]]; ws = ws[(ws != v) & (ws != u)]; return float(np.mean(Z[ws] @ Z[u] < Z[u] @ Z[v]))
    def passes(Z, a, b):
        return min(pct(Z, a, b), pct(Z, b, a)) >= 0.9

    pos, row, col = position(d); comp = np.array([V.norm(x) for x in d["drug"]])
    treated = np.array([not str(x).strip().startswith("DMSO") for x in d["drug"]]) & ok & (fin.sum(0) >= 10)
    res = {"rules": __doc__}
    # A1 decoys
    rng = np.random.default_rng(0); tj = np.where(treated)[0]; dec = collections.defaultdict(list)
    def draw(kind):
        while True:
            a, b = rng.choice(tj, 2, replace=False)
            if comp[a] == comp[b] or pos[a] == pos[b] and plate[a] == plate[b]:
                continue
            if kind == "one plate" and plate[a] == plate[b]:
                return a, b
            if kind == "plates 6/14" and {plate[a], plate[b]} == {"plate6", "plate14"} and pos[a] != pos[b]:
                return a, b
            if kind == "different plates" and plate[a] != plate[b] and {plate[a], plate[b]} != {"plate6", "plate14"} and pos[a] != pos[b]:
                return a, b
    for kind in ("one plate", "different plates", "plates 6/14"):
        for _ in range(5000):
            a, b = draw(kind) if kind != "plates 6/14" else draw(kind)
            e, v = passes(ME, a, b), passes(MV, a, b)
            dist = max(abs(row[a] - row[b]), abs(col[a] - col[b])) if plate[a] == plate[b] else None
            dec[kind].append(dict(e=e, v=v, nod=bool(max(dvw[a], dvw[b]) < THR), dist=dist))
    res["A1"] = {}
    for kind, rs in dec.items():
        subs = {"all": rs, "both without viability effect": [r for r in rs if r["nod"]]}
        if kind == "one plate":
            subs["adjacent"] = [r for r in rs if r["dist"] == 1]; subs["far (>= 4)"] = [r for r in rs if r["dist"] >= 4]
        for sn, s in subs.items():
            if s:
                res["A1"]["%s | %s" % (kind, sn)] = dict(n=len(s), expression=round(float(np.mean([r["e"] for r in s])), 4),
                                                         viability=round(float(np.mean([r["v"] for r in s])), 4), either=round(float(np.mean([r["e"] or r["v"] for r in s])), 4))
    print("A1", json.dumps(res["A1"]), flush=True)

    # pairs and their verification under each rule
    P = [p for p in V.pairs(d) if (fin[:, p["a"]] & fin[:, p["b"]]).sum() >= 10 and p["group"] != "Ada"]
    for p in P:
        a, b = p["a"], p["b"]
        p["E"], p["Vi"] = passes(ME, a, b), passes(MV, a, b); p["EA"], p["EB"] = passes(MEA, a, b), passes(MEB, a, b)
        p["nod"] = bool(max(dvw[a], dvw[b]) < THR); p["dist"] = int(max(abs(row[a] - row[b]), abs(col[a] - col[b]))) if plate[a] == plate[b] else None
        if p["group"] == "G614":
            p["group"] = "G614" if pos[a] == pos[b] else "G614 cross-salt different positions"
    res["pairs_by_group"] = dict(collections.Counter(p["group"] for p in P))
    res["sameplate_distances"] = dict(collections.Counter(p["dist"] for p in P if p["group"] == "sameplate"))
    res["expected_chance_passes"] = {}
    for g, kind in (("sameplate", "one plate"), ("diffpos", "different plates"), ("G614", "plates 6/14")):
        n = sum(p["group"] == g for p in P); r = res["A1"]["%s | all" % kind]
        res["expected_chance_passes"][g] = dict(pairs=n, observed_either=int(sum(p["group"] == g and (p["E"] or p["Vi"]) for p in P)),
                                               observed_expression=int(sum(p["group"] == g and p["E"] for p in P)), observed_viability=int(sum(p["group"] == g and p["Vi"] for p in P)),
                                               expected_either_if_all_wrong=round(r["either"] * n, 2), expected_expression=round(r["expression"] * n, 2), expected_viability=round(r["viability"] * n, 2))
    print("expected", json.dumps(res["expected_chance_passes"]), flush=True)

    # A2 scores
    def scores(Xs, ps):
        Ss = Xs.sum(0); nf = fin.sum(0); out = []
        for p in ps:
            a, b = p["a"], p["b"]
            for t in np.where(fin[:, a] & fin[:, b])[0]:
                Gt = np.where(base[t] >= 0.2)[0]; sa, oa = [], []
                for tg, rp in ((a, b), (b, a)):
                    y = Xs[t, tg, Gt]; mo = lambda j: (Ss[j, Gt] - Xs[t, j, Gt]) / (nf[j] - 1)
                    sa.append(C.pear(mo(tg), y)); oa.append(C.pear(mo(rp), y))
                out.append((p["comp"], float(np.mean(sa)), float(np.mean(oa))))
        return out
    def summ(rec, ps):
        if not rec:
            return dict(pairs=len(ps), compounds=len({p["comp"] for p in ps}))
        by = collections.defaultdict(list)
        for c, s, o in rec:
            by[c].append(s - o)
        cm = [np.mean(v) for v in by.values()]; k = int(sum(x > 0 for x in cm))
        return dict(pairs=len(ps), compounds=len(by), same=round(float(np.median([r[1] for r in rec])), 4), other=round(float(np.median([r[2] for r in rec])), 4),
                    diff_record_mean=round(float(np.mean([r[1] - r[2] for r in rec])), 4), diff_compound_mean=round(float(np.mean(cm)), 4),
                    compounds_positive=k, sign_p=sign_p(k, len(cm)))
    rules = {"paper rule (either readout)": lambda p: p["E"] or p["Vi"], "viability only": lambda p: p["Vi"], "expression only": lambda p: p["E"],
             "viability, not expression": lambda p: p["Vi"] and not p["E"], "all pairs": lambda p: True}
    res["A2"] = {}
    for g in ("diffpos", "sameplate", "G614", "G614 cross-salt different positions"):
        for rn, f in rules.items():
            ps = [p for p in P if p["group"] == g and f(p)]
            res["A2"]["%s | %s" % (g, rn)] = summ(scores(X, ps), ps)
            print("A2", g, rn, json.dumps(res["A2"]["%s | %s" % (g, rn)]), flush=True)
        # cross-fitting: verify on one half (expression signature of that half, or viability), score on the other half
        for h, (key, Xo) in enumerate((("EA", XB), ("EB", XA))):
            ps = [p for p in P if p["group"] == g and (p[key] or p["Vi"])]; psa = [p for p in P if p["group"] == g]
            res["A2"]["%s | cross-fit %s, verified" % (g, "AB"[h])] = summ(scores(Xo, ps), ps)
            res["A2"]["%s | cross-fit %s, all pairs (same half)" % (g, "AB"[h])] = summ(scores(Xo, psa), psa)
            print("A2", g, "cross-fit", h, json.dumps(res["A2"]["%s | cross-fit %s, verified" % (g, "AB"[h])]), flush=True)
    res["pairs"] = [dict(a=p["a"], b=p["b"], group=p["group"], comp=p["comp"], salt=p["salt"], E=p["E"], Vi=p["Vi"], EA=p["EA"], EB=p["EB"], nod=p["nod"], dist=p["dist"]) for p in P]
    json.dump(res, open(OUT + "ver99.json", "w"), indent=1, default=int); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
