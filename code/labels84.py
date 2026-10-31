"""Label classes of the verified pairs, saved with their wells, and per-plate counts for the control-well pairs
(samewell/notes.md §84; A100). Classification as check83.py part A: expression (line-mean D' signature, common
genes) and viability pattern (each line's viability log2 odds against the plate's DMSO wells); a pair is 'confirmed'
if both directions reach the 90th percentile among the other treated wells of the partner's plate (plate 13
excluded) in either readout; 'inert' if both wells' dvia are below the median dvia of treated wells and it is not
confirmed; 'discordant' if one well's dvia is above the 80th percentile and the other's below the median;
otherwise 'unresolved'. Control pairs (H11 - H12, plate 13 excluded): per plate, median over lines of the
same-pair and other-plates predictions; number of plates where same > other.
Output $WORKSPACE/rt/samewell/labels84.json"""
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
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    M = np.stack([Dp[fin[:, j], j][:, G0].mean(0) if fin[:, j].any() else np.zeros(len(G0)) for j in range(J)]).astype(np.float64)
    M -= M.mean(1, keepdims=True); Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    idx = json.load(open(V.AGG + "index.json")); cnt = np.load(V.AGG + "counts.npy"); Ps = len(idx["samples"]); nl = len(idx["lines"])
    cells = cnt[:, 0].reshape(nl, Ps, 2).sum(2); sm = pq.read_table(V.META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    spos = {s: k for k, s in enumerate(idx["samples"])}; dmso = collections.defaultdict(list)
    for k, s in enumerate(idx["samples"]):
        if info[s][0].startswith("DMSO"):
            dmso[info[s][1]].append(k)
    def via(s):
        k = spos[s]; p = info[s][1]; nD = cells[:, dmso[p]].sum(1); ND = nD.sum(); n_ = cells[:, k]; N = n_.sum()
        return np.log2((n_ + 0.5) / (N - n_ + 0.5)) - np.log2((nD + 0.5) / (ND - nD + 0.5))
    Vv = np.stack([via(str(s)) for s in d["sample"]]); Vv -= Vv.mean(1, keepdims=True); Vn = Vv / np.maximum(np.linalg.norm(Vv, axis=1, keepdims=True), 1e-12)
    dv = V.dvia_all(); dvw = np.array([dv[str(s)] for s in d["sample"]]); med_dv = np.median(dvw[plate != "plate13"]); q80 = np.quantile(dvw[plate != "plate13"], 0.8)
    def pct(Z, u, v):
        ws = np.where((plate == plate[v]) & (np.arange(J) != v) & (np.arange(J) != u) & (plate != "plate13"))[0]
        return float(np.mean(Z[ws] @ Z[u] < Z[u] @ Z[v]))
    out = []
    for p in V.pairs(d, dv):
        a, b = p["a"], p["b"]
        if (fin[:, a] & fin[:, b]).sum() < 10:
            continue
        pe = [pct(Mn, a, b), pct(Mn, b, a)]; pv = [pct(Vn, a, b), pct(Vn, b, a)]
        conf = min(pe) >= 0.9 or min(pv) >= 0.9
        cls = "confirmed" if conf else ("inert" if max(dvw[a], dvw[b]) < med_dv else ("discordant" if max(dvw[a], dvw[b]) > q80 and min(dvw[a], dvw[b]) < med_dv else "unresolved"))
        out.append(dict(a=a, b=b, group=p["group"], comp=p["comp"], salt=p["salt"], cls=cls))
    res = {"rules": __doc__, "pairs": out, "counts": {"%s|%s" % (g, c): n for (g, c), n in collections.Counter((x["group"], x["cls"]) for x in out).items()}}
    print(json.dumps(res["counts"]), flush=True)
    prof, plates = sw_v1a.control_profiles(d); Rc = {}
    for pl in plates:
        if pl == "plate13":
            continue
        for i in range(L):
            w = prof[(i, pl)]
            if len(w) == 2 and w[0] is not None and w[1] is not None:
                Rc[(i, pl)] = (w[0] - w[1]).astype(np.float64)
    ps = [pl for pl in plates if pl != "plate13"]; per = {}
    for pl in ps:
        have = [i for i in range(L) if (i, pl) in Rc]; sm_, ot = [], []
        for t in have:
            Gt = np.where(base[t] >= 0.2)[0]
            sm_.append(C.pear(np.mean([Rc[(i, pl)] for i in have if i != t], 0)[Gt], Rc[(t, pl)][Gt]))
            ot.append(float(np.mean([C.pear(np.mean([Rc[(i, q)] for i in range(L) if (i, q) in Rc and i != t], 0)[Gt], Rc[(t, pl)][Gt]) for q in ps if q != pl])))
        per[pl] = [round(float(np.median(sm_)), 3), round(float(np.median(ot)), 3)]
    res["control_per_plate"] = per; res["control_plates_same_gt_other"] = int(sum(v[0] > v[1] for v in per.values()))
    print("control", json.dumps(per), res["control_plates_same_gt_other"], flush=True)
    json.dump(res, open(OUT + "labels84.json", "w"), indent=1, default=int); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
