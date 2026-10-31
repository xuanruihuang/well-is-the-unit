"""Are the replicate wells real replicates, and is cytotoxicity a reliable strength measure? (samewell/notes.md §81
item 8; A100). Prompted by g13check81: the two wells of a plate-13 pair barely agree on cytotoxicity (r 0.06), and a
strong compound (topotecan 5 uM) lost cells in one well and gained them in the other.
1 Label check: line-mean D' signature of each treated well (lines present, common genes G0, centred). For each replicate
  pair (a, b): Pearson(M_a, M_b) and its percentile among Pearson(M_a, M_w) for every other treated well w on b's plate
  (and the same with a and b swapped). Correct labels and an active drug put the partner near the top; the
  percentile is reported against the strength of the signature (|M| of the original well, by tertile).
2 Strength readouts between replicate wells: correlation over pairs of cytotoxicity (total cells against the plate's
  DMSO wells) and of dvia (SD over lines of the viability log2 odds, §78), per group.
Groups: G13, Gother, salt pairs not involving plate 13, plates 6/14. Output $WORKSPACE/rt/samewell/label81.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
from strength import cytotoxicity, AGG, META

OUT = (_WS + "/rt/samewell/")


def main():
    d = C.load(); D = d["D"]; base = d["base"]; plate = np.array(d["plate"]); L, J = D.shape[:2]; cy = cytotoxicity()
    fin = np.isfinite(D).all(2); Dz = np.where(fin[:, :, None], D, 0).astype(np.float32); Dp = np.zeros_like(Dz)
    for p in sorted(set(plate)):
        js = np.where(plate == p)[0]; S = Dz[:, js].sum(1, dtype=np.float64); n = fin[:, js].sum(1)
        for j in js:
            Dp[:, j] = Dz[:, j] - ((S - Dz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]).astype(np.float32)
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    M = np.stack([Dp[fin[:, j], j][:, G0].mean(0) for j in range(J)]).astype(np.float64); M -= M.mean(1, keepdims=True)
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True); amp = np.linalg.norm(M, axis=1)
    # dvia per sample
    idx = json.load(open(AGG + "index.json")); cnt = np.load(AGG + "counts.npy"); Ps = len(idx["samples"]); nl = len(idx["lines"])
    cells = cnt[:, 0].reshape(nl, Ps, 2).sum(2); sm = pq.read_table(META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    spos = {s: k for k, s in enumerate(idx["samples"])}; dmso = collections.defaultdict(list)
    for k, s in enumerate(idx["samples"]):
        if info[s][0].startswith("DMSO"):
            dmso[info[s][1]].append(k)
    def dvia(s):
        k = spos[s]; p = info[s][1]; nD = cells[:, dmso[p]].sum(1); ND = nD.sum(); n_ = cells[:, k]; N = n_.sum()
        return float(np.std(np.log2((n_ + 0.5) / (N - n_ + 0.5)) - np.log2((nD + 0.5) / (ND - nD + 0.5))))
    pairs, _ = C.groups(d); rows = []
    nm = collections.defaultdict(list)
    for j in range(J):
        nm[(C.norm(d["drug"][j]), d["conc"][j])].append(j)
    sets = [(g, a, b) for c, a, b, g in pairs if g in ("G13", "Gother", "G614")]
    for k, js in nm.items():
        for x in range(len(js)):
            for y in range(x + 1, len(js)):
                a, b = js[x], js[y]
                if d["drug"][a] != d["drug"][b] and plate[a] != plate[b] and "plate13" not in (plate[a], plate[b]):
                    sets.append(("salt (no plate 13)", a, b))
    for g, a, b in sets:
        pct = []
        for u, v in ((a, b), (b, a)):
            ws = np.where((plate == plate[v]) & (np.arange(J) != v))[0]
            r_uv = float(Mn[u] @ Mn[v]); null = Mn[ws] @ Mn[u]; pct.append(float(np.mean(null < r_uv)))
        sa, sb = str(d["sample"][a]), str(d["sample"][b])
        rows.append(dict(group=g, drug=d["drug"][a], dose=d["conc"][a], plates=[plate[a], plate[b]], r=float(Mn[a] @ Mn[b]),
                         pct=pct, amp=float(max(amp[a], amp[b])), cyto=[cy[sa], cy[sb]], dvia=[dvia(sa), dvia(sb)]))
    res = {"rules": __doc__}
    allamp = np.array([r["amp"] for r in rows]); q = np.quantile(allamp, [1 / 3, 2 / 3])
    for g in ("G13", "Gother", "salt (no plate 13)", "G614"):
        rs = [r for r in rows if r["group"] == g]
        if not rs:
            continue
        ca = np.array([r["cyto"] for r in rs]); da = np.array([r["dvia"] for r in rs])
        e = dict(n_pairs=len(rs), signature_r_median=round(float(np.median([r["r"] for r in rs])), 3),
                 partner_percentile_median=round(float(np.median([np.mean(r["pct"]) for r in rs])), 3),
                 share_partner_top5pct=round(float(np.mean([min(r["pct"]) >= 0.95 for r in rs])), 3),
                 cyto_corr=round(float(np.corrcoef(ca[:, 0], ca[:, 1])[0, 1]), 3), dvia_corr=round(float(np.corrcoef(da[:, 0], da[:, 1])[0, 1]), 3))
        for t in range(3):
            st = [r for r in rs if int(np.digitize(r["amp"], q)) == t]
            if st:
                e["amp tertile %d" % t] = dict(n=len(st), partner_percentile_median=round(float(np.median([np.mean(r["pct"]) for r in st])), 3),
                                               signature_r_median=round(float(np.median([r["r"] for r in st])), 3))
        res[g] = e; print(g, json.dumps(e), flush=True)
    res["pairs"] = rows
    top = sorted([r for r in rows if r["group"] in ("G13", "salt (no plate 13)", "Gother")], key=lambda r: -r["amp"])[:15]
    for r in top:
        print("strong", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(res, open(OUT + "label81.json", "w"), indent=1, default=float); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
