"""Data for the redrawn figures (samewell/notes.md §110; A100). Heat maps of what two wells do not share, and the
replicate-label similarity matrices.
Profiles (Figs 1 and 3): pseudobulk of raw counts per (line, well, half) (pb/, half = lowest bit of the CRC32 hash of the cell's barcode,
pb_extract, i.e. a random half of the cells),
log1p(1e4 c / total) over the coding genes; a whole well = its two halves summed. Lines as bp91_data (QC failure
<= 0.9); a well counts for a line if it passes QC (>= 50 cells, >= 35,000 coding UMIs) and each half has >= 10 cells.
Genes: median over the kept lines of the line's DMSO profile (mean over plates' control wells) >= 0.2.
Lines are split at random (seed 0) into halves A and B. Per pair of wells the genes kept are the K with the highest and
the K with the lowest mean whole-well difference (well 1 - well 2) over the A lines, ordered by that mean; the figure
shows the B lines, so a stripe shared by the B lines cannot come from the selection.
Per pair and line:
  full     well 1 - well 2 (whole wells)
  between  well 1 half 0 - well 2 half 1 (two wells, one half each)
  within   well 1 half 0 - well 1 half 1 (two halves of one well)
  between and within are both half against half, so they carry the same sampling noise.
Drug-treated wells: each profile minus the line's mean over the other QC-passing treated wells of its plate (D', whole
wells; it cancels in within). Control pairs: the plate's two DMSO wells (plate 13 apart), no reference.
Pairs: control pairs of the 13 plates other than plate 13; verified.pairs groups G614 and diffpos with the labels84 class.
Per pair and B line: Pearson over the kept genes between the line's row and the A-line mean of full (stripe score).
Fig 2: as p13strong92 (cached D' line-mean signatures over genes with median baseline >= 0.2, centred; length = norm);
strong pairs (length >= 3 in either well, >= 10 lines in both) of plates 6/14, same-name and salt-name pairs across plates (diffpos), one-plate pairs
and plate 13 (first well = original, second = plate-13 well); matrix C[i, k] = Pearson(first well of pair i,
second well of pair k); lineup = for each direction, the similarity of the well to every other treated well of the
partner's plate (sorted) and to the partner. Argument "labels": only this part, to fig110_labels.json.
Output $WORKSPACE/rt/samewell/fig110.npz and fig110.json. Usage: fig110.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import sys, json, collections
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
import verified as V
import bp91_data as B

OUT = (_WS + "/rt/samewell/"); PB = OUT + "pb/"; META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")
K = 150


def heat():
    idx = json.load(open(PB + "index.json")); lines, samples = idx["lines"], idx["samples"]; L, P = len(lines), len(samples)
    cnt4 = np.load(PB + "cnt_lsh.npy").reshape(L, P, 2, 4); cnt = cnt4.sum(2)
    Xc = np.load(PB + "counts_lsh.npy", mmap_mode="r")
    sm = pq.read_table(META).to_pydict(); info = {s: (d.strip(), p) for s, d, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    drug = [info[s][0] for s in samples]; plate = [info[s][1] for s in samples]
    isd = np.array([x.startswith("DMSO") for x in drug])
    exists = cnt[:, :, 0] > 0; qc = (cnt[:, :, 0] >= 50) & (cnt[:, :, 1] >= 35000)
    fail = 1 - (qc & exists).sum(1) / np.maximum(exists.sum(1), 1)
    keep_l = [i for i in range(L) if fail[i] <= 0.9 and exists[i].any()]
    ok = qc & (cnt4[:, :, :, 0].min(2) >= 10)
    lg = lambda c: np.log1p(1e4 * c / c.sum())
    def half(i, j, h):
        return np.asarray(Xc[(i * P + j) * 2 + h], np.float64)
    def whole(i, j):
        return lg(half(i, j, 0) + half(i, j, 1))
    plates = sorted({p for p in plate if p != "plate13"}, key=lambda p: int(p[5:]))
    ctrl = {pl: sorted([j for j in range(P) if isd[j] and plate[j] == pl], key=lambda j: int(samples[j][4:])) for pl in plates}
    # genes
    base = []
    for i in keep_l:
        ps = [whole(i, j) for pl in plates for j in ctrl[pl] if qc[i, j]]
        base.append(np.mean(ps, 0))
    G0 = np.where(np.median(base, 0) >= 0.2)[0]; print("genes", len(G0), "lines", len(keep_l), flush=True)
    # D' references (sum over QC-passing treated wells of the plate, whole wells)
    d = B.light_d(); si = {s: k for k, s in enumerate(samples)}
    lab = {(min(x["a"], x["b"]), max(x["a"], x["b"])): x["cls"] for x in json.load(open(OUT + "labels84.json"))["pairs"]}
    pairs = []
    for pl in plates:
        if len(ctrl[pl]) == 2:
            pairs.append(dict(kind="control", plate=pl, a=ctrl[pl][0], b=ctrl[pl][1]))
    for e in V.pairs(d):
        if e["group"] in ("G614", "diffpos"):
            a, b = min(e["a"], e["b"]), max(e["a"], e["b"])
            pairs.append(dict(kind=e["group"], cond=e["cond"], comp=e["comp"], salt=bool(e["salt"]), a=si[str(d["sample"][a])], b=si[str(d["sample"][b])],
                              pa=plate[si[str(d["sample"][a])]], pb=plate[si[str(d["sample"][b])]], cls=lab.get((a, b))))
    need_pl = sorted({plate[p[k]] for p in pairs if p["kind"] != "control" for k in ("a", "b")})
    tsum, tn = {}, {}
    for i in keep_l:
        for pl in need_pl:
            js = [j for j in range(P) if plate[j] == pl and not isd[j] and qc[i, j]]
            if js:
                tsum[(i, pl)] = np.sum([whole(i, j) for j in js], 0); tn[(i, pl)] = len(js)
        print("ref line", lines[i], flush=True)
    rng = np.random.default_rng(0); perm = rng.permutation(len(keep_l)); A = set(np.array(keep_l)[perm[: len(keep_l) // 2]].tolist())
    def ref(i, j):
        if (i, plate[j]) not in tn or tn[(i, plate[j])] < 2:
            return None
        return (tsum[(i, plate[j])] - whole(i, j)) / (tn[(i, plate[j])] - 1)
    arrs = {}; meta = []
    for n, p in enumerate(pairs):
        a, b = p["a"], p["b"]; rows = []
        for i in keep_l:
            if not (ok[i, a] and ok[i, b]):
                continue
            if p["kind"] == "control":
                ra = rb = 0.0
            else:
                ra, rb = ref(i, a), ref(i, b)
                if ra is None or rb is None:
                    continue
            a0, a1, b1 = lg(half(i, a, 0)), lg(half(i, a, 1)), lg(half(i, b, 1))
            full = (whole(i, a) - ra) - (whole(i, b) - rb)
            rows.append((i, full[G0], ((a0 - ra) - (b1 - rb))[G0], (a0 - a1)[G0]))
        if len(rows) < 10:
            continue
        inA = np.array([r[0] in A for r in rows])
        F = np.array([r[1] for r in rows]); Bt = np.array([r[2] for r in rows]); W = np.array([r[3] for r in rows])
        mA = F[inA].mean(0); o = np.argsort(-mA); sel = np.concatenate([o[:K], o[-K:]])
        stripe = lambda M: [float(np.corrcoef(M[k, sel], mA[sel])[0, 1]) for k in np.where(~inA)[0]]
        e = dict(p, n=n, lines=[lines[r[0]] for r in rows], inA=inA.tolist(), genes=[idx["genes"][G0[g]] for g in sel],
                 stripe_full=stripe(F), stripe_between=stripe(Bt), stripe_within=stripe(W))
        for k in ("a", "b"):
            e[k] = int(e[k]); e["sample_" + k] = samples[e[k]]
        meta.append(e)
        arrs["p%d_full" % n] = F[:, sel].astype(np.float16); arrs["p%d_between" % n] = Bt[:, sel].astype(np.float16)
        arrs["p%d_within" % n] = W[:, sel].astype(np.float16); arrs["p%d_meanA" % n] = mA[sel].astype(np.float32)
        print(p["kind"], p.get("cond", p.get("plate")), "rows", len(rows), "stripe B median full/between/within",
              round(float(np.median(e["stripe_full"])), 3), round(float(np.median(e["stripe_between"])), 3), round(float(np.median(e["stripe_within"])), 3), flush=True)
    return arrs, meta


def labels():
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
    for c, a, b, g in C.groups(d)[0]:
        if g == "G13":
            a, b = (a, b) if plate[b] == "plate13" else (b, a)
            P.append(("plate13", a, b, c))
    out = {}
    for g in ("G614", "diffpos same name", "diffpos salt", "sameplate", "plate13"):
        rs = [(a, b, c) for gg, a, b, c in P if gg == g and fin[:, a].sum() >= 10 and fin[:, b].sum() >= 10 and max(amp[a], amp[b]) >= 3]
        Ai = [a for a, _, _ in rs]; Bi = [b for _, b, _ in rs]
        def lineup(u, v):
            """Similarity of u to every treated well of v's plate other than v, sorted, and to v itself."""
            ws = np.where((plate == plate[v]) & (np.arange(J) != v) & (np.arange(J) != u))[0]
            return sorted((Mn[ws] @ Mn[u]).round(4).tolist(), reverse=True), round(float(Mn[u] @ Mn[v]), 4)
        out[g] = dict(cond=[c for _, _, c in rs], plates=[[str(plate[a]), str(plate[b])] for a, b, _ in rs],
                      amp=[[float(amp[a]), float(amp[b])] for a, b, _ in rs], pct=[[pct(a, b), pct(b, a)] for a, b, _ in rs],
                      C=(Mn[Ai] @ Mn[Bi].T).round(4).tolist(), lineup=[[lineup(a, b), lineup(b, a)] for a, b, _ in rs])
        print("labels", g, len(rs), "recognised", sum(min(x) >= 0.9 for x in out[g]["pct"]), flush=True)
    return out


def main():
    if sys.argv[1:] == ["labels"]:               # label part only (fig110_labels.json)
        json.dump(labels(), open(OUT + "fig110_labels.json", "w")); print("SUMMARY done", flush=True); return
    arrs, meta = heat()
    np.savez_compressed(OUT + "fig110.npz", **arrs)
    json.dump(dict(rules=__doc__, pairs=meta, labels=labels()), open(OUT + "fig110.json", "w"))
    print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
