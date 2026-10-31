"""sci-Plex3 pseudobulk per well and per random half of the cells (samewell/notes.md §76; 5090).
Input (read-only, extracted from GSE139944 into $WORKSPACE/mainline_rt_samewell/sciplex/): the screen UMI triplets
(gene, cell, count; 1-based), the cell annotations (row order of the triplets) and the per-cell pData (same order).
Cells: assigned to a hash sample (Combo not NA), 24 h. Well = culture plate (plate_oligo) x position (from well_oligo,
assumed to match the culture well). Halves: each cell to A or B at random (seed 0). Profiles log1p(1e4 x summed counts /
summed total) on human genes; genes kept if the mean full-well profile >= 0.05 in at least one cell line.
Output, in the input format of audit_samewell.py: <OUT>/{X.npy, XA.npy, XB.npy, obs.csv (+ n_cells, n_a, n_b), genes.txt},
plus tot.npy (human-gene UMI totals of half A and half B per well, 2 x wells) so that halves can be re-pooled exactly.
Usage: sciplex_pb.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, csv, gzip
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

D = (_WS + "/mainline_rt_samewell/sciplex/"); OUT = D + "pb/"
PDATA = (_WS + "/vcc2026/data/raw/official_datasets/sciplex/meta/GSM4150378_sciPlex3_pData.txt.gz")
PFX = D + "GSM4150378_sciPlex3_A549_MCF7_K562_screen_"


def main():
    os.makedirs(OUT, exist_ok=True)
    genes = [l.split() for l in gzip.open(PFX + "gene.annotations.txt.gz", "rt")]
    if genes[0][0] == "id":
        genes = genes[1:]
    human = np.array([g[0].startswith("ENSG") for g in genes]); ng = len(genes)
    cells = [l.split()[0] for l in gzip.open(PFX + "cell.annotations.txt.gz", "rt")]
    rd = csv.reader(gzip.open(PDATA, "rt"), delimiter=" "); hdr = next(rd)
    col = {h: k + 1 for k, h in enumerate(hdr)}  # rows carry the row name first
    combo, meta = [], {}
    for k, r in enumerate(rd):
        assert r[0] == cells[k], (k, r[0], cells[k])
        c = r[col["Combo"]]
        ok = c != "NA" and r[col["time_point"]] == "24"
        combo.append(c if ok else None)
        if ok and c not in meta:
            meta[c] = dict(context=r[col["cell_type"]], plate=r[col["plate_oligo"]], position=r[col["well_oligo"]].split("_")[1],
                           condition="vehicle" if r[col["vehicle"]] == "TRUE" else "%s|%s" % (r[col["treatment"]], r[col["dose"]]),
                           is_control=int(r[col["vehicle"]] == "TRUE"), replicate=r[col["replicate"]])
    assert len(combo) == len(cells)
    wells = sorted(meta); widx = {w: i for i, w in enumerate(wells)}; nw = len(wells)
    cw = np.array([widx[c] if c else -1 for c in combo], np.int64)
    half = np.random.default_rng(0).integers(0, 2, len(cells))
    row = np.where(cw >= 0, half * nw + cw, -1)  # rows 0..nw-1 half A, nw..2nw-1 half B
    print("cells", len(cells), "kept", int((cw >= 0).sum()), "wells", nw, "genes", ng, flush=True)
    acc = None; n = 0
    for ch in pd.read_csv(PFX + "UMI.count.matrix.gz", sep="\t", header=None, names=["g", "c", "v"], dtype=np.int64, chunksize=50_000_000):
        r = row[ch["c"].values - 1]; keep = r >= 0
        m = csr_matrix((ch["v"].values[keep].astype(np.float64), (r[keep], ch["g"].values[keep] - 1)), shape=(2 * nw, ng))
        acc = m if acc is None else acc + m; n += len(ch)
        print("triplets", n, flush=True)
    tot = np.asarray(acc[:, human].sum(1)).ravel()  # human-gene totals per half-well
    acc = acc[:, np.where(human)[0]].tocsr(); gh = [genes[k] for k in np.where(human)[0]]
    A, B = acc[:nw], acc[nw:]
    def prof(M, t):
        M = M.multiply(1e4 / np.maximum(t, 1)[:, None]).tocsr(); M.data = np.log1p(M.data); return M
    F = prof(A + B, tot[:nw] + tot[nw:])
    ctx = np.array([meta[w]["context"] for w in wells])
    keepg = np.zeros(F.shape[1], bool)
    for c in sorted(set(ctx)):
        keepg |= np.asarray(F[ctx == c].mean(0)).ravel() >= 0.05
    gi = np.where(keepg)[0]; np.save(OUT + "tot.npy", np.stack([tot[:nw], tot[nw:]]).astype(np.float64))
    np.save(OUT + "X.npy", F[:, gi].toarray().astype(np.float32))
    np.save(OUT + "XA.npy", prof(A, tot[:nw])[:, gi].toarray().astype(np.float32))
    np.save(OUT + "XB.npy", prof(B, tot[nw:])[:, gi].toarray().astype(np.float32))
    with open(OUT + "genes.txt", "w") as f:
        f.write("\n".join("%s\t%s" % (gh[k][0], gh[k][1] if len(gh[k]) > 1 else "") for k in gi) + "\n")
    na = np.bincount(cw[(cw >= 0) & (half == 0)], minlength=nw); nb = np.bincount(cw[(cw >= 0) & (half == 1)], minlength=nw)
    with open(OUT + "obs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["context", "well", "plate", "position", "condition", "is_control", "replicate", "n_cells", "n_a", "n_b"]); w.writeheader()
        for i, wl in enumerate(wells):
            m = meta[wl]; w.writerow(dict(m, well="%s|%s" % (m["plate"], m["position"]), n_cells=int(na[i] + nb[i]), n_a=int(na[i]), n_b=int(nb[i])))
    print("genes kept", len(gi), "median cells per well", float(np.median(na + nb)), "SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
