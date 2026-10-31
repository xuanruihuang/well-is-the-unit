"""Tahoe-100M single cells -> per (cell line, sample, random half) aggregates, on the A100 (numpy / scipy / pyarrow
from $WORKSPACE/pylib). One streaming pass; used by the M2 (context dimensions, principle two) and M5 (fate, gain)
analyses, so Tahoe goes through the same pipeline as the CRISPR data (cell halves for denoising, own phase calls).
Usage:
  tahoe_extract.py map <data_dir> <out_dir> <workers>    processes parquet files not yet listed in <out_dir>/done.txt;
      each worker keeps one dense accumulator (groups x genes, float32) over its share of files and writes one
      partial (files are not grouped by sample: ~5,000 groups per file, so per-file partials would be ~1.4 TB)
  tahoe_extract.py reduce <out_dir>                         add all partials to the running totals (sums64.npy,
      counts.npy, float64) and delete them; run after every map batch
Gene axis: the 18,151 genes of the PIE Tahoe preprocessed dir (meta.json, >= 5 CPM); token -> symbol from
gene_metadata. Expression: log1p CP10K, library size = sum of all counts of the cell (tokens < 3 are special and
skipped). Half: crc32(BARCODE_SUB_LIB_ID) mod 2.
Phase per cell (as explore_L_cellcycle): Tirosh S / G2M genes split at random (seed 7), half 1 used; score = mean
log1p of the set genes minus mean of a control set (10 random genes from the same of 25 expression bins per set gene;
bins from that cell line's DMSO mean, PIE ctrl_means.npy); G1 if both scores <= 0, else the larger.
Per group (line, sample, half): sum of log1p CP10K over the gene axis (float64 in the reducer), cell count, phase
counts (G1, S, G2M), sum of library sizes."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, glob, json, zlib, time
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from multiprocessing import Pool

S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1",
           "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3",
           "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
           "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO",
       "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
       "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
       "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3",
       "CBX5", "CENPA"]
P53 = ["MDM2", "FDXR", "TP53I3", "RRM2B", "ZMAT3", "BAX", "CDKN1A", "GDF15", "SESN1", "SESN2", "TNFRSF10B", "AEN",
       "PLK3", "TP53INP1", "DDB2", "XPC", "RPS27L", "TRIAP1", "FAS", "BBC3", "PHLDA3", "SPATA18", "POLH", "TIGAR",
       "CCNG1", "APAF1", "EDA2R", "SULF2"]
ROOT = (_WS + "/data_ext/tahoe")
PIE = ROOT + "/PIE_tahoe100m/preprocessed"
META = ROOT + "/Tahoe-100M/metadata"
G_ = {}


def setup():
    meta = json.load(open(PIE + "/meta.json"))
    genes = meta["genes"]; gpos = {g: k for k, g in enumerate(genes)}
    lines = sorted(meta["context_to_id"], key=lambda c: meta["context_to_id"][c])
    gm = pq.read_table(glob.glob(META + "/gene_metadata*.parquet")[0]).to_pydict()
    tok = np.array([int(t) for t in gm["token_id"]])
    lut = np.full(tok.max() + 1, -1, np.int64)
    for t, s in zip(tok, gm["gene_symbol"]):
        if s in gpos:
            lut[t] = gpos[s]
    sm = pq.read_table(glob.glob(META + "/sample_metadata*.parquet")[0]).to_pydict()
    samples = sorted(set(sm["sample"]))
    ctrl = np.load(PIE + "/ctrl_means.npy")           # C x G, log1p, rows in context_to_id order
    rng = np.random.default_rng(7)
    sS = [g for g in S_GENES if g in gpos]; sS = list(rng.permutation(sS))[: len(sS) // 2]
    sG = [g for g in G2M if g in gpos]; sG = list(rng.permutation(sG))[: len(sG) // 2]
    iS, iG = np.array([gpos[g] for g in sS]), np.array([gpos[g] for g in sG])
    ctl = {}
    for c in range(len(lines)):                        # per line control sets (as explore_L_cellcycle.phases)
        mu = ctrl[c]; bins = np.digitize(mu, np.quantile(mu, np.linspace(0, 1, 26)[1:-1]))
        r = np.random.default_rng(100 + c)
        ctl[c] = tuple(np.concatenate([r.choice(np.where(bins == bins[j])[0], 10, replace=False) for j in ix])
                       for ix in (iS, iG))
    sub = np.array(sorted({gpos[g] for g in P53 + S_GENES + G2M if g in gpos}))
    return dict(sub=sub, genes=genes, lines=lines, lidx={c: k for k, c in enumerate(lines)}, lut=lut, samples=samples,
                sidx={s: k for k, s in enumerate(samples)}, iS=iS, iG=iG, ctl=ctl)


def init():
    G_.update(setup())


def do_chunk(args):
    files, out, tag = args
    C, P, G = len(G_["lines"]), len(G_["samples"]), len(G_["genes"])
    acc = np.zeros((C * P * 2, G), np.float32); cnt = np.zeros((C * P * 2, 5))
    sub = G_["sub"]; accp = np.zeros((C * P * 2, 3, len(sub)), np.float32)   # phase-resolved sums on the gene subset
    ncell = 0; t0 = time.time()
    for fn in files:
        pf = pq.ParquetFile(fn)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["genes", "expressions", "cell_line_id", "sample", "BARCODE_SUB_LIB_ID"])
            gcol, ecol = t.column("genes").combine_chunks(), t.column("expressions").combine_chunks()
            off = gcol.offsets.to_numpy(); tok = gcol.values.to_numpy(); val = ecol.values.to_numpy().astype(np.float64)
            n = len(off) - 1; ncell += n
            cell = np.repeat(np.arange(n), np.diff(off))
            ok = (tok >= 3) & (val > 0)
            lib = np.bincount(cell[ok], weights=val[ok], minlength=n)
            gi = np.where(tok < len(G_["lut"]), G_["lut"][np.minimum(tok, len(G_["lut"]) - 1)], -1)
            keep = ok & (gi >= 0)
            y = np.log1p(val[keep] / lib[cell[keep]] * 1e4)
            X = csr_matrix((y, (cell[keep], gi[keep])), shape=(n, G))
            line = np.array([G_["lidx"].get(c, -1) for c in t.column("cell_line_id").to_pylist()])
            smp = np.array([G_["sidx"].get(s_, -1) for s_ in t.column("sample").to_pylist()])
            half = np.array([zlib.crc32(b_.encode()) & 1 for b_ in t.column("BARCODE_SUB_LIB_ID").to_pylist()])
            ph = np.zeros(n, np.int8)
            sS_all = np.asarray(X[:, G_["iS"]].mean(1)).ravel(); sG_all = np.asarray(X[:, G_["iG"]].mean(1)).ravel()
            for c in np.unique(line[line >= 0]):
                m = np.where(line == c)[0]
                cS, cG = G_["ctl"][c]
                a = sS_all[m] - np.asarray(X[m][:, cS].mean(1)).ravel()
                b = sG_all[m] - np.asarray(X[m][:, cG].mean(1)).ravel()
                ph[m] = np.where((a <= 0) & (b <= 0), 0, np.where(a >= b, 1, 2))
            valid = (line >= 0) & (smp >= 0)
            key = (line.astype(np.int64) * P + smp) * 2 + half
            kv = key[valid]; iv = np.where(valid)[0]
            uk, inv = np.unique(kv, return_inverse=True)
            Gm = csr_matrix((np.ones(len(iv)), (inv, iv)), shape=(len(uk), n))
            acc[uk] += (Gm @ X).toarray().astype(np.float32)
            Xs = X[:, sub]
            for q in range(3):
                mq = ph[iv] == q
                if mq.any():
                    ukq, invq = np.unique(kv[mq], return_inverse=True)
                    Gq = csr_matrix((np.ones(mq.sum()), (invq, iv[mq])), shape=(len(ukq), n))
                    accp[ukq, q] += (Gq @ Xs).toarray().astype(np.float32)
            np.add.at(cnt, (uk[inv], 0), 1)
            for q in range(3):
                np.add.at(cnt, (kv[ph[iv] == q], 1 + q), 1)
            np.add.at(cnt, (kv, 4), lib[iv])
    np.save(os.path.join(out, "parts", "acc_%s.npy" % tag), acc)
    np.save(os.path.join(out, "parts", "cnt_%s.npy" % tag), cnt)
    np.save(os.path.join(out, "parts", "accp_%s.npy" % tag), accp)
    return files, ncell, time.time() - t0


def reduce(out):
    S = setup(); C, P, G = len(S["lines"]), len(S["samples"]), len(S["genes"])
    # running totals across batches: merged partials are added to sums64.npy / counts.npy and then deleted
    if os.path.exists(out + "/sums64.npy"):
        acc = np.load(out + "/sums64.npy"); cnt = np.load(out + "/counts.npy"); accp = np.load(out + "/phase_sums.npy")
    else:
        acc = np.zeros((C * P * 2, G), np.float64); cnt = np.zeros((C * P * 2, 5))
        accp = np.zeros((C * P * 2, 3, len(S["sub"])), np.float64)
    parts = sorted(glob.glob(out + "/parts/acc_*.npy"))
    for f in parts:
        acc += np.load(f); cnt += np.load(f.replace("acc_", "cnt_")); accp += np.load(f.replace("acc_", "accp_"))
        print("reduce", f, flush=True)
    np.save(out + "/sums64.npy.tmp.npy", acc); os.replace(out + "/sums64.npy.tmp.npy", out + "/sums64.npy")
    np.save(out + "/counts.npy", cnt); np.save(out + "/phase_sums.npy", accp)
    for f in parts:
        os.remove(f); os.remove(f.replace("acc_", "cnt_")); os.remove(f.replace("acc_", "accp_"))
    json.dump(dict(lines=S["lines"], samples=S["samples"], genes=S["genes"], phase_sum_genes=[S["genes"][j] for j in S["sub"]], n_parts_merged=len(parts),
                   key="(line * n_samples + sample) * 2 + half", counts_cols=["n", "G1", "S", "G2M", "lib_sum"]),
              open(out + "/index.json", "w"))
    print("reduced", len(parts), "parts; cells", int(cnt[:, 0].sum()), flush=True)


def main():
    mode = sys.argv[1]
    if mode == "map":
        data, out, nw = sys.argv[2], sys.argv[3], int(sys.argv[4])
        os.makedirs(out + "/parts", exist_ok=True)
        done = set(open(out + "/done.txt").read().split()) if os.path.exists(out + "/done.txt") else set()
        files = [f for f in sorted(glob.glob(data + "/train-*.parquet")) if os.path.basename(f) not in done]
        batch = time.strftime("%m%d%H%M%S")
        chunks = [(files[k::nw], out, "%s_%02d" % (batch, k)) for k in range(nw) if files[k::nw]]
        t0 = time.time(); tot = 0
        with Pool(len(chunks), initializer=init) as pool:
            for fs, n, dt in pool.imap_unordered(do_chunk, chunks):
                tot += n
                with open(out + "/done.txt", "a") as fh:
                    fh.write("".join(os.path.basename(f) + "\n" for f in fs))
                print("chunk done", len(fs), "files", n, "cells", "%.0fs" % dt, flush=True)
        print("map done", len(files), "files", tot, "cells", "%.0fs" % (time.time() - t0), flush=True)
    else:
        reduce(sys.argv[2])


if __name__ == "__main__":
    main()
