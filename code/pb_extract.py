"""Raw-count pseudobulks of Tahoe-100M in the BioPert layout (A100), for the same-well leakage study.
One streaming pass over the 3,388 data parquets. Per (cell line, sample, random cell half): sum of raw UMI counts over
the protein-coding genes (HGNC list shipped with the BioPert repository, data/protein-coding_gene.txt; tokens of
gene_metadata whose symbol is in it, as BioPert's preprocess_tahoe.jl), number of cells, coding UMIs, all UMIs
(tokens >= 3) and mitochondrial UMIs (symbols starting with MT-). Half = crc32(BARCODE_SUB_LIB_ID) & 1, as tahoe_extract.py.
BioPert pseudobulks are the sums of the two halves.
Usage: pb_extract.py map <workers> | reduce. Output dir $WORKSPACE/rt/samewell/pb/."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, glob, json, zlib, time
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from multiprocessing import Pool

DATA = (_WS + "/data_ext/tahoe/Tahoe-100M/data")
META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata")
OUT = (_WS + "/rt/samewell/pb")
PC = (_WS + "/rt/samewell/protein-coding_gene.txt")
G_ = {}


def setup():
    hdr = open(PC).readline().rstrip("\n").split("\t"); si = hdr.index("symbol")
    coding = {l.rstrip("\n").split("\t")[si] for l in open(PC).readlines()[1:]}
    gm = pq.read_table(META + "/gene_metadata.parquet").to_pydict()
    tok = [int(t) for t in gm["token_id"]]; sym = gm["gene_symbol"]
    keep = [(t, s) for t, s in zip(tok, sym) if s in coding]
    lut = np.full(max(tok) + 1, -1, np.int64)
    for k, (t, s) in enumerate(keep):
        lut[t] = k
    mito = np.zeros(max(tok) + 1, bool)
    for t, s in zip(tok, sym):
        if isinstance(s, str) and s.startswith("MT-"):
            mito[t] = True
    lines = json.load(open((_WS + "/tahoe_agg/index.json")))["lines"]
    samples = sorted(set(pq.read_table(META + "/sample_metadata.parquet").column("sample").to_pylist()))
    return dict(lut=lut, mito=mito, genes=[s for _, s in keep], tokens=[t for t, _ in keep], lines=lines,
                lidx={c: k for k, c in enumerate(lines)}, samples=samples, sidx={s: k for k, s in enumerate(samples)})


def init():
    G_.update(setup())


def do_chunk(args):
    files, tag = args
    C, P, G = len(G_["lines"]), len(G_["samples"]), len(G_["genes"])
    acc = np.zeros((C * P * 2, G), np.float32); cnt = np.zeros((C * P * 2, 4)); lut, mito = G_["lut"], G_["mito"]
    ncell = 0; nbad = 0; t0 = time.time()
    for fn in files:
        pf = pq.ParquetFile(fn)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["genes", "expressions", "cell_line_id", "sample", "BARCODE_SUB_LIB_ID"])
            gcol, ecol = t.column("genes").combine_chunks(), t.column("expressions").combine_chunks()
            off = gcol.offsets.to_numpy(); tok = gcol.values.to_numpy(); val = ecol.values.to_numpy().astype(np.float64)
            n = len(off) - 1; ncell += n
            cell = np.repeat(np.arange(n), np.diff(off))
            ok = (tok >= 3) & (val > 0)
            tt = np.minimum(tok, len(lut) - 1)
            gi = np.where(ok, lut[tt], -1)
            keep = gi >= 0
            X = csr_matrix((val[keep], (cell[keep], gi[keep])), shape=(n, G))
            allumi = np.bincount(cell[ok], weights=val[ok], minlength=n)
            cod = np.bincount(cell[keep], weights=val[keep], minlength=n)
            mt = ok & mito[tt]
            mtu = np.bincount(cell[mt], weights=val[mt], minlength=n)
            line = np.array([G_["lidx"].get(c, -1) for c in t.column("cell_line_id").to_pylist()])
            smp = np.array([G_["sidx"].get(s_, -1) for s_ in t.column("sample").to_pylist()])
            half = np.array([zlib.crc32(b_.encode()) & 1 for b_ in t.column("BARCODE_SUB_LIB_ID").to_pylist()])
            valid = (line >= 0) & (smp >= 0); nbad += int((~valid).sum())
            key = (line.astype(np.int64) * P + smp) * 2 + half
            kv = key[valid]; iv = np.where(valid)[0]
            uk, inv = np.unique(kv, return_inverse=True)
            Gm = csr_matrix((np.ones(len(iv)), (inv, iv)), shape=(len(uk), n))
            acc[uk] += (Gm @ X).toarray().astype(np.float32)
            np.add.at(cnt, (kv, 0), 1); np.add.at(cnt, (kv, 1), cod[iv]); np.add.at(cnt, (kv, 2), allumi[iv]); np.add.at(cnt, (kv, 3), mtu[iv])
    np.save(os.path.join(OUT, "parts", "acc_%s.npy" % tag), acc)
    np.save(os.path.join(OUT, "parts", "cnt_%s.npy" % tag), cnt)
    return len(files), ncell, nbad, time.time() - t0


def run_map(workers):
    os.makedirs(OUT + "/parts", exist_ok=True)
    files = sorted(glob.glob(DATA + "/*.parquet"))
    chunks = [(files[i::workers], "%03d" % i) for i in range(workers)]
    with Pool(workers, initializer=init) as p:
        for nf, nc, nb, dt in p.imap_unordered(do_chunk, chunks):
            print("chunk done", nf, "files", nc, "cells", nb, "unmatched", round(dt), "s", flush=True)


def reduce():
    S = setup(); C, P, G = len(S["lines"]), len(S["samples"]), len(S["genes"])
    acc = np.zeros((C * P * 2, G), np.float64); cnt = np.zeros((C * P * 2, 4))
    for f in sorted(glob.glob(OUT + "/parts/acc_*.npy")):
        acc += np.load(f); cnt += np.load(f.replace("acc_", "cnt_")); print("reduce", f, flush=True)
    np.save(OUT + "/counts_lsh.npy", acc.astype(np.float32)); np.save(OUT + "/cnt_lsh.npy", cnt)
    json.dump(dict(lines=S["lines"], samples=S["samples"], genes=S["genes"], tokens=S["tokens"],
                   layout="row = (line * n_samples + sample) * 2 + half", cnt_cols=["cells", "coding_umi", "all_umi", "mito_umi"]),
              open(OUT + "/index.json", "w"))
    print("cells", cnt[:, 0].sum(), "genes", G, flush=True)
    for f in glob.glob(OUT + "/parts/*.npy"):
        os.remove(f)


if __name__ == "__main__":
    if sys.argv[1] == "map":
        run_map(int(sys.argv[2]))
    else:
        reduce()
