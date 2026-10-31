"""samewell notes section 103: out-of-sample control-pair test on Tahoe Emerald Bay (DMSO_TF wells).

Input: parquet files with only the DMSO_TF rows of tahoebio/EmeraldBay expression_data
(written by the extraction step: one file per source row group; columns genes, expressions,
drug, drugname_drugconc, cell_line, sample, BARCODE_SUB_LIB_ID; genes = token ids of non-zero
genes, expressions = raw counts).

Design (copied from notes.md section 103, fixed before the run):
- per cell: log(1 + 1e4 * count / total); per (cell line, well) mean; keep (line, well) with >= 30 cells.
- genes of a line: the line's pooled-control mean (all its cells in the 4 DMSO_TF wells) >= 0.2.
- 6 control-well pairs. d_t = line t's difference between the two wells. Same-pair predictor = mean
  of the other lines' d in the same pair; other-pair predictor = mean of the other lines' d in the
  complementary disjoint pair. Record = line x pair; Pearson on the line's genes.
- summaries: record medians; per-pair medians and number of pairs with same > other; sign test over
  lines (each line's mean over the pairs); adjacent-barcode pairs (85-86, 86-87) vs pairs with 54;
  cells and depth per well.
Implementation choices not fixed by the text (set before results were seen):
- wells are identified by the first-round barcode = BARCODE_SUB_LIB_ID.split('_')[0];
- each difference is (lower barcode well) - (higher barcode well), for the pair and its complement;
- other lines entering a predictor must pass the >= 30 cell filter in both wells of that pair;
  a line enters records of a pair only if it passes in both wells of the pair and of its complement;
- depth = total raw counts per cell (the sum of `expressions`).

Usage: python emerald103.py <dir with filtered parquet files> <out.json>
"""
import sys, os, glob, json, itertools
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

WELLS = ['54', '85', '86', '87']
NG = 63287          # token ids 3..63286 (gene_metadata)
MINCELLS = 30
GENE_MIN = 0.2


def load(d):
    files = sorted(glob.glob(os.path.join(d, '*.parquet')))
    lines_all, wells_all, tot_all, nnz_all = [], [], [], []
    keys = {}                       # (line, well) -> group index
    sums = {}                       # group index -> per-gene sum of log values
    seen_cond = set(); seen_samp = {}
    neg = 0
    for fn in files:
        t = pq.read_table(fn)
        cond = t['drugname_drugconc'].to_pylist()
        seen_cond.update(cond)
        cl = t['cell_line'].to_pylist()
        bc = t['BARCODE_SUB_LIB_ID'].to_pylist()
        sm = t['sample'].to_pylist()
        w = [b.split('_')[0] for b in bc]
        for a, b in zip(w, sm):
            seen_samp.setdefault(a, set()).add(b)
        g_list = t['genes'].combine_chunks()
        e_list = t['expressions'].combine_chunks()
        off = g_list.offsets.to_numpy()
        gv = g_list.values.to_numpy().astype(np.int64)
        ev = e_list.values.to_numpy().astype(np.float64)
        assert np.array_equal(off, e_list.offsets.to_numpy())
        neg += int((ev < 0).sum())
        n = len(cl)
        lens = np.diff(off)
        cell_of = np.repeat(np.arange(n), lens)
        tot = np.bincount(cell_of, weights=ev, minlength=n)
        x = np.log1p(1e4 * ev / tot[cell_of])
        gi = np.empty(n, dtype=np.int64)
        for k in range(n):
            key = (cl[k], w[k])
            if key not in keys:
                keys[key] = len(keys)
            gi[k] = keys[key]
        for g in np.unique(gi):
            m = gi[cell_of] == g
            s = np.bincount(gv[m], weights=x[m], minlength=NG)
            sums[g] = sums.get(g, 0) + s
        lines_all += cl; wells_all += w; tot_all.append(tot); nnz_all.append(lens)
    return dict(keys=keys, sums=sums, lines=np.array(lines_all), wells=np.array(wells_all),
                tot=np.concatenate(tot_all), nnz=np.concatenate(nnz_all), conds=sorted(seen_cond),
                samples={k: sorted(v) for k, v in seen_samp.items()}, neg=neg, nfiles=len(files))


def main(d, out):
    D = load(d)
    assert D['conds'] == ["[('DMSO_TF', 0.0, 'uM')]"], D['conds']
    assert sorted(D['samples']) == WELLS, D['samples']
    lines, wells, tot, nnz = D['lines'], D['wells'], D['tot'], D['nnz']
    L = sorted(set(lines))
    # counts per (line, well)
    ncell = {(l, w): 0 for l in L for w in WELLS}
    for l, w in zip(lines, wells):
        ncell[(l, w)] += 1
    # per (line, well) means; pooled means per line
    mean = {}
    pooled = {}
    for l in L:
        s_all = np.zeros(NG); n_all = 0
        for w in WELLS:
            k = D['keys'].get((l, w))
            if k is None:
                continue
            s_all += D['sums'][k]; n_all += ncell[(l, w)]
            if ncell[(l, w)] >= MINCELLS:
                mean[(l, w)] = D['sums'][k] / ncell[(l, w)]
        pooled[l] = s_all / n_all
    genes = {l: np.where(pooled[l] >= GENE_MIN)[0] for l in L}

    pairs = list(itertools.combinations(WELLS, 2))
    comp = {p: tuple(sorted(set(WELLS) - set(p))) for p in pairs}
    ok = {p: [l for l in L if (l, p[0]) in mean and (l, p[1]) in mean] for p in pairs}
    dd = {p: {l: mean[(l, p[0])] - mean[(l, p[1])] for l in ok[p]} for p in pairs}

    recs = []
    for p in pairs:
        q = comp[p]
        for t in ok[p]:
            if t not in dd[q]:
                continue
            G = genes[t]
            same = np.mean([dd[p][s] for s in ok[p] if s != t], axis=0)
            other = np.mean([dd[q][s] for s in ok[q] if s != t], axis=0)
            y = dd[p][t][G]
            rs = float(np.corrcoef(y, same[G])[0, 1])
            ro = float(np.corrcoef(y, other[G])[0, 1])
            recs.append(dict(line=t, pair='-'.join(p), comp='-'.join(q), n_genes=int(len(G)),
                             n_same=len(ok[p]) - 1, n_other=len([s for s in ok[q] if s != t]),
                             same=rs, other=ro))

    def med(v):
        return float(np.median(v)) if len(v) else None
    S = np.array([r['same'] for r in recs]); O = np.array([r['other'] for r in recs])
    per_pair = {}
    for p in pairs:
        pk = '-'.join(p)
        rr = [r for r in recs if r['pair'] == pk]
        per_pair[pk] = dict(comp='-'.join(comp[p]), n=len(rr), same_median=med([r['same'] for r in rr]),
                            other_median=med([r['other'] for r in rr]))
    n_pairs_same_gt = sum(1 for v in per_pair.values() if v['same_median'] > v['other_median'])
    # sign test over lines on their mean over pairs
    per_line = {}
    for l in L:
        rr = [r for r in recs if r['line'] == l]
        if rr:
            per_line[l] = dict(n_pairs=len(rr), same_mean=float(np.mean([r['same'] for r in rr])),
                               other_mean=float(np.mean([r['other'] for r in rr])))
    k = sum(1 for v in per_line.values() if v['same_mean'] > v['other_mean'])
    ties = sum(1 for v in per_line.values() if v['same_mean'] == v['other_mean'])
    n = len(per_line) - ties
    p_sign = float(binomtest(k, n, 0.5, alternative='two-sided').pvalue)
    groups = {'adjacent (85-86, 86-87)': ['85-86', '86-87'], 'far (54-85, 54-86, 54-87)': ['54-85', '54-86', '54-87'],
              'neither (85-87)': ['85-87']}
    grp = {}
    for gname, pk in groups.items():
        rr = [r for r in recs if r['pair'] in pk]
        grp[gname] = dict(n=len(rr), same_median=med([r['same'] for r in rr]), other_median=med([r['other'] for r in rr]),
                          n_same_gt_other=int(sum(r['same'] > r['other'] for r in rr)))
    # data description
    wells_desc = {}
    for w in WELLS:
        m = wells == w
        npl = [ncell[(l, w)] for l in L]
        wells_desc[w] = dict(sample=D['samples'][w], cells=int(m.sum()), lines_present=int(sum(c > 0 for c in npl)),
                             lines_ge30=int(sum(c >= MINCELLS for c in npl)),
                             median_cells_per_line=float(np.median([c for c in npl if c > 0])),
                             min_cells_per_line=int(min(c for c in npl if c > 0)),
                             median_total_counts=float(np.median(tot[m])), mean_total_counts=float(np.mean(tot[m])),
                             median_genes_detected=float(np.median(nnz[m])))
    res = dict(
        design='samewell notes section 103', n_files=D['nfiles'], n_cells=int(len(lines)), n_lines=len(L),
        negative_expression_values=D['neg'], wells=wells_desc,
        genes_per_line=dict(median=float(np.median([len(genes[l]) for l in L])),
                            min=int(min(len(genes[l]) for l in L)), max=int(max(len(genes[l]) for l in L))),
        n_records=len(recs), record_median_same=med(S), record_median_other=med(O),
        record_n_same_gt_other=int((S > O).sum()),
        per_pair=per_pair, n_pairs_same_gt_other=n_pairs_same_gt,
        sign_test=dict(k_same_gt_other=k, n=n, ties=ties, p_two_sided=p_sign,
                       median_line_same_mean=med([v['same_mean'] for v in per_line.values()]),
                       median_line_other_mean=med([v['other_mean'] for v in per_line.values()])),
        adjacent_vs_far=grp,
        expectations={
            'same > other in >= 5 of 6 pairs': n_pairs_same_gt >= 5,
            'record median same in [0.5, 0.8]': bool(0.5 <= med(S) <= 0.8),
            'record median other in [0, 0.3]': bool(0.0 <= med(O) <= 0.3),
            'sign test p < 0.001': p_sign < 0.001,
        },
        cells_per_line_well={f'{l}|{w}': ncell[(l, w)] for l in L for w in WELLS},
        per_line=per_line, records=recs)
    json.dump(res, open(out, 'w'), indent=1)
    show = {k: v for k, v in res.items() if k not in ('records', 'per_line', 'cells_per_line_well')}
    print(json.dumps(show, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
