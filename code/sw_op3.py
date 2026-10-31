"""D (samewell/notes.md §6, written before running): same-well sharing in OP3 (NeurIPS 2023 PBMC compound screen).
Data: Open Problems pseudobulk per (well, cell type), raw counts (workflow-resources/neurips-2023-data/pseudobulk.h5ad;
3,318 x 21,265). Profiles log1p(1e4 x count / total). Response of cell type c in well w = profile minus the mean of the
same cell type in the plate's DMSO wells (leaving the well itself out when it is a DMSO well). Genes: control mean
log1p >= 0.2 in the target cell type; the well mean uses genes expressed in all six cell types.
Replicates within donor: DMSO, belinostat and dabrafenib each have 16 wells per donor (two plates x eight rows).
For pairs of such wells (up to 60 random pairs per compound x donor, seed 0) and each target cell type:
  R2: target response vs the mean response of the other cell types in its own well / in the other well
  pred: other cell types' mean as the prediction (Kaggle structure: B cells and myeloid cells predicted from T and NK
        cell types), input from the same well vs the other well; bound sqrt(max(r, 0)) with r the target's agreement
        between the two wells.
Secondary (all 146 compounds): the Kaggle task across donors, input from the same well vs from the same compound's
well in another donor (changes donor as well as well).
Output $WORKSPACE/mainline_rt_samewell/op3/op3_results.json. Usage: python sw_op3.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections, itertools
import numpy as np
import h5py
from scipy.sparse import csr_matrix

DIR = (_WS + "/mainline_rt_samewell/op3/")
TN = ["T cells CD4+", "T cells CD8+", "T regulatory cells", "NK cells"]; BM = ["B cells", "Myeloid cells"]


def pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def load():
    f = h5py.File(DIR + "pseudobulk.h5ad", "r"); o = f["obs"]
    def col(k):
        g = o[k]
        if isinstance(g, h5py.Group):
            cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]; return np.array([cats[i] for i in g["codes"][:]])
        return np.array([x.decode() if isinstance(x, bytes) else x for x in g[:]])
    X = csr_matrix((f["X"]["data"][:], f["X"]["indices"][:], f["X"]["indptr"][:]), shape=tuple(f["X"].attrs["shape"]))
    tot = np.asarray(X.sum(1)).ravel()
    P = np.log1p(np.asarray(X.multiply(1e4 / tot[:, None]).todense()))
    meta = {k: col(k) for k in ("cell_type", "sm_name", "donor_id", "plate_name", "well", "row")}
    return P, meta


def main():
    P, m = load(); ct, sm, don, pl, wl = m["cell_type"], m["sm_name"], m["donor_id"], m["plate_name"], m["well"]
    cts = sorted(set(ct)); isd = sm == "Dimethyl Sulfoxide"
    expr = {c: P[(ct == c) & isd].mean(0) >= 0.2 for c in cts}
    common = np.all([expr[c] for c in cts], 0)
    idx = {(pl[i], wl[i], ct[i]): i for i in range(len(ct))}
    R = {}
    for i in range(len(ct)):
        refs = [j for j in np.where((ct == ct[i]) & isd & (pl == pl[i]))[0] if j != i]
        if refs:
            R[i] = P[i] - P[refs].mean(0)
    wells = collections.defaultdict(list)
    for (p_, w_, c_), i in idx.items():
        wells[(p_, w_)].append(i)
    def resp(p_, w_, c_):
        i = idx.get((p_, w_, c_)); return R.get(i) if i is not None else None
    def others_mean(p_, w_, cs, genes):
        v = [resp(p_, w_, c)[genes] for c in cs if resp(p_, w_, c) is not None]
        return np.mean(v, 0) if len(v) >= 2 else None
    rng = np.random.default_rng(0); recs = []
    for comp in ("Dimethyl Sulfoxide", "Belinostat", "Dabrafenib"):
        for dn in sorted(set(don)):
            ws = sorted({(pl[i], wl[i]) for i in np.where((sm == comp) & (don == dn))[0]})
            pairs = list(itertools.combinations(ws, 2)); rng.shuffle(pairs)
            for (a, b) in pairs[:60]:
                for t in cts:
                    ya, yb = resp(*a, t), resp(*b, t)
                    if ya is None or yb is None:
                        continue
                    g = np.where(expr[t])[0]; gc = np.where(common)[0]
                    oth = [c for c in cts if c != t]
                    inp = TN if t in BM else oth
                    ma, mb = others_mean(*a, inp, g), others_mean(*b, inp, g)
                    wa, wb = others_mean(*a, oth, gc), others_mean(*b, oth, gc)
                    if ma is None or mb is None or wa is None or wb is None:
                        continue
                    r = pear(ya[g], yb[g]); bnd = np.sqrt(max(r, 0))
                    same = 0.5 * (pear(ma, ya[g]) + pear(mb, yb[g])); other = 0.5 * (pear(mb, ya[g]) + pear(ma, yb[g]))
                    recs.append(dict(comp=comp, donor=dn, target=t, ceiling=r, same=same, other=other,
                                     same_gt=bool(pear(ma, ya[g]) > bnd), other_gt=bool(pear(mb, ya[g]) > bnd),
                                     r2_same=0.5 * (pear(ya[gc], wa) ** 2 + pear(yb[gc], wb) ** 2), r2_other=0.5 * (pear(ya[gc], wb) ** 2 + pear(yb[gc], wa) ** 2)))
    res = {"n_genes_common": int(common.sum())}
    def summ(sub):
        f = lambda k: round(float(np.nanmedian([r[k] for r in sub])), 3)
        return dict(n=len(sub), ceiling=f("ceiling"), same=f("same"), other=f("other"), r2_same=f("r2_same"), r2_other=f("r2_other"),
                    same_gt_bound=round(float(np.mean([r["same_gt"] for r in sub])), 3), other_gt_bound=round(float(np.mean([r["other_gt"] for r in sub])), 3))
    for comp in ("Dimethyl Sulfoxide", "Belinostat", "Dabrafenib"):
        res[comp] = summ([r for r in recs if r["comp"] == comp])
        res[comp + " | B and myeloid targets"] = summ([r for r in recs if r["comp"] == comp and r["target"] in BM])
        print(comp, json.dumps(res[comp]), flush=True)
    # secondary: Kaggle structure across donors, all compounds
    sec = []
    comps = sorted(set(sm[~isd]))
    for comp in comps:
        byd = collections.defaultdict(list)
        for i in np.where(sm == comp)[0]:
            byd[don[i]].append((pl[i], wl[i]))
        ds = sorted(byd)
        for d1, d2 in itertools.permutations(ds, 2):
            a, b = byd[d1][0], byd[d2][0]
            for t in BM:
                ya, yb = resp(*a, t), resp(*b, t)
                if ya is None or yb is None:
                    continue
                g = np.where(expr[t])[0]; ma, mb = others_mean(*a, TN, g), others_mean(*b, TN, g)
                if ma is None or mb is None:
                    continue
                r = pear(ya[g], yb[g]); bnd = np.sqrt(max(r, 0))
                sec.append(dict(comp=comp, target=t, same=pear(ma, ya[g]), other=pear(mb, ya[g]), ceiling=r,
                                same_gt=bool(pear(ma, ya[g]) > bnd), other_gt=bool(pear(mb, ya[g]) > bnd)))
    f = lambda k: round(float(np.nanmedian([r[k] for r in sec])), 3)
    res["kaggle_structure_across_donors"] = dict(n=len(sec), n_comp=len({r["comp"] for r in sec}), same=f("same"), other=f("other"), ceiling=f("ceiling"),
                                                 same_gt_bound=round(float(np.mean([r["same_gt"] for r in sec])), 3), other_gt_bound=round(float(np.mean([r["other_gt"] for r in sec])), 3))
    print("kaggle structure", json.dumps(res["kaggle_structure_across_donors"]), flush=True)
    json.dump(res, open(DIR + "op3_results.json", "w"), indent=1)
    json.dump(recs + [dict(r, comp="sec:" + r["comp"]) for r in sec], open(DIR + "op3_records.json", "w"))


if __name__ == "__main__":
    main()
