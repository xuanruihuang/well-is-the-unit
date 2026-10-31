"""samewell/notes.md §19: cell-eval-style metrics on pseudobulk changes, Rhaister (clean inputs, §12) and the other
lines' mean, scored against the same well (training plate a) and a replicate well (another plate b) (A100).
Records: held-out line x replicated test drug-dose (5 splits); truths from the per-plate cell_eval files (2,000 genes).
For a replicated drug-dose with several other plates, b is the first other plate in plate order.
Metrics per record: pearson_delta; mse_delta, mae_delta; top-N overlap (N = 50, 100) of genes ranked by |change|;
direction match on the true top 100 genes. Discrimination (L1, L2, cosine): within one held-out line and split, among
its replicated test drug-doses, 1 - (number of other drug-doses whose truth is closer to the prediction than its own
truth) / (number of drug-doses), with all truths from the same well type (a, or each drug-dose's b).
Predictors: Rhaister few-shot, Rhaister-O zero-shot, other lines' mean in well a (training plate).
Output $WORKSPACE/rt/samewell/rhaister/out/metrics_same_other.json. Usage: rh_metrics.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, json, collections
import numpy as np
import pyarrow.parquet as pq

RH = (_WS + "/rt/samewell/rhaister"); ORIG, OUT = RH + "/data", RH + "/out"


def pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def topn(a, b, n):
    ta = set(np.argsort(-np.abs(a))[:n]); tb = set(np.argsort(-np.abs(b))[:n]); return len(ta & tb) / n


def main():
    info = json.load(open(RH + "/replicated_treatments.json")); rep, plates = info["train_plate"], info["replicated"]
    cache = {}

    def plate(p, genes):
        if p not in cache:
            cache[p] = pq.read_table(ORIG + "/cell_eval/plate_plate%d.parquet" % p).to_pandas().set_index(["cell_line", "treatment"])
        return cache[p].reindex(columns=genes)
    res = {}
    for mode, sfx in (("fewshot", ""), ("zeroshot", "_zs")):
        recs, groups = [], collections.defaultdict(list)
        for k in range(5, 10):
            z = np.load(OUT + "/pred_clean%s_%d.npz" % (sfx, k)); genes = list(z["genes"])
            for i, (c, t) in enumerate(zip(z["cells"], z["treatments"])):
                if t not in rep:
                    continue
                a = rep[t]; others = [p for p in plates[t] if p != a]
                if not others:
                    continue
                b = others[0]; pa, pb = plate(a, genes), plate(b, genes)
                if (c, t) not in pa.index or (c, t) not in pb.index:
                    continue
                ya = np.nan_to_num(pa.loc[(c, t)].to_numpy(np.float64)); yb = np.nan_to_num(pb.loc[(c, t)].to_numpy(np.float64))
                pm = pa.xs(t, level="treatment").drop(index=c, errors="ignore").fillna(0).to_numpy(np.float64).mean(0)
                preds = {"rhaister": z["D"][i].astype(np.float64), "others_mean": pm}
                rec = dict(split=k, cell=c, treatment=t, plate_a=a, plate_b=b)
                for name, p in preds.items():
                    for w, y in (("same", ya), ("other", yb)):
                        rec["%s|%s|pearson_delta" % (name, w)] = pear(p, y)
                        rec["%s|%s|mse_delta" % (name, w)] = float(np.mean((p - y) ** 2))
                        rec["%s|%s|mae_delta" % (name, w)] = float(np.mean(np.abs(p - y)))
                        for n in (50, 100):
                            rec["%s|%s|top%d" % (name, w, n)] = topn(p, y, n)
                        top = np.argsort(-np.abs(y))[:100]
                        rec["%s|%s|dir_match100" % (name, w)] = float(np.mean(np.sign(p[top]) == np.sign(y[top])))
                recs.append(rec); groups[(k, c)].append((len(recs) - 1, preds, ya, yb))
        # discrimination within line x split
        for (k, c), items in groups.items():
            if len(items) < 3:
                continue
            for name in ("rhaister", "others_mean"):
                P = np.array([it[1][name] for it in items])
                for w, idx in (("same", 2), ("other", 3)):
                    Y = np.array([it[idx] for it in items]); n = len(items)
                    dists = {"L1": np.abs(P[:, None, :] - Y[None]).sum(2), "L2": np.sqrt(((P[:, None, :] - Y[None]) ** 2).sum(2))}
                    Pn = P / np.linalg.norm(P, axis=1, keepdims=True); Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
                    dists["cosine"] = 1 - Pn @ Yn.T
                    for dn, D in dists.items():
                        for j, it in enumerate(items):
                            recs[it[0]]["%s|%s|discrimination_%s" % (name, w, dn)] = 1 - float((D[j] < D[j, j]).sum()) / n
        keys = sorted({kk for r in recs for kk in r if "|" in kk})
        summ = {}
        for subset, f in (("all", lambda r: True), ("plates_6_14", lambda r: {r["plate_a"], r["plate_b"]} == {6, 14}),
                          ("other_pairs", lambda r: {r["plate_a"], r["plate_b"]} != {6, 14})):
            rs = [r for r in recs if f(r)]
            summ[subset] = {"n": len(rs), **{kk: round(float(np.nanmedian([r[kk] for r in rs if kk in r])), 4) for kk in keys}}
        res[mode] = dict(summary=summ, records=recs)
        print(mode, json.dumps(summ["all"]), flush=True)
    json.dump(dict(rules=__doc__, **res), open(OUT + "/metrics_same_other.json", "w"), indent=1)


if __name__ == "__main__":
    main()
