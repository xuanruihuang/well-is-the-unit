"""Rhaister (Tahoe Therapeutics; Hugging Face tahoebio/Rhaister) under same-well and other-well evaluation
(samewell/notes.md §12; A100). Rhaister's loader averages a (cell line, drug-dose) over the plates it was measured on, so
both wells of a replicated drug-dose enter training and the test truth. The clean variant keeps, for every drug-dose
measured on more than one plate, only its training plate a (plate 6 for plates 6/14; plate 1 for adagrasib; otherwise
the lowest-numbered plate other than 13) in Rhaister's inputs (pdex and cell_eval); the held-out lines are then scored
against their delta on plate a (same well, what the unchanged pipeline reports) and on each other plate b (never in
training). The model code and hyperparameters are Rhaister's own, called through rhaister.train.train_and_evaluate.
  build              write data_clean/tahoe/ (pdex restricted to the 2,000 static genes, row groups per (cell line,
                     plate); cell_eval per plate; replicated drug-doses kept on plate a only); zeroshot/ and
                     definition/ linked from the original download
  run <orig|clean> <k> [zeroshot]   train and evaluate split tahoe/<k>_holdout (k = 5..9) on the original or clean
                     data; saves Rhaister's metrics and the test predictions (fold changes, deltas)
  eval               per held-out line x replicated test drug-dose: Pearson over the static genes of the predicted delta
                     with the line's delta on plate a and on each other plate b; replicate agreement r (a vs b);
                     share above sqrt(max(r, 0)); same for the training lines' mean on plate a (a perturbation-mean
                     baseline); the line's mean delta over its other treatments on plate a (no drug information;
                     carries the line x plate part, e.g. the error of the plate's control reference); the same scores
                     with that line x plate offset removed from each truth (truth a minus the line's mean over its
                     other treatments on a, truth b likewise on b; prediction untouched, §12b); line-specific
                     version (minus the mean of the other lines on the same plate)
Paths: data $WORKSPACE/rt/samewell/rhaister/{data,data_clean}; caches and outputs $WORKSPACE/rt/samewell/rhaister/out.
Usage: rh_clean.py build | run <orig|clean> <k> [zeroshot] | eval [zeroshot]."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, json, glob, re, time
import numpy as np

RH = (_WS + "/rt/samewell/rhaister")
ORIG, CLEAN, OUT = RH + "/data", RH + "/data_clean", RH + "/out"


def tplates(ce_files):
    """treatment -> set of plates, from the per-plate cell_eval files."""
    import pyarrow.parquet as pq
    tp = {}
    for f in ce_files:
        p = int(re.search(r"plate(\d+)\.parquet$", f).group(1))
        for t in set(pq.read_table(f, columns=["treatment"]).column(0).to_pylist()):
            tp.setdefault(t, set()).add(p)
    return tp


def train_plate(t, ps):
    if {6, 14} <= ps:
        return 6
    if t.startswith("[('Adagrasib', 0.05") and 1 in ps:
        return 1
    rest = sorted(ps - {13})
    return rest[0] if rest else min(ps)


def build():
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    ce_files = sorted(glob.glob(ORIG + "/cell_eval/plate_plate*.parquet"))
    tp = tplates(ce_files)
    rep = {t: train_plate(t, ps) for t, ps in tp.items() if len(ps) >= 2}
    json.dump({"replicated": {t: sorted(tp[t]) for t in rep}, "train_plate": rep}, open(RH + "/replicated_treatments.json", "w"), indent=1)
    print("treatments", len(tp), "replicated across plates", len(rep), flush=True)
    os.makedirs(CLEAN + "/tahoe/cell_eval", exist_ok=True); os.makedirs(CLEAN + "/tahoe/pdex", exist_ok=True)
    for d in ("zeroshot", "definition"):
        if not os.path.exists(CLEAN + "/tahoe/" + d):
            os.symlink(ORIG + "/" + d, CLEAN + "/tahoe/" + d)
    for f in ce_files:
        p = int(re.search(r"plate(\d+)\.parquet$", f).group(1))
        t = pq.read_table(f)
        keep = [not (tr in rep and rep[tr] != p) for tr in t.column("treatment").to_pylist()]
        pq.write_table(t.filter(pa.array(keep)), CLEAN + "/tahoe/cell_eval/" + os.path.basename(f))
        print("cell_eval plate", p, t.num_rows, "->", sum(keep), flush=True)
    genes = json.load(open(ORIG + "/definition/static_2k_genes.json"))
    pf = pq.ParquetFile(ORIG + "/pdex/all_plates_pdex.parquet")
    gset = pa.array(genes); parts, kept, total = [], 0, 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g)
        t = t.filter(pc.is_in(t.column("feature"), value_set=gset))
        total += t.num_rows
        tgt, pl = t.column("target").to_pylist(), t.column("plate").to_pylist()
        t = t.filter(pa.array([not (x in rep and rep[x] != p) for x, p in zip(tgt, pl)]))
        kept += t.num_rows; parts.append(t)
        if g % 200 == 0:
            print("pdex row group", g, "of", pf.num_row_groups, "kept", kept, "of", total, flush=True)
    t = pa.concat_tables(parts); del parts
    t = t.sort_by([("cell_line", "ascending"), ("plate", "ascending")])
    cl, pl = t.column("cell_line").to_pylist(), t.column("plate").to_pylist()
    writer = pq.ParquetWriter(CLEAN + "/tahoe/pdex/all_plates_pdex.parquet", t.schema)
    start = 0
    for i in range(1, len(cl) + 1):                       # one row group per (cell line, plate), as Rhaister's zeroshot scan expects
        if i == len(cl) or (cl[i], pl[i]) != (cl[start], pl[start]):
            writer.write_table(t.slice(start, i - start), row_group_size=i - start); start = i
    writer.close()
    print("pdex static-gene rows", total, "kept", kept, flush=True)


def run(mode, k, zeroshot):
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["RHAISTER_DATA_ROOT"] = ORIG if mode == "orig" else CLEAN
    import rhaister.prepare_combined as pcb
    pcb.CACHE_DIR = OUT + "/cache_" + mode
    from rhaister.train import train_and_evaluate
    split = "tahoe/%d_holdout" % k
    t0 = time.time()
    data = pcb.prepare_all(split, zeroshot=zeroshot)
    cap, ev0 = {}, data["evaluate_test"]

    def ev(Y_pred_fc, D_pred=None, F_pred=None, P_pred=None, compute_discrimination=False):
        cap.update(Y=np.asarray(Y_pred_fc, np.float32), D=None if D_pred is None else np.asarray(D_pred, np.float32))
        return ev0(Y_pred_fc, D_pred=D_pred, F_pred=F_pred, P_pred=P_pred, compute_discrimination=compute_discrimination)
    data["evaluate_test"] = ev
    m = train_and_evaluate(experiment_name="samewell_%s_%d" % (mode, k), split_name=split, log=False, data=data, zeroshot=zeroshot)
    tag = "%s%s_%d" % (mode, "_zs" if zeroshot else "", k)
    os.makedirs(OUT, exist_ok=True)
    np.savez(OUT + "/pred_%s.npz" % tag, Y=cap["Y"], D=cap["D"], cells=np.asarray(data["test_cells"]).astype(str),
             treatments=np.asarray(data["test_treatments"]).astype(str), genes=np.asarray(data["gene_cols"]).astype(str),
             D_test=np.load(os.path.join(pcb._cache_path("tahoe", "%d_holdout" % k), "D_test.npy")).astype(np.float32))
    json.dump(dict(mode=mode, split=split, zeroshot=zeroshot, seconds=round(time.time() - t0), metrics={kk: (float(v) if hasattr(v, "__float__") else v) for kk, v in m.items()}),
              open(OUT + "/metrics_%s.json" % tag, "w"), indent=1)
    print("DONE", tag, "state/pearson_delta_mean", m.get("state/pearson_delta_mean"), flush=True)


def pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def evaluate(zeroshot):
    import pyarrow.parquet as pq
    info = json.load(open(RH + "/replicated_treatments.json")); rep = info["train_plate"]
    recs = []
    sfx = "_zs" if zeroshot else ""
    ce_cache = {}

    def plate_df(p, genes):
        if p not in ce_cache:
            t = pq.read_table(ORIG + "/cell_eval/plate_plate%d.parquet" % p).to_pandas()
            ce_cache[p] = t.set_index(["cell_line", "treatment"])
        return ce_cache[p].reindex(columns=genes)
    for k in range(5, 10):
        f = OUT + "/pred_clean%s_%d.npz" % (sfx, k)
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=False); genes = list(z["genes"])
        for i, (c, t) in enumerate(zip(z["cells"], z["treatments"])):
            if t not in rep:
                continue
            a = rep[t]; pa_ = plate_df(a, genes)
            if (c, t) not in pa_.index:
                continue
            ya = np.nan_to_num(pa_.loc[(c, t)].to_numpy(np.float64))
            assert np.allclose(ya, z["D_test"][i], atol=1e-5), (c, t)
            yhat = z["D"][i].astype(np.float64)
            others_a = pa_.xs(t, level="treatment").drop(index=c, errors="ignore").fillna(0).to_numpy(np.float64)
            pm = others_a.mean(0)                                   # all other lines on plate a (training lines + other held-out)
            lpa = pa_.xs(c, level="cell_line").drop(index=t, errors="ignore").fillna(0).to_numpy(np.float64).mean(0)   # line's mean over its other treatments on plate a
            for b in info["replicated"][t]:
                if b == a:
                    continue
                pb = plate_df(b, genes)
                if (c, t) not in pb.index:
                    continue
                yb = np.nan_to_num(pb.loc[(c, t)].to_numpy(np.float64))
                others_b = pb.xs(t, level="treatment").drop(index=c, errors="ignore").fillna(0).to_numpy(np.float64)
                r = pear(ya, yb)
                lpb = pb.xs(c, level="cell_line").drop(index=t, errors="ignore").fillna(0).to_numpy(np.float64).mean(0)
                ya_c, yb_c = ya - lpa, yb - lpb                       # truths without the line x plate offset (prediction untouched)
                ls = lambda y, o: y - o.mean(0)
                recs.append(dict(split=k, cell=c, treatment=t, plate_a=a, plate_b=b, r=r,
                                 rh_same=pear(yhat, ya), rh_other=pear(yhat, yb), pm_same=pear(pm, ya), pm_other=pear(pm, yb),
                                 lp_same=pear(lpa, ya), lp_other=pear(lpa, yb),
                                 r_c=pear(ya_c, yb_c), rh_same_c=pear(yhat, ya_c), rh_other_c=pear(yhat, yb_c),
                                 pm_same_c=pear(pm, ya_c), pm_other_c=pear(pm, yb_c),
                                 ls_rh_same=pear(yhat - pm, ls(ya, others_a)), ls_rh_other=pear(yhat - pm, ls(yb, others_b)),
                                 ls_ceiling=pear(ls(ya, others_a), ls(yb, others_b))))
    out = {"n_records": len(recs), "n_conditions": len({(x["split"], x["treatment"]) for x in recs})}
    def summ(rs):
        s = {"n": len(rs), "n_treatments": len({x["treatment"] for x in rs})}
        for key in ("r", "rh_same", "rh_other", "pm_same", "pm_other", "lp_same", "lp_other", "ls_rh_same", "ls_rh_other", "ls_ceiling",
                    "r_c", "rh_same_c", "rh_other_c", "pm_same_c", "pm_other_c"):
            s[key + "_median"] = round(float(np.nanmedian([x[key] for x in rs])), 4)
            s[key + "_mean"] = round(float(np.nanmean([x[key] for x in rs])), 4)
        bound = lambda x: np.sqrt(max(x["r"], 0))
        for key in ("rh_same", "rh_other", "pm_same", "pm_other"):
            s[key + "_above_bound"] = round(float(np.mean([x[key] > bound(x) for x in rs])), 4)
        for key in ("rh_same_c", "rh_other_c"):
            s[key + "_above_bound"] = round(float(np.mean([x[key] > np.sqrt(max(x["r_c"], 0)) for x in rs])), 4)
        return s
    out["all"] = summ(recs)
    out["plates_6_14"] = summ([x for x in recs if {x["plate_a"], x["plate_b"]} == {6, 14}])
    out["other_pairs"] = summ([x for x in recs if {x["plate_a"], x["plate_b"]} != {6, 14}])
    out["acted_r_ge_0.5"] = summ([x for x in recs if x["r"] >= 0.5])
    json.dump(dict(summary=out, records=recs), open(OUT + "/eval_clean%s.json" % sfx, "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]), len(sys.argv) > 4 and sys.argv[4] == "zeroshot")
    else:
        evaluate(len(sys.argv) > 2 and sys.argv[2] == "zeroshot")
