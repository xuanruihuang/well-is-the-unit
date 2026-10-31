"""BioPert reproduction, model and evaluation (samewell/notes.md §3). PyTorch port of BioPert's Julia MLP training
(Models.jl, predict_target_delta_profiles.jl) with their Tahoe hyperparameters (hp_tahoe.tex, row BioPert):
  inputs  PCA-128 of the A549 reference delta and PCA-128 of the target line's mean DMSO profile (sklearn-style PCA,
          centred, fitted on the training-observation matrices)
  MLP     [1024, 128, 1024], ReLU, linear output over 19,020 genes; Glorot-uniform weights, zero biases (Flux default)
  loss    RMSE; AdamW lr 8.91e-4, betas (0.9, 0.999), weight decay 1e-3; batch 256; 200 epochs
  LR      linear warm-up from 1e-8 over 10 epochs, then cosine annealing from lr to lr / 10 over the total step count
  checkpoint  validation average Spearman, evaluated every 10 epochs; best checkpoint used for test
Evaluation:
  standard  test observations: Pearson over all genes between prediction and the well-averaged ground truth, averaged
            over observations (their metric); also copy-A549 and train-mean baselines; Systema-style variant with the
            target line's training centroid subtracted from prediction and truth.
  wells     for drug-doses measured in more than one well whose compound is in the test split: for each target line and
            target well, the A549 reference taken from the same well vs from each other well; ground truth = the target's
            delta in its own well. Pearson over all genes, over genes expressed in the line (DMSO log1p >= 0.2), and
            centred. Ceiling = the target's own delta in the other well.
Usage: bp_train.py <A|B> [seed]. Output $WORKSPACE/rt/samewell/bp/run_<split>_<seed>.json (+ _records.json)."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, json, math, time, collections
import numpy as np
import torch
from scipy.stats import rankdata

BP = (_WS + "/rt/samewell/bp")
HP = dict(lr=8.91e-4, bs=256, wd=1e-3, hidden=[1024, 128, 1024], epochs=200, npca=128)


def pca_fit(X, w, k):
    """PCA of the matrix whose rows are X repeated w times (weights), via the weighted unique rows."""
    w = np.asarray(w, np.float64); mu = (w[:, None] * X).sum(0) / w.sum()
    Z = (X - mu) * np.sqrt(w)[:, None]
    _, S, Vt = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
    V = np.zeros((X.shape[1], k)); V[:, :min(k, Vt.shape[0])] = Vt[:k].T
    return mu.astype(np.float32), V.astype(np.float32)


def pear_rows(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1))


def spear_rows(A, B):
    return pear_rows(np.apply_along_axis(rankdata, 1, A), np.apply_along_axis(rankdata, 1, B))


class MLP(torch.nn.Module):
    def __init__(self, din, hidden, dout):
        super().__init__()
        dims = [din] + hidden; layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(hidden[-1], dout))
        self.net = torch.nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight); torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def main():
    split = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(BP + "/meta.json")); obs = meta["obs"]; lab = np.array(meta["split" + split])
    Y = np.load(BP + "/Y.npy", mmap_mode="r"); refD = np.load(BP + "/refD.npy"); cellrep = np.load(BP + "/cellrep.npy")
    tl = meta["target_lines"]; li = np.array([tl.index(o["line"]) for o in obs]); ri = np.array([o["ref"] for o in obs])
    tr, va, te = (np.where(lab == s)[0] for s in ("train", "val", "test"))
    print("split", split, len(tr), len(va), len(te), "device", dev, flush=True)
    # PCA on training-observation matrices
    ur, cr = np.unique(ri[tr], return_counts=True); mu_r, V_r = pca_fit(refD[ur], cr, HP["npca"])
    ul, cl = np.unique(li[tr], return_counts=True); mu_c, V_c = pca_fit(cellrep[ul], cl, HP["npca"])
    Pr = (refD - mu_r) @ V_r; Pc = (cellrep - mu_c) @ V_c
    X = lambda ix: np.hstack([Pr[ri[ix]], Pc[li[ix]]]).astype(np.float32)
    Xtr = torch.tensor(X(tr), device=dev); Ytr = torch.tensor(np.asarray(Y[tr]), device=dev)
    Xva = torch.tensor(X(va), device=dev); Yva = np.asarray(Y[va])
    model = MLP(Xtr.shape[1], HP["hidden"], Ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], betas=(0.9, 0.999), weight_decay=HP["wd"], eps=1e-8)
    spe = math.ceil(len(tr) / HP["bs"]); total = spe * HP["epochs"]; warm = spe * 10
    def lr_at(step):
        if step < warm:
            return 1e-8 + (HP["lr"] - 1e-8) * step / warm
        t = step - warm
        return HP["lr"] / 10 + (HP["lr"] - HP["lr"] / 10) * (1 + math.cos(math.pi * t / total)) / 2
    def predict(Xt):
        model.eval(); out = []
        with torch.no_grad():
            for b in range(0, Xt.shape[0], 2048):
                out.append(model(Xt[b:b + 2048]).float().cpu().numpy())
        return np.vstack(out)
    best, best_state, best_ep, step, hist = -np.inf, None, None, 0, []
    t0 = time.time()
    for ep in range(1, HP["epochs"] + 1):
        model.train(); perm = torch.randperm(len(tr), device=dev)
        for b in range(0, len(tr), HP["bs"]):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            step += 1
            ix = perm[b:b + HP["bs"]]
            loss = torch.sqrt(torch.mean((model(Xtr[ix]) - Ytr[ix]) ** 2))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0:
            sp = float(np.mean(spear_rows(predict(Xva), Yva)))
            hist.append((ep, round(float(loss), 5), round(sp, 4)))
            print("epoch", ep, "loss", float(loss), "val spearman", sp, "t", round(time.time() - t0), flush=True)
            if sp > best:
                best, best_ep = sp, ep; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # ---- standard evaluation
    Yte = np.asarray(Y[te]); Pte = predict(torch.tensor(X(te), device=dev))
    trmean = np.asarray(Y[tr]).mean(0)
    cent = np.zeros((len(tl), Y.shape[1]), np.float32)
    for l in range(len(tl)):
        m = tr[li[tr] == l]
        if len(m):
            cent[l] = np.asarray(Y[m]).mean(0)
    res = dict(split=split, seed=seed, n=[len(tr), len(va), len(te)], best_epoch=best_ep, val_spearman=round(best, 4), history=hist,
               n_test_compounds=len({obs[i]["smiles"] for i in te}))
    res["test_pearson"] = round(float(np.nanmean(pear_rows(Pte, Yte))), 4)
    res["test_copy"] = round(float(np.nanmean(pear_rows(refD[ri[te]], Yte))), 4)
    res["test_trainmean"] = round(float(np.nanmean(pear_rows(np.broadcast_to(trmean, Yte.shape), Yte))), 4)
    res["test_linemean"] = round(float(np.nanmean(pear_rows(cent[li[te]], Yte))), 4)
    C = cent[li[te]]
    res["test_pearson_centred"] = round(float(np.nanmean(pear_rows(Pte - C, Yte - C))), 4)
    print("STANDARD", json.dumps({k: v for k, v in res.items() if k != "history"}), flush=True)
    # ---- per-well evaluation on replicated drug-doses in the test split
    pk = [tuple(k) for k in meta["perwell_keys"]]; PW = np.load(BP + "/perwell.npy", mmap_mode="r"); pwi = {k: n for n, k in enumerate(pk)}
    lines = meta["lines"]; iref = lines.index("CVCL_0023"); plate = meta["plate"]
    test_smiles = {obs[i]["smiles"] for i in te}
    wells = collections.defaultdict(list)
    for j, (dr, do) in enumerate(zip(meta["drug"], meta["dose"])):
        if not dr.startswith("DMSO"):
            wells[(dr, do)].append(j)
    smi_of = {(o["drug"], str(o["dose"])): o["smiles"] for o in obs}
    expr = cellrep >= 0.2
    recs = []
    for (dr, do), js in wells.items():
        if len(js) < 2 or smi_of.get((dr, do)) not in test_smiles:
            continue
        rw = [j for j in js if (iref, j) in pwi]
        if len(rw) < 2:
            continue
        Rw = {j: np.asarray(PW[pwi[(iref, j)]]) for j in rw}
        for l, ln in enumerate(tl):
            gi = lines.index(ln); tw = [j for j in rw if (gi, j) in pwi]
            if len(tw) < 2:
                continue
            Yw = {j: np.asarray(PW[pwi[(gi, j)]]) for j in tw}
            Xin = np.array([np.r_[(Rw[j] - mu_r) @ V_r, Pc[l]] for j in tw], np.float32)
            Pw = dict(zip(tw, predict(torch.tensor(Xin, device=dev))))
            for a in tw:
                others = [b for b in tw if b != a]
                for gs, ix in (("all", slice(None)), ("expr", np.where(expr[l])[0])):
                    y = Yw[a][ix]; c = cent[l][ix]
                    p_s = lambda u, v: float(pear_rows(u[None], v[None])[0])
                    recs.append(dict(cond="%s_%s" % (dr, do), plates="|".join(sorted({plate[b] for b in tw})), target=ln, well=a, genes=gs,
                                     same=p_s(Pw[a][ix], y), other=float(np.mean([p_s(Pw[b][ix], y) for b in others])),
                                     copy_same=p_s(Rw[a][ix], y), copy_other=float(np.mean([p_s(Rw[b][ix], y) for b in others])),
                                     same_c=p_s(Pw[a][ix] - c, y - c), other_c=float(np.mean([p_s(Pw[b][ix] - c, y - c) for b in others])),
                                     linemean=p_s(c, y), ceiling=float(np.mean([p_s(Yw[b][ix], y) for b in others])),
                                     ref_self=float(np.mean([p_s(Rw[b][ix], Rw[a][ix]) for b in others]))))
    json.dump(recs, open(BP + "/run_%s_%d_records.json" % (split, seed), "w"))
    def summ(rs):
        out = {"n_rec": len(rs), "n_cond": len({r["cond"] for r in rs})}
        for k in ("same", "other", "copy_same", "copy_other", "same_c", "other_c", "linemean", "ceiling"):
            out[k] = round(float(np.nanmedian([r[k] for r in rs])), 3)
            out[k + "_mean"] = round(float(np.nanmean([r[k] for r in rs])), 3)
        byc = collections.defaultdict(list)
        for r in rs:
            byc[r["cond"]].append((r["same"] - r["other"], r["same_c"] - r["other_c"]))
        cm = np.array([np.mean(v, 0) for v in byc.values()]); rng = np.random.default_rng(0)
        bs = np.array([cm[rng.integers(0, len(cm), len(cm))].mean(0) for _ in range(1000)])
        out["diff"] = [round(float(cm[:, 0].mean()), 3)] + [round(float(q), 3) for q in np.quantile(bs[:, 0], [0.025, 0.975])]
        out["diff_centred"] = [round(float(cm[:, 1].mean()), 3)] + [round(float(q), 3) for q in np.quantile(bs[:, 1], [0.025, 0.975])]
        out["same_gt_ceiling"] = round(float(np.mean([r["same"] > r["ceiling"] for r in rs])), 3)
        return out
    for gs in ("all", "expr"):
        rg = [r for r in recs if r["genes"] == gs]
        if not rg:
            continue
        res["wells_" + gs] = summ(rg)
        res["wells_" + gs + "_6_14"] = summ([r for r in rg if r["plates"] == "plate14|plate6"]) if any(r["plates"] == "plate14|plate6" for r in rg) else None
        res["wells_" + gs + "_other_pairs"] = summ([r for r in rg if r["plates"] != "plate14|plate6"]) if any(r["plates"] != "plate14|plate6" for r in rg) else None
        res["wells_" + gs + "_acted"] = summ([r for r in rg if r["ceiling"] >= 0.5]) if any(r["ceiling"] >= 0.5 for r in rg) else None
        print("WELLS", gs, json.dumps(res["wells_" + gs]), flush=True)
    json.dump(res, open(BP + "/run_%s_%d.json" % (split, seed), "w"), indent=1)


if __name__ == "__main__":
    main()
