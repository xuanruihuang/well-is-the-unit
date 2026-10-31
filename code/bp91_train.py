"""BioPert (A549-response representation) and its ECFP fingerprint model on the §91 data (samewell/notes.md §91; A100).
Model, hyperparameters and training schedule as bp_train.py (BioPert) and bp_molec.py (ECFP6-2048 row), split C of
bp91_data.py. Standard test score as theirs (Pearson over all genes against the well-averaged truth, mean over
observations). Pair scoring, for every verified pair (a, b) of a test compound in both directions (target well t, other
well o) and every target line measured in both wells:
  bp    same = prediction from A549's delta in t, other = prediction from A549's delta in o, both against the line's
        delta in t; copy_same / copy_other = A549's delta itself; ceiling = the line's delta in o against t;
        linemean = the line's training centroid.
  ecfp  fp = the fingerprint prediction for the condition (no well enters it) against the line's delta in t.
Pearson over the genes expressed in the line (mean DMSO log1p >= 0.2; main) and over all genes (their metric).
Usage: bp91_train.py <dmso|dprime> <bp|ecfp> [seed]. Output $WORKSPACE/rt/samewell/bp91/<ref>/<model>_<seed>.json
(+ _records.json)."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import sys, json, math, time
import numpy as np
import torch
import verified as V
from bp_train import HP as HP_BP, pca_fit, pear_rows, spear_rows, MLP
from bp_molec import HP as HP_MOL, embed, Gated

ROOT = (_WS + "/rt/samewell/bp91/")


def train(model, Xtr, Ytr, Xva, Yva, hp, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], betas=(0.9, 0.999), weight_decay=hp["wd"], eps=1e-8)
    spe = math.ceil(len(Xtr) / hp["bs"]); total = spe * hp["epochs"]; warm = spe * 10
    lr_at = lambda st: 1e-8 + (hp["lr"] - 1e-8) * st / warm if st < warm else hp["lr"] / 10 + 0.9 * hp["lr"] * (1 + math.cos(math.pi * (st - warm) / total)) / 2
    def predict(Xt):
        model.eval(); out = []
        with torch.no_grad():
            for b in range(0, Xt.shape[0], 2048):
                out.append(model(Xt[b:b + 2048]).float().cpu().numpy())
        return np.vstack(out)
    best, best_state, best_ep, step, t0 = -np.inf, None, None, 0, time.time()
    for ep in range(1, hp["epochs"] + 1):
        model.train(); perm = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(Xtr), hp["bs"]):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            step += 1; ix = perm[b:b + hp["bs"]]
            loss = torch.sqrt(torch.mean((model(Xtr[ix]) - Ytr[ix]) ** 2)); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0:
            sp = float(np.mean(spear_rows(predict(Xva), Yva)))
            print("epoch", ep, "val spearman", round(sp, 4), "t", round(time.time() - t0), flush=True)
            if sp > best:
                best, best_ep = sp, ep; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict, best_ep, best


def main():
    ref, kind = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42; BP = ROOT + ref
    torch.manual_seed(seed); np.random.seed(seed); dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(BP + "/meta.json")); obs = meta["obs"]; lab = np.array(meta["splitC"])
    Y = np.load(BP + "/Y.npy", mmap_mode="r"); refD = np.load(BP + "/refD.npy"); cellrep = np.load(BP + "/cellrep.npy")
    tl = meta["target_lines"]; lines = meta["lines"]; li = np.array([tl.index(o["line"]) for o in obs])
    smi = [o["smiles"] for o in obs]; keep = np.array([bool(s) for s in smi]) if kind == "ecfp" else np.ones(len(obs), bool)
    tr, va, te = (np.where((lab == s) & keep)[0] for s in ("train", "val", "test"))
    print(ref, kind, seed, "split", len(tr), len(va), len(te), dev, flush=True)
    if kind == "bp":
        hp = HP_BP; ri = np.array([o["ref"] for o in obs])
        ur, cr = np.unique(ri[tr], return_counts=True); mu_r, V_r = pca_fit(refD[ur], cr, hp["npca"])
        ul, cl = np.unique(li[tr], return_counts=True); mu_c, V_c = pca_fit(cellrep[ul], cl, hp["npca"])
        Pr = (refD - mu_r) @ V_r; Pc = (cellrep - mu_c) @ V_c
        X = lambda ix: np.hstack([Pr[ri[ix]], Pc[li[ix]]]).astype(np.float32)
        model = MLP(2 * hp["npca"], hp["hidden"], Y.shape[1]).to(dev)
    else:
        hp = HP_MOL["ecfp"]; dose = np.array([math.log1p(float(o["dose"])) for o in obs], np.float32)
        E = embed({s for s in smi if s}, "ecfp"); us, cs = np.unique([smi[i] for i in tr], return_counts=True)
        mu_m, V_m = pca_fit(np.stack([E[s] for s in us]), cs, hp["npca_mol"])
        ul, cl = np.unique(li[tr], return_counts=True); mu_c, V_c = pca_fit(cellrep[ul], cl, hp["npca_cell"])
        Pc = (cellrep - mu_c) @ V_c; Pm = {s: (E[s] - mu_m) @ V_m for s in E}
        X = lambda ix: np.hstack([np.stack([Pm[smi[i]] for i in ix]), dose[ix, None], Pc[li[ix]]]).astype(np.float32)
        model = Gated(hp["npca_mol"] + 1 + hp["npca_cell"], hp["hidden"], Y.shape[1], hp["npca_mol"]).to(dev)
    T = lambda a: torch.tensor(a, device=dev)
    predict, best_ep, best = train(model, T(X(tr)), T(np.asarray(Y[tr])), T(X(va)), np.asarray(Y[va]), hp, dev)
    Yte = np.asarray(Y[te]); Pte = predict(T(X(te)))
    res = dict(rules=__doc__, ref=ref, model=kind, seed=seed, n=[len(tr), len(va), len(te)], best_epoch=best_ep, val_spearman=round(best, 4),
               n_test_compounds=len({obs[i]["comp"] for i in te}), test_pearson=round(float(np.nanmean(pear_rows(Pte, Yte))), 4))
    cent = np.zeros((len(tl), Y.shape[1]), np.float32)
    for l in range(len(tl)):
        m = tr[li[tr] == l]
        if len(m):
            cent[l] = np.asarray(Y[m]).mean(0)
    print("STANDARD", json.dumps({k: v for k, v in res.items() if k != "rules"}), flush=True)
    # pair scoring
    PW = np.load(BP + "/perwell.npy", mmap_mode="r"); pwi = {tuple(k): n for n, k in enumerate(meta["perwell_keys"])}
    W = lambda i, j: np.asarray(PW[pwi[(i, j)]], np.float64)
    iref = lines.index("CVCL_0023"); expr = cellrep >= 0.2; oi = {(o["line"], o["cond"]): i for i, o in enumerate(obs)}
    condof = lambda j: "%s_%s" % (V.norm(meta["drug"][j]), meta["dose"][j])
    pc = lambda u, v: float(pear_rows(u[None], v[None])[0]); cache = {}; recs = []
    def pred(l, j):
        if (l, j) not in cache:
            if kind == "bp":
                x = np.r_[(W(iref, j).astype(np.float32) - mu_r) @ V_r, Pc[l]][None].astype(np.float32)
            else:
                x = X(np.array([oi[(tl[l], condof(j))]]))
            cache[(l, j)] = predict(T(x))[0].astype(np.float64)
        return cache[(l, j)]
    for n, p in enumerate(meta["pairs"]):
        if kind == "bp" and not ((iref, p["a"]) in pwi and (iref, p["b"]) in pwi):
            continue
        for t, o in ((p["a"], p["b"]), (p["b"], p["a"])):
            for l, ln in enumerate(tl):
                gi = lines.index(ln)
                if (gi, t) not in pwi or (gi, o) not in pwi or (ln, condof(t)) not in oi:
                    continue
                yt, yo = W(gi, t), W(gi, o); ps = pred(l, t)
                if kind == "bp":
                    po = pred(l, o); at, ao = W(iref, t), W(iref, o)
                for gs, ix in (("expr", np.where(expr[l])[0]), ("all", slice(None))):
                    r = dict(pair=n, t=int(t), o=int(o), target=ln, genes=gs, group=p["group"], comp=p["comp"], cls84=p["cls84"], cls85=p["cls85"])
                    if kind == "bp":
                        r.update(same=pc(ps[ix], yt[ix]), other=pc(po[ix], yt[ix]), copy_same=pc(at[ix], yt[ix]), copy_other=pc(ao[ix], yt[ix]),
                                 ceiling=pc(yo[ix], yt[ix]), linemean=pc(cent[l][ix], yt[ix]))
                    else:
                        r.update(fp=pc(ps[ix], yt[ix]))
                    recs.append(r)
    print("pair records", len(recs), flush=True)
    out = ROOT + "%s/%s_%d" % (ref, kind, seed)
    json.dump(res, open(out + ".json", "w"), indent=1); json.dump(recs, open(out + "_records.json", "w"))
    print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
