"""B3 (samewell/notes.md §6, written before running): BioPert's chemical-structure and random-vector representations,
trained as in the original code (Models.jl DoserGate; hp_tahoe.tex), for comparison with the A549-response
representation under same-well and other-well evaluation.
  ecfp    ECFP6 2048-bit fingerprint (RDKit Morgan radius 3) -> PCA 32 -> dose gate s = sigmoid(u*beta + b) - sigmoid(b),
          u = log1p(dose uM), applied to the embedding block; target line = PCA 64 of its mean control profile;
          MLP [1024, 128, 1024]; AdamW lr 7.91e-5, weight decay 0.1, batch 256, 200 epochs (row ECFP6-2048).
  random  one Gaussian vector (256 dims, seed 42) per SMILES -> PCA 32 -> dose gate; target PCA 128; MLP [512, 512, 512];
          lr 2.19e-5, weight decay 1e-4, batch 256, 200 epochs (row Random).
Inputs carry no measurement of the test compound, so the prediction is the same for every well of a condition; it is
scored against the target's response in each well (all genes; and genes expressed in the line), as bp_train.py does.
Usage: bp_molec.py <A|B> <ecfp|random> [seed]. Output $WORKSPACE/rt/samewell/bp/molec_<split>_<rep>_<seed>.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import sys, json, math, time, collections
import numpy as np
import torch
from bp_train import pca_fit, pear_rows, spear_rows, MLP

BP = (_WS + "/rt/samewell/bp")
HP = {"ecfp": dict(lr=7.91e-5, bs=256, wd=0.1, hidden=[1024, 128, 1024], epochs=200, npca_cell=64, npca_mol=32),
      "random": dict(lr=2.19e-5, bs=256, wd=1e-4, hidden=[512, 512, 512], epochs=200, npca_cell=128, npca_mol=32)}


def embed(smiles, rep):
    if rep == "ecfp":
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048); out = {}
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            out[s] = np.array(gen.GetFingerprintAsNumPy(m), np.float32) if m is not None else np.zeros(2048, np.float32)
        return out
    rng = np.random.default_rng(42)
    return {s: rng.standard_normal(256).astype(np.float32) for s in sorted(smiles)}


class Gated(torch.nn.Module):
    def __init__(self, din, hidden, dout, nmol):
        super().__init__(); self.net = MLP(din, hidden, dout); self.nmol = nmol
        self.beta = torch.nn.Parameter(torch.ones(1)); self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        u = x[:, self.nmol:self.nmol + 1]
        s = torch.sigmoid(u * self.beta + self.b) - torch.sigmoid(self.b)
        return self.net(torch.cat([x[:, :self.nmol] * s, x[:, self.nmol:]], 1))


def main():
    split, rep = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42; hp = HP[rep]
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(BP + "/meta.json")); obs = meta["obs"]; lab = np.array(meta["split" + split])
    Y = np.load(BP + "/Y.npy", mmap_mode="r"); cellrep = np.load(BP + "/cellrep.npy"); tl = meta["target_lines"]
    keep = np.array([bool(o["smiles"]) for o in obs])
    li = np.array([tl.index(o["line"]) for o in obs]); dose = np.array([math.log1p(float(o["dose"])) for o in obs], np.float32)
    E = embed({o["smiles"] for o in obs if o["smiles"]}, rep)
    tr, va, te = (np.where((lab == s) & keep)[0] for s in ("train", "val", "test"))
    smi = [o["smiles"] for o in obs]
    us, cs = np.unique([smi[i] for i in tr], return_counts=True)
    mu_m, V_m = pca_fit(np.stack([E[s] for s in us]), cs, hp["npca_mol"])
    ul, cl = np.unique(li[tr], return_counts=True); mu_c, V_c = pca_fit(cellrep[ul], cl, hp["npca_cell"])
    Pc = (cellrep - mu_c) @ V_c
    Pm = {s: (E[s] - mu_m) @ V_m for s in E}
    X = lambda ix: np.hstack([np.stack([Pm[smi[i]] for i in ix]), dose[ix, None], Pc[li[ix]]]).astype(np.float32)
    Xtr = torch.tensor(X(tr), device=dev); Ytr = torch.tensor(np.asarray(Y[tr]), device=dev)
    Xva = torch.tensor(X(va), device=dev); Yva = np.asarray(Y[va])
    model = Gated(Xtr.shape[1], hp["hidden"], Ytr.shape[1], hp["npca_mol"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], betas=(0.9, 0.999), weight_decay=hp["wd"], eps=1e-8)
    spe = math.ceil(len(tr) / hp["bs"]); total = spe * hp["epochs"]; warm = spe * 10
    lr_at = lambda st: 1e-8 + (hp["lr"] - 1e-8) * st / warm if st < warm else hp["lr"] / 10 + 0.9 * hp["lr"] * (1 + math.cos(math.pi * (st - warm) / total)) / 2
    def predict(Xt):
        model.eval(); out = []
        with torch.no_grad():
            for b in range(0, Xt.shape[0], 2048):
                out.append(model(Xt[b:b + 2048]).float().cpu().numpy())
        return np.vstack(out)
    best, best_state, best_ep, step = -np.inf, None, None, 0; t0 = time.time()
    for ep in range(1, hp["epochs"] + 1):
        model.train(); perm = torch.randperm(len(tr), device=dev)
        for b in range(0, len(tr), hp["bs"]):
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
    Yte = np.asarray(Y[te]); Pte = predict(torch.tensor(X(te), device=dev))
    res = dict(split=split, rep=rep, seed=seed, n=[len(tr), len(va), len(te)], best_epoch=best_ep, val_spearman=round(best, 4),
               test_pearson=round(float(np.nanmean(pear_rows(Pte, Yte))), 4))
    print("STANDARD", json.dumps(res), flush=True)
    # per-well scoring on drug-doses with more than one well whose compound is in the test split
    pk = [tuple(k) for k in meta["perwell_keys"]]; PW = np.load(BP + "/perwell.npy", mmap_mode="r"); pwi = {k: n for n, k in enumerate(pk)}
    lines = meta["lines"]; test_smiles = {smi[i] for i in te}
    wells = collections.defaultdict(list)
    for j, (dr, do) in enumerate(zip(meta["drug"], meta["dose"])):
        if not dr.startswith("DMSO"):
            wells[(dr, do)].append(j)
    oi = {(o["line"], o["drug"], str(o["dose"])): i for i, o in enumerate(obs)}
    expr = cellrep >= 0.2; recs = []
    for (dr, do), js in wells.items():
        if len(js) < 2:
            continue
        for l, ln in enumerate(tl):
            i = oi.get((ln, dr, do))
            if i is None or smi[i] not in test_smiles:
                continue
            gi = lines.index(ln); tw = [j for j in js if (gi, j) in pwi]
            if len(tw) < 2:
                continue
            p = predict(torch.tensor(X(np.array([i])), device=dev))[0]
            for a in tw:
                y = np.asarray(PW[pwi[(gi, a)]]); oth = [np.asarray(PW[pwi[(gi, b)]]) for b in tw if b != a]
                for gs, ix in (("all", slice(None)), ("expr", np.where(expr[l])[0])):
                    pr = lambda u, v: float(pear_rows(u[None], v[None])[0])
                    recs.append(dict(cond="%s_%s" % (dr, do), target=ln, genes=gs, acc=pr(p[ix], y[ix]), ceiling=float(np.mean([pr(o[ix], y[ix]) for o in oth])),
                                     plates="|".join(sorted({meta["plate"][b] for b in tw}))))
    for gs in ("all", "expr"):
        rg = [r for r in recs if r["genes"] == gs]
        if rg:
            res["wells_" + gs] = dict(n_rec=len(rg), n_cond=len({r["cond"] for r in rg}), acc_mean=round(float(np.mean([r["acc"] for r in rg])), 3),
                                      acc_median=round(float(np.median([r["acc"] for r in rg])), 3),
                                      acc_mean_acted=round(float(np.mean([r["acc"] for r in rg if r["ceiling"] >= 0.5])), 3),
                                      acc_mean_6_14=round(float(np.mean([r["acc"] for r in rg if r["plates"] == "plate14|plate6"])), 3))
            print("WELLS", gs, json.dumps(res["wells_" + gs]), flush=True)
    json.dump(res, open(BP + "/molec_%s_%s_%d.json" % (split, rep, seed), "w"), indent=1)
    json.dump(recs, open(BP + "/molec_%s_%s_%d_records.json" % (split, rep, seed), "w"))


if __name__ == "__main__":
    main()
