"""BioPert reproduction, data (samewell/notes.md §3). Rebuilds BioPert's Tahoe inputs from the raw-count pseudobulks of
pb_extract.py, following their preprocess_tahoe.jl / Observations.jl:
  pseudobulk per (line, well) = summed raw counts over 19,020 protein-coding genes (both cell halves); keep >= 50 cells and
  >= 35,000 coding UMIs; drop lines with > 90% failing; x = log1p(1e4 * c / sum c).
  delta = treated x - mean of the (line, plate) DMSO profiles.
  observation = (target line != A549, drug, SMILES, dose); ground truth = delta averaged over all wells of that
  drug-dose; reference = A549 delta of the same drug-dose, one well drawn at random (seed 42) and fixed; target-line
  representation = mean of all the line's DMSO profiles.
Split A: their Butina split (ECFP6 2048 bits, Tanimoto distance cutoff 0.4, seed 42, train / val / test = 0.75 / 0.10 /
0.15) re-run with their code on our SMILES list (order of compounds differs from theirs, so the compounds differ).
Split B: every compound with more than one well (salt-stripped name) goes to test; val = split A's val minus those.
Also saved: per-well deltas of the target lines and A549 for drug-doses with more than one well (for same-well vs
other-well testing). Output $WORKSPACE/rt/samewell/bp/. Usage: bp_data.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, json, ast, re, random, collections
import numpy as np
import pyarrow.parquet as pq

PB = (_WS + "/rt/samewell/pb")
META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata")
OUT = (_WS + "/rt/samewell/bp")
REF = "CVCL_0023"
norm = lambda s: re.sub(r"\s*\(.*\)|\s+(citrate|hydrochloride|maleate|mesylate|sulfate|hcl)$", "", s, flags=re.I).strip().lower()
FIX = {"Sacubitril/Valsartan": "CCCCC(=O)N(CC1=CC=C(C=C1)C2=CC=CC=C2C3=NNN=N3)[C@@H](C(C)C)C(=O)O.CCOC(=O)[C@H](C)C[C@@H](CC1=CC=C(C=C1)C2=CC=CC=C2)NC(=O)CCC(=O)O",
       "Verteporfin": "CC1=C(C2=CC3=C(C(=C(N3)C=C4[C@]5([C@H](C(=CC=C5C(=CC6=C(C(=C(N6)C=C1N2)C=C)C)N4)C(=O)OC)C(=O)OC)C)C)CCC(=O)OC)CCC(=O)O.CC1=C(C2=CC3=C(C(=C(N3)C=C4[C@]5([C@H](C(=CC=C5C(=CC6=C(C(=C(N6)C=C1N2)C=C)C)N4)C(=O)O)C(=O)OC)C)C)CCC(=O)O)CCC(=O)OC"}


def butina_split(unique_smiles, cutoff=0.4, val_frac=0.1, test_frac=0.15, seed=42):
    """Verbatim logic of BioPert python/src/chem_utils.py butina_split."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    train_frac = 1.0 - val_frac - test_frac
    us = list(unique_smiles); random.seed(seed); random.shuffle(us)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fps, valid = [], []
    for s in us:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(gen.GetFingerprint(m)); valid.append(s)
    n = len(fps); dists = []
    for i in range(1, n):
        dists.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    tr, va, te = [], [], []
    for cl in clusters:
        seen = len(tr) + len(va) + len(te) + len(cl)
        if len(tr) < train_frac * seen:
            tr.extend(cl)
        elif len(va) < val_frac * seen:
            va.extend(cl)
        else:
            te.extend(cl)
    return {valid[i] for i in tr}, {valid[i] for i in va}, {valid[i] for i in te}, len(clusters)


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(PB + "/index.json")); lines, samples = idx["lines"], idx["samples"]; L, P = len(lines), len(samples)
    G = len(idx["genes"])
    cnt = np.load(PB + "/cnt_lsh.npy").reshape(L, P, 2, 4).sum(2)              # cells, coding, all, mito
    Xc = np.load(PB + "/counts_lsh.npy", mmap_mode="r")
    sm = pq.read_table(META + "/sample_metadata.parquet").to_pydict()
    info = {s: (d.strip(), ast.literal_eval(dc)[0][1], p) for s, d, dc, p in zip(sm["sample"], sm["drug"], sm["drugname_drugconc"], sm["plate"])}
    dm = pq.read_table(META + "/drug_metadata.parquet").to_pydict()
    smi = {d.strip(): (s if s else FIX.get(d.strip(), "")) for d, s in zip(dm["drug"], dm["canonical_smiles"])}
    drug = np.array([info[s][0] for s in samples]); dose = np.array([info[s][1] for s in samples]); plate = np.array([info[s][2] for s in samples])
    isd = np.char.startswith(drug.astype(str), "DMSO")
    exists = cnt[:, :, 0] > 0
    qc = (cnt[:, :, 0] >= 50) & (cnt[:, :, 1] >= 35000)
    fail = 1 - (qc & exists).sum(1) / np.maximum(exists.sum(1), 1)
    keep_l = [i for i in range(L) if fail[i] <= 0.9 and exists[i].any()]
    print("lines kept", len(keep_l), "removed", [lines[i] for i in range(L) if i not in keep_l], flush=True)
    # log-normalised profiles for passing pseudobulks (line, sample)
    def prof(i, j):
        c = (np.asarray(Xc[(i * P + j) * 2], np.float64) + np.asarray(Xc[(i * P + j) * 2 + 1], np.float64))
        return np.log1p(1e4 * c / c.sum()).astype(np.float32)
    dmso_lp, dmso_all = {}, {}
    for i in keep_l:
        allp = []
        for pl in sorted(set(plate)):
            js = [j for j in np.where(isd & (plate == pl))[0] if qc[i, j]]
            if js:
                ps = [prof(i, j) for j in js]; dmso_lp[(i, pl)] = np.mean(ps, 0); allp += ps
        dmso_all[i] = np.mean(allp, 0)
    print("dmso done", flush=True)
    # per-well deltas
    wells = collections.defaultdict(list)                       # (drug, dose) -> sample idx
    for j in np.where(~isd)[0]:
        wells[(drug[j], dose[j])].append(j)
    ir = lines.index(REF)
    def delta(i, j):
        if not qc[i, j] or (i, plate[j]) not in dmso_lp:
            return None
        return prof(i, j) - dmso_lp[(i, plate[j])]
    tl = [i for i in keep_l if i != ir]
    obs = []; Y = []; ref_rows = []; refD = []; ref_key = {}
    rng = np.random.default_rng(42)
    rep_keys = [k for k, v in wells.items() if len(v) >= 2]
    perwell = {}                                                  # (line, sample) -> delta, replicated drug-doses only
    for k, js in sorted(wells.items()):
        rd = [(j, delta(ir, j)) for j in js]; rd = [(j, x) for j, x in rd if x is not None]
        if not rd:
            continue
        pick = rd[int(rng.integers(len(rd)))]
        ref_key[k] = len(refD); refD.append(pick[1]); ref_rows.append(int(pick[0]))
        if len(js) >= 2:
            for j, x in rd:
                perwell[(ir, int(j))] = x
        for i in tl:
            ds = [(j, delta(i, j)) for j in js]; ds = [(j, x) for j, x in ds if x is not None]
            if not ds:
                continue
            obs.append(dict(line=lines[i], li=i, drug=k[0], dose=k[1], smiles=smi.get(k[0], ""), wells=[int(j) for j, _ in ds], ref=ref_key[k]))
            Y.append(np.mean([x for _, x in ds], 0))
            if len(js) >= 2:
                for j, x in ds:
                    perwell[(i, int(j))] = x
    Y = np.array(Y, np.float32); refD = np.array(refD, np.float32)
    print("observations", len(obs), "reference drug-doses", len(refD), "per-well entries", len(perwell), flush=True)
    cellrep = np.array([dmso_all[i] for i in tl], np.float32)
    # splits
    usmi = sorted({o["smiles"] for o in obs if o["smiles"]})
    tr, va, te, ncl = butina_split(usmi)
    rep_names = {norm(k[0]) for k in rep_keys}
    rep_smi = {o["smiles"] for o in obs if norm(o["drug"]) in rep_names}
    splitA = ["train" if o["smiles"] in tr else "val" if o["smiles"] in va else "test" if o["smiles"] in te else "none" for o in obs]
    splitB = ["test" if o["smiles"] in rep_smi else ("val" if o["smiles"] in va else "train") for o in obs]
    print("clusters", ncl, "split A", collections.Counter(splitA), "compounds", len(tr), len(va), len(te), flush=True)
    print("split B", collections.Counter(splitB), "replicated compounds", len(rep_smi), flush=True)
    print("split A test compounds with >1 well:", len(te & rep_smi), flush=True)
    np.save(OUT + "/Y.npy", Y); np.save(OUT + "/refD.npy", refD); np.save(OUT + "/cellrep.npy", cellrep)
    pk = sorted(perwell); np.save(OUT + "/perwell.npy", np.array([perwell[k] for k in pk], np.float32))
    json.dump(dict(obs=obs, splitA=splitA, splitB=splitB, target_lines=[lines[i] for i in tl], ref_rows=ref_rows,
                   ref_keys=[list(k) for k in ref_key], perwell_keys=[list(k) for k in pk], samples=samples,
                   plate=list(plate), drug=list(drug), dose=[str(x) for x in dose], rep_keys=[list(k) for k in rep_keys],
                   cells=cnt[:, :, 0].tolist(), coding_umi=cnt[:, :, 1].tolist(), lines=lines, qc_pass=qc.tolist()),
              open(OUT + "/meta.json", "w"))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
