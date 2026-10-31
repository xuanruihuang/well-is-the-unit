"""BioPert inputs rebuilt for verified replicate pairs under two references (samewell/notes.md §91; A100).
As bp_data.py (pseudobulk of raw counts over 19,020 coding genes, >= 50 cells and >= 35,000 coding UMIs, the same 48
lines, x = log1p(CP10K)), with these changes:
  - plate-13 wells are dropped everywhere (§81);
  - conditions are keyed by the salt-stripped name (verified.norm) and dose, so salt-name replicates are one condition;
  - two references: 'dmso' = x minus the mean of the (line, plate) DMSO profiles (BioPert's); 'dprime' = x minus the
    mean of the line's other QC-passing treated wells on the same plate (D', the well left out);
  - split C: every compound with a replicated drug-dose (any verified.pairs pair) goes to test with all its doses,
    and so do its stereoisomers and other salts (same structure without stereochemistry, largest fragment);
    val = the Butina val compounds (bp_data.butina_split on this SMILES list) minus those; the rest train.
Observation = (target line != A549, condition); training truth = delta averaged over the condition's wells; A549
reference = one well drawn at random (seed 42) and fixed; target-line representation = mean of its DMSO profiles.
Saved per reference: Y.npy, refD.npy, perwell.npy (every line's delta in every well of a verified pair) and a shared
meta.json with the pairs (pb sample indices), their group (verified.pairs) and label classes (labels84, labels85).
Output $WORKSPACE/rt/samewell/bp91/<dmso|dprime>/. Usage: bp91_data.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, json, ast, collections
import numpy as np
import pyarrow.parquet as pq
import verified as V
from bp_data import butina_split, FIX

PB = (_WS + "/rt/samewell/pb"); META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata")
OUT = (_WS + "/rt/samewell/bp91"); SW = (_WS + "/rt/samewell/")
SCO = (_WS + "/rt/tahoe/tahoe_scores.npz"); REF = "CVCL_0023"


def light_d():
    """The condition list of sw_core.load without the expression arrays (same order, so labels84 indices apply)."""
    s = np.load(SCO, allow_pickle=True); qc = json.load(open((_WS + "/rt/tahoe/sample_qc.json")))
    drug, conc, plate = (np.array([str(x) for x in s[k]]) for k in ("drug", "conc", "plate"))
    return dict(drug=drug, conc=conc, plate=plate, sample=np.array([c["sample"] for c in qc["conditions"]]))


def main():
    for r in ("dmso", "dprime"):
        os.makedirs(OUT + "/" + r, exist_ok=True)
    idx = json.load(open(PB + "/index.json")); lines, samples = idx["lines"], idx["samples"]; L, P = len(lines), len(samples)
    cnt = np.load(PB + "/cnt_lsh.npy").reshape(L, P, 2, 4).sum(2)
    Xc = np.load(PB + "/counts_lsh.npy", mmap_mode="r")
    sm = pq.read_table(META + "/sample_metadata.parquet").to_pydict()
    info = {s: (d.strip(), ast.literal_eval(dc)[0][1], p) for s, d, dc, p in zip(sm["sample"], sm["drug"], sm["drugname_drugconc"], sm["plate"])}
    dm = pq.read_table(META + "/drug_metadata.parquet").to_pydict()
    smi = {d.strip(): (s if s else FIX.get(d.strip(), "")) for d, s in zip(dm["drug"], dm["canonical_smiles"])}
    drug = [info[s][0] for s in samples]; dose = [info[s][1] for s in samples]; plate = [info[s][2] for s in samples]
    isd = np.array([x.startswith("DMSO") for x in drug]); use = np.array([p != "plate13" for p in plate])
    exists = cnt[:, :, 0] > 0; qc = (cnt[:, :, 0] >= 50) & (cnt[:, :, 1] >= 35000)
    fail = 1 - (qc & exists).sum(1) / np.maximum(exists.sum(1), 1)
    keep_l = [i for i in range(L) if fail[i] <= 0.9 and exists[i].any()]
    print("lines kept", len(keep_l), flush=True)
    def prof(i, j):
        c = np.asarray(Xc[(i * P + j) * 2], np.float64) + np.asarray(Xc[(i * P + j) * 2 + 1], np.float64)
        return np.log1p(1e4 * c / c.sum()).astype(np.float32)
    plates = sorted({p for p in plate if p != "plate13"})
    treated = [j for j in range(P) if use[j] and not isd[j]]
    Xw = {}; dmso_lp = {}; dmso_all = {}; tsum = {}; tn = {}
    for i in keep_l:
        allp = []
        for pl in plates:
            js = [j for j in range(P) if isd[j] and plate[j] == pl and qc[i, j]]
            if js:
                ps = [prof(i, j) for j in js]; dmso_lp[(i, pl)] = np.mean(ps, 0); allp += ps
            ts = [j for j in treated if plate[j] == pl and qc[i, j]]
            for j in ts:
                Xw[(i, j)] = prof(i, j)
            if ts:
                tsum[(i, pl)] = np.sum([Xw[(i, j)] for j in ts], 0, dtype=np.float64); tn[(i, pl)] = len(ts)
        dmso_all[i] = np.mean(allp, 0)
        print("line", lines[i], "wells", sum(1 for k in Xw if k[0] == i), flush=True)
    def delta(i, j, ref):
        """None unless both references exist, so the two data sets hold the same observations."""
        if (i, j) not in Xw or (i, plate[j]) not in dmso_lp or tn[(i, plate[j])] < 2:
            return None
        if ref == "dmso":
            return Xw[(i, j)] - dmso_lp[(i, plate[j])]
        return (Xw[(i, j)] - (tsum[(i, plate[j])] - Xw[(i, j)]) / (tn[(i, plate[j])] - 1)).astype(np.float32)
    # verified pairs in pb sample indices, with label classes
    d = light_d(); si = {s: k for k, s in enumerate(samples)}
    lab = {}
    for fn in ("labels84", "labels85"):
        for x in json.load(open(SW + fn + ".json"))["pairs"]:
            lab[(fn, min(x["a"], x["b"]), max(x["a"], x["b"]))] = x["cls"]
    pairs = []
    for e in V.pairs(d):
        a, b = min(e["a"], e["b"]), max(e["a"], e["b"])
        pairs.append(dict(a=si[str(d["sample"][a])], b=si[str(d["sample"][b])], group=e["group"], comp=e["comp"], cond=e["cond"], salt=e["salt"],
                          cls84=lab.get(("labels84", a, b)), cls85=lab.get(("labels85", a, b))))
    test_comp = {p["comp"] for p in pairs}; pw_wells = sorted({p[k] for p in pairs for k in ("a", "b")})
    print("pairs", len(pairs), collections.Counter((p["group"], p["cls84"]) for p in pairs), "test compounds", len(test_comp), flush=True)
    # conditions keyed by salt-stripped name and dose
    wells = collections.defaultdict(list)
    for j in treated:
        wells[(V.norm(drug[j]), dose[j])].append(j)
    ir = lines.index(REF); tl = [i for i in keep_l if i != ir]; cellrep = np.array([dmso_all[i] for i in tl], np.float32)
    obs = []; built = {}
    for ref in ("dmso", "dprime"):
        rng = np.random.default_rng(42); Y = []; refD = []; ref_key = {}; ob = []
        for k, js in sorted(wells.items()):
            rd = [(j, delta(ir, j, ref)) for j in js]; rd = [(j, x) for j, x in rd if x is not None]
            if not rd:
                continue
            pick = rd[int(rng.integers(len(rd)))]; ref_key[k] = len(refD); refD.append(pick[1])
            names = sorted({drug[j] for j in js}); sm_ = next((smi.get(n, "") for n in names if smi.get(n, "")), "")
            for i in tl:
                ds = [x for x in (delta(i, j, ref) for j in js) if x is not None]
                if not ds:
                    continue
                ob.append(dict(line=lines[i], comp=k[0], dose=k[1], cond="%s_%s" % k, names=names, smiles=sm_, ref=ref_key[k], ref_well=int(pick[0])))
                Y.append(np.mean(ds, 0))
        pk = [(i, j) for i in keep_l for j in pw_wells if delta(i, j, ref) is not None]
        np.save(OUT + "/%s/Y.npy" % ref, np.array(Y, np.float32)); np.save(OUT + "/%s/refD.npy" % ref, np.array(refD, np.float32))
        np.save(OUT + "/%s/perwell.npy" % ref, np.array([delta(i, j, ref) for i, j in pk], np.float32)); np.save(OUT + "/%s/cellrep.npy" % ref, cellrep)
        built[ref] = dict(n_obs=len(ob), perwell_keys=[[int(i), int(j)] for i, j in pk]); obs.append(ob)
        print(ref, "observations", len(ob), "conditions", len(refD), "per-well entries", len(pk), flush=True)
    assert [(o["line"], o["cond"]) for o in obs[0]] == [(o["line"], o["cond"]) for o in obs[1]]
    ob = obs[0]
    usmi = sorted({o["smiles"] for o in ob if o["smiles"]}); tr, va, te, ncl = butina_split(usmi)
    # a stereoisomer or another salt of a test compound (same structure without stereochemistry, largest fragment)
    # would give the models the test compound under another name: those compounds go to test too
    from rdkit import Chem
    def flat(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s
        return Chem.MolToSmiles(max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumHeavyAtoms()), isomericSmiles=False)
    fk = {s: flat(s) for s in usmi}; test_keys = {fk[o["smiles"]] for o in ob if o["comp"] in test_comp and o["smiles"]}
    moved = sorted({o["comp"] for o in ob if o["comp"] not in test_comp and o["smiles"] and fk[o["smiles"]] in test_keys})
    print("moved to test (stereoisomer or salt of a test compound):", moved, flush=True)
    splitC = ["test" if (o["comp"] in test_comp or o["comp"] in moved) else ("val" if o["smiles"] in va else "train") for o in ob]
    print("split C", collections.Counter(splitC), "clusters", ncl, flush=True)
    for ref in ("dmso", "dprime"):
        json.dump(dict(rules=__doc__, obs=ob, splitC=splitC, target_lines=[lines[i] for i in tl], lines=lines, samples=samples, plate=plate,
                       drug=drug, dose=[str(x) for x in dose], pairs=pairs, moved_to_test=moved, perwell_keys=built[ref]["perwell_keys"], reference=ref),
                  open(OUT + "/%s/meta.json" % ref, "w"))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
