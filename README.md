# well-is-the-unit

Analysis code and result files for the preprint *Cell lines that share a well are not independent observations in the
Tahoe-100M drug screen* (Xuanrui Huang, Department of Biomedical Informatics, Yong Loo Lin School of Medicine, National
University of Singapore).

The scripts in `code/` are the ones that produced the numbers in the paper. For release only the server paths were
changed: every absolute path now sits under one configurable root, `WORKSPACE`, with the same relative layout as on the
machines where the analyses ran (see **Verification**). `data/` holds the result files behind every figure and table,
so the figures can be redrawn without the raw data.

## Quick start: figures from the included result files

```bash
pip install -r requirements.txt
python code/make_figs110.py data figures      # Fig. 2 and Supplementary Fig. 1 (also early versions of Figs 1, 3, 4)
python code/make_figs111.py data figures      # Figs 1, 3 and 4 (overwrites the early versions)
```

This writes `fig1`–`fig4` and `figS1` (PDF and PNG) and the statistics quoted in the captions (`figstats110.json`,
`figstats111.json`). Drawn this way they are identical, pixel for pixel, to the figures of the preprint (checked with
Python 3.10, NumPy 2.2 and Matplotlib 3.10). Arial is used if installed. `paper/` holds the manuscript sources
(`main.tex`, `supp_tables.tex`, `refs.bib`); they are typeset with the Springer Nature LaTeX template (`sn-jnl.cls`,
`sn-nature.bst`), which is not redistributed here.

## What is where

| Paper item | Result file in `data/` | Script |
|---|---|---|
| Fig. 1b, c; Supplementary Fig. 1 | `fig110.json`, `fig110.npz` | `fig110.py` |
| Fig. 1e | `figdata89.json` (`control_per_plate`); Emerald Bay row: `other_screens/emerald103_results.json` | `labels84.py`, `emerald103.py` |
| Fig. 2 | `fig110_labels.json` | `fig110.py labels` |
| Fig. 3b, Fig. 4c | `figdata89.json` (`pairs`) | `figexport89.py` |
| Fig. 3c, d | `fig110.json`, `fig110.npz`, `figdata89.json` | `fig110.py`, `figexport89.py` |
| Fig. 4b | `eval82_fewshot_dprime_confirmed.json`, `bp91/summary.json`, `mol91_dprime.json`, `rhaister/out/rh99.json` | `eval82.py`, `bp91_*.py`, `mol91.py`, `rh_*.py`, `rh99.py` |
| Supplementary Table 1 (control-well pairs) | `ctrl99.json`, `cont99.json`, `other_screens/emerald103_results.json` | `ctrl99.py`, `cont99.py`, `emerald103.py` |
| Supplementary Table 2 (replicates, label check) | `labels84.json`, `labels85.json`, `ver99.json`, `p13strong92.json`, `p13len92.json`, `p13r99.json` | `labels84.py`, `check85.py`, `ver99.py`, `p13*.py` |
| Supplementary Table 3 (same-well against replicate-well prediction) | `core82*.json`, `stats99.json`, `perm89.json`, `ref88.json`, `ref99*.json`, `ver99.json`, `str92.json`, `well99.json` | the scripts of the same names |
| Supplementary Table 4 (threshold, per compound) | `thr96.json` (from `figdata89.json`) | `thr96.py` |
| Supplementary Table 5 (alternative explanations) | `corr92.json`, `well99.json`, `cont99.json`, `cont99_f2.json`, `check83.json`, `ref88.json`, `ref88b.json` | the scripts of the same names |
| Supplementary Table 6 (well's share) | `stats99.json`, `str92.json`, `str92_dmso.json` | `stats99.py`, `str92.py` |
| Supplementary Table 7 (model evaluation) | `eval82_*.json`, `rhaister/out/rh99.json`, `bp91/summary.json`, `mol91_*.json` | `eval82.py`, `rh99.py`, `bp91_sum.py`, `mol91.py` |
| Supplementary Table 8 (ceilings, cell-cycle test) | `figdata89.json` (`pairs`), `cc_signflip.json`, `cc99.json` | `figexport89.py`, `cc_signflip.py`, `cc99.py` |
| Supplementary Table 9 (other screens) | `other_screens/op3_results.json`, `other_screens/sciplex_checks.json` | `sw_op3.py`, `sciplex_checks.py` |
| Text: adagrasib on plate 13; simulation of the estimators; BioPert reproduction (0.79); figure checks | `ada_k.json`, `sim79.json`, `bp/run_A_42.json`, `fig110qc.json` | `ada_k.py`, `sim79.py`, `bp_train.py`, `fig110qc.py` |

The supplementary tables were typeset by hand from these files; each number in them can be found in the file named
above. In the descriptive text fields of the result files the server roots were replaced by `$WORKSPACE`; no value was
changed. Script docstrings cite section numbers (§) of the author's lab notes, which are not part of the release.

## Full analysis

### Inputs

| Input | Source | Place under `WORKSPACE` |
|---|---|---|
| Tahoe-100M (`data/*.parquet`, `metadata/*.parquet`) | Hugging Face `tahoebio/Tahoe-100M` | `data_ext/tahoe/Tahoe-100M/` |
| PIE preprocessed Tahoe gene axis and control means (`preprocessed/meta.json`, `ctrl_means.npy`) | Hugging Face `arcinstitute/PIE_tahoe100m` | `data_ext/tahoe/PIE_tahoe100m/preprocessed/` |
| Protein-coding genes (HGNC list `data/protein-coding_gene.txt` of the BioPert repository) | GitHub `lemieux-lab/biopert` | `rt/samewell/protein-coding_gene.txt` |
| MSigDB Hallmark 2020 (Enrichr text format; required by the command line of `tahoe_program.py`, used by no number in the paper) | Enrichr library `MSigDB_Hallmark_2020` | `genesets/MSigDB_Hallmark_2020.txt` |
| Rhaister code, model card and inputs (`pdex/`, `cell_eval/`, `definition/`, holdout splits) | Hugging Face `tahoebio/Rhaister`, `tahoebio/tahoe-de-rhaister` | code in `rt/samewell/rhaister/code/`, inputs in `rt/samewell/rhaister/data/`; its Python environment as `rt/samewell/pylib_rh` and `rt/samewell/pylib_bp` (pip `--target` folders) |
| Emerald Bay `expression_data` | Hugging Face `tahoebio/EmeraldBay` (read over HTTP by `emerald_scan.py` and `emerald_filter.py`) | – |
| OP3 pseudobulk (`pseudobulk.h5ad`) | Open Problems, NeurIPS 2023 single-cell perturbation data (Szałata et al. 2024) | `mainline_rt_samewell/op3/` |
| sci-Plex3 (`GSM4150378_sciPlex3_A549_MCF7_K562_screen_*` and `GSM4150378_sciPlex3_pData.txt.gz`) | GEO GSE139944 | count files in `mainline_rt_samewell/sciplex/`, pData in `vcc2026/data/raw/official_datasets/sciplex/meta/` |

BioPert itself is not needed at run time: its network, training schedule and Tahoe-100M hyperparameters are
re-implemented in PyTorch in `bp_train.py`, `bp_molec.py` and `bp91_train.py` (the port reproduces the published test
correlation, 0.79, on the published compound split).

### Layout

Every script reads and writes under `WORKSPACE`: `data_ext/` (raw data), `tahoe_agg/` (per line × sample × half sums,
written by `tahoe_extract.py`), `tahoe_program/tahoe_arrays.npz` (cached effects), `rt/tahoe/` (condition metadata,
sample QC, depth), `rt/samewell/` (all same-well outputs, pseudobulks `pb/`, BioPert runs `bp/`, `bp91/`, Rhaister
`rhaister/`) and `mainline_rt_samewell/` (the three other screens). The upstream scripts `rt_*.py`, `tahoe_program.py`,
`tahoe_arms.py`, `tahoe_readout.py` and `cd4_m3.py` come from a broader project and contain parts for other data sets
(CD4, Replogle, X-Atlas/Orion); only the parts named in `pipelines/run_all.sh` are used here, and the paths of the other
parts point to inputs that this analysis does not need.

### Running it

Copy `config.example.env` to `config.env`, set `WORKSPACE`, place the inputs, then run the steps of
`pipelines/run_all.sh` (it can be run as a whole, but the extraction and the GPU steps take days and are better run
one at a time). Its last block collects the result files into `out/data` and draws the figures from them. The analyses
ran with Python 3.11, NumPy 2.2.6, SciPy 1.15.3, PyArrow 20.0.0, scikit-learn 1.9.1, h5py 3.16.0, PyTorch 2.5.1 and
RDKit 2024.03.2, and the figures with Python 3.10 and Matplotlib 3.10.8.

## Verification

`code/RELEASE_CHECK.tsv` lists each released script with the SHA-256 prefix of the version that ran on the servers.
For each, the release substitution (server root → `WORKSPACE`, one inserted import line; in docstrings and comments
server root → `$WORKSPACE`) was reversed and the result compared with that version: all 64 files differ from it only
in these paths. The figures drawn from `data/` match the preprint's pixel for pixel (Quick start).

## Licence

MIT (see `LICENSE`). The data sets and the Rhaister and BioPert code belong to their authors and are not
redistributed here.
