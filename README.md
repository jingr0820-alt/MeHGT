# MeHGT

MeHGT is a mechanism-aware heterogeneous graph learning framework for
herb-disease candidate prioritization. This repository is the minimal public
implementation of MeHGT, including data preprocessing, graph construction,
model training, candidate ranking, and mechanism analysis. Input schemas and
execution instructions are provided below.

## Repository layout

```text
code/                  Data preparation, graph construction, training,
                       prediction, and mechanism-analysis scripts
data/example/          Small schema examples with synthetic or redacted values
requirements.txt       Python dependencies
```

## Input data

The source tables are available from the
[TCM-MKG dataset on Zenodo](https://doi.org/10.5281/zenodo.13763953).
The manuscript used the dataset obtained on April 27, 2026. Follow the
provider's terms when obtaining or redistributing these data.

Place the following tab-separated files directly under `data/`:

| File | Relation or attributes |
| --- | --- |
| `D4_CPM_CHP.tsv` | Patent medicine to herb |
| `D5_CPM_ICD11.tsv` | Patent medicine to ICD-11 indication |
| `D7_CHP_Medicinal_properties.tsv` | Herb attributes |
| `D9_CHP_InChIKey.tsv` | Herb to compound |
| `D13_InChIKey_EntrezID.tsv` | Compound to target |
| `D20_ICD11_MeSH.tsv` | ICD-11 to MeSH mapping |
| `D23_MeSH_targets.tsv` | Disease to target |

The files in `data/example/` contain fictional identifiers and document input
columns only. They are not a runnable study dataset. `D18_ICD11.tsv` is included
as an additional terminology schema used in the wider study.

## Installation

Use a clean Python environment. PyTorch and PyTorch Geometric must be
installed with versions compatible with the local CPU/CUDA platform.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an archival release, pin the exact tested versions and record the output
of `python -m pip freeze`.

## Core pipeline

Run commands from the repository root. Intermediate files and checkpoints are
written below `out/`, which is local-only and ignored by Git.

```bash
python code/step1_calc_idf.py
python code/step2_build_dataset.py --help
python code/step2_build_dataset.py \
  --dataset-mode external_cpm_strict \
  --seed 42 \
  --neg-rank-start 100 \
  --neg-rank-end 800
python code/step3_build_graph.py --help
python code/step3_build_graph.py
python code/step4_train_main_model.py --help
python code/step4_train_main_model.py \
  --use-mechanism \
  --seed 42 \
  --tag hgt_mech_main_s42
python code/step5_predict_main_model.py --use-mechanism --tag hgt_mech_main_s42 --mech-alpha 0.35
python code/step6_analyze_mechanism_main.py --tag hgt_mech_main_s42
```

The commands above run the core pipeline on the reference disease split
(data seed 42), using external indication-derived labels and the 100-800
negative-candidate rank window. The `external_cpm_strict` dataset mode refers
to label construction; it does not implement the strict post-reveal knowledge
visibility protocol.

The manuscript's primary evaluation uses 20 fixed disease splits with three
model initialization seeds per split. Its strict post-reveal and
external-feature-only experiments require additional graph and training
protocols. Those experiment drivers and the full split manifest are not
included in this core package. Manuscript experiments should be interpreted
using their corresponding protocols and supplementary materials.

## Data and results policy

Training metrics, checkpoints, candidate rankings, and mechanism outputs are
generated locally under `out/`. Manuscript figures, study result archives, and
the GSE26712 case-study workflow are maintained separately from this core
implementation.

## Citation and license

Repository: https://github.com/jingr0820-alt/MeHGT

Citation metadata will be updated when the manuscript is finalized.
A code license is pending author approval. Third-party datasets remain
governed by their original terms.
