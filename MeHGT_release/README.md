# MeHGT

MeHGT is a mechanism-aware heterogeneous graph learning framework for
herb-disease candidate prioritization. This repository is the minimal public
code package accompanying the manuscript. It contains the model pipeline and
schema-only example inputs; it does not contain manuscript results,
checkpoints, private working notes, or the complete experimental archive.

## Repository layout

```text
code/                  Data preparation, graph construction, training,
                       prediction, and mechanism-analysis scripts
data/example/          Small schema examples with synthetic or redacted values
requirements.txt       Python dependencies
```

The full source datasets are intentionally not bundled. Obtain each dataset
from its original provider, confirm its redistribution terms, and place the
files under `data/` with the filenames expected by the scripts. The files in
`data/example/` document the required columns only and are not sufficient for
reproducing the manuscript numbers.

## Installation

Use a clean Python environment. PyTorch and PyTorch Geometric must be
installed with versions compatible with the local CPU/CUDA platform.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
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
python code/step5_predict_main_model.py --help
python code/step6_analyze_mechanism_main.py --help
```

The exact split protocol, seeds, hyperparameters, and evaluation choices used
for the manuscript are part of the study record and are not represented by a
single public result file here. Running the commands above on independently
obtained inputs produces local outputs; it is not a claim that the published
metrics can be reproduced from this minimal package alone.

## Data and results policy

No manuscript tables, figures, per-seed summaries, trained checkpoints, case
study outputs, or unpublished intermediate results are distributed in this
repository. Results may be deposited separately if required by the journal and
after checking data-use, privacy, and third-party licensing terms.

## Citation and license

Citation metadata, author-approved repository URL, and a code license will be
added after the manuscript and public archive are frozen. Until then, this
repository is an author-controlled code release and does not grant a reuse
license. Third-party data remain governed by their original terms.
