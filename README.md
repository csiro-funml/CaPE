# Causal Preference Elicitation (CaPE)

Code release for **Causal Preference Elicitation (CaPE)** (Bonilla, Zhao, Steinberg, 2026).

- Pre-print: https://arxiv.org/abs/2602.01483
- Repo: https://github.com/csiro-funml/CaPE 

![Reproducible](https://img.shields.io/badge/reproducibility-one_command-brightgreen)

## One-command reproduction

From a fresh clone (assuming python points to python3.14):

```bash
python -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
./reproduce_paper.sh
```

This runs the paper experiments and regenerates figures using the plotting pipeline.

## Repository layout

- `dag/`, `feedback/`, `generation/`, `inference/`, `likelihood/`, `metrics/`, `expts/` – core library modules
- `experiments/` – paper experiment entrypoints
- `scripts/` – helper scripts (cluster submission, plotting, misc)
- `configs/` – reference hyperparameters (YAML)
- `docs/` – dataset setup notes
- `results/`, `figures/`, `data/` – generated artifacts (gitignored)

## Key entrypoints (paper)

- Sachs (HITL): `python experiments/run_sachs_hitl.py`
- Synthetic (cluster): `bash scripts/cluster/submit_synthetic_cluster.sh`
- CausalBench (cluster): `bash scripts/cluster/submit_causalbench_cluster.sh`
- Plots from saved results: `bash scripts/plots/generate_all_plots.sh`

## Existing results 
Results ready to be parsed by `scripts/generate_paper_figures.sh` and regenerate the figures in the submission can be found in `results_final.zip`

## Cluster Experiments 
Scripts for launching the synthetic and causalbench experiments in the cluster are at  `scripts/cluster/submit_causalbench_cluster.sh` and `scripts/cluster/submit_synthetic_cluster.sh`. 

## Citation

See `CITATION.cff`.
