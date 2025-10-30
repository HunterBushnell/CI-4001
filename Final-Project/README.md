# LUT Micturition ML Project Structure

This directory contains the scaffold for a machine learning project focused on LUT micturition analysis.

## Directory Overview

- `config/` – Configuration files such as hyperparameters and experiment settings.
- `data/`
  - `raw/` – Immutable source datasets.
  - `processed/` – Cleaned and feature-ready datasets.
- `docs/` – Project documentation and literature reviews.
- `logs/` – Training and evaluation logs.
- `models/` – Serialized model artifacts and checkpoints.
- `notebooks/` – Exploratory analyses and experiments in notebooks.
- `references/` – External references, papers, and supporting material.
- `reports/`
  - `figures/` – Generated plots and visualizations.
  - `tables/` – Generated tables and summaries.
- `scripts/` – Command-line scripts for data processing, training, and evaluation.
- `src/`
  - `data/` – Data loading and preprocessing utilities.
  - `features/` – Feature engineering logic specific to LUT micturition signals.
  - `models/` – Model definitions and training loops.
  - `visualization/` – Plotting utilities and dashboards.
- `tests/` – Automated tests for data pipelines and model code.

Each subdirectory contains a `.gitkeep` file to ensure version control captures the empty scaffold.
