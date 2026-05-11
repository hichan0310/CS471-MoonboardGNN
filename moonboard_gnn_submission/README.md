# MoonBoard GNN Grade Prediction

This folder contains a reproducible research-project package for predicting MoonBoard 2016 climbing grades with graph neural networks.

## Files

- `moonboard_gnn_reproducible_experiment.ipynb`
  - Main reproducible notebook.
  - Contains dataset checks, graph construction, GNN model code, training, evaluation, and ablation templates.
- `moonGen_scrape_2016_final`
  - MoonBoardRNN raw MoonBoard 2016 dataset copied from `jrchang612/MoonBoardRNN`.
- `HoldFeature2016LeftHand.csv`
  - Optional left-hand hold difficulty feature file from MoonBoardRNN.
- `HoldFeature2016RightHand.csv`
  - Optional right-hand hold difficulty feature file from MoonBoardRNN.
- `experiment_summary_preliminary.csv`
  - Preliminary short-run GNN results recorded during project exploration.

## Environment

The notebook was prepared for Python 3.10 with:

- `torch`
- `torch-geometric`
- `scikit-learn`
- `numpy`
- `pandas`

The previous local experiment environment used:

- `torch 2.11.0`
- `torch-geometric 2.7.0`
- `scikit-learn 1.7.2`
- `numpy 2.2.6`
- `pandas 2.3.3`

## How to Reproduce

1. Open `moonboard_gnn_reproducible_experiment.ipynb`.
2. Run the dataset check cell.
3. Run the core implementation cell.
4. Run the notebook experiment runner cell.
5. For a quick check, set `RUN_SMOKE = True`.
6. For the main controlled-scale experiment, set `RUN_MAIN = True`.

The main run uses:

- `max_samples = 25096`
- `split_mode = "moonboardrnn_size"`
- train/validation/test sizes: `20157 / 2442 / 2497`

This matches the dataset scale reported in MoonBoardRNN's GradeNet notebook, while using graph input instead of BetaMove-generated sequence input.

## Reported Baseline

MoonBoardRNN's saved `GradeNet.ipynb` output reports the following test metrics:

- exact accuracy: `0.4666`
- +/-1 grade accuracy: `0.8474`
- macro-F1: approximately `0.2546`

Current GNN results should be compared against this baseline with care because the input representation and training schedule are different.

## Related Work

This project was influenced by:

- Yi-Shiou Duh and Ray Chang. "Recurrent Neural Network for MoonBoard Climbing Route Classification and Generation." arXiv:2102.01788, 2021. https://arxiv.org/abs/2102.01788
