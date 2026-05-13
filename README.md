# CS471 MoonBoard GNN

This repository contains code and data for a CS471 research project on predicting MoonBoard climbing problem difficulty with graph neural networks.

## Main Files

- `run_moonboard_gnn.py`: command-line experiment script for graph construction, training, and evaluation.
- `moonboard_rnn_gnn_experiment.ipynb`: lightweight experiment notebook.
- `moonboard_gnn_submission/moonboard_gnn_reproducible_experiment.ipynb`: self-contained reproducibility notebook for project submission.
- `moonGen_scrape_2016_final`: MoonBoardRNN raw MoonBoard 2016 dataset.
- `HoldFeature2016LeftHand.csv`, `HoldFeature2016RightHand.csv`: optional hold difficulty feature files from MoonBoardRNN.

## Experiment Variables

The main research question is how to represent a MoonBoard climbing problem as a graph for grade prediction. Therefore, the primary experiments fix GAT as the graph encoder and compare different node features and edge construction methods.

### Fixed Graph Encoder

- `gat`: Graph Attention Network

GAT is used as the main encoder because different hold-to-hold relationships may have different importance. For example, a reach near the crux of a climb may matter more than an easier local connection. Fixing the encoder also keeps the experiment focused on graph representation instead of turning the project into a broad comparison of GNN architectures.

The code still supports `gcn` and `sage`, but they are optional extensions rather than the main experimental axis.

### Node Feature Variants

- `xy`: normalized hold position.
- `type`: `xy` plus start/middle/end hold type.
- `type_order`: `type` plus normalized position in the stored hold list.
- `direction`: `type` plus MoonBoard 2016 hold direction vector.
- `direction_order`: `direction` plus normalized order.
- `difficulty`: `type` plus left/right hand hold difficulty scores from MoonBoardRNN.
- `difficulty_order`: `difficulty` plus normalized order.
- `difficulty_direction`: `difficulty` plus hold direction vector.
- `difficulty_direction_order`: `difficulty_direction` plus normalized order.

The difficulty-score features are treated as auxiliary features because their provenance is less clear than the raw hold configuration. Results should therefore be interpreted separately for models with and without difficulty features.

### Edge Construction Variants

- `spatial`: connect holds that are within a normalized reachability threshold.
- `sequence`: connect adjacent holds in the stored hold order.
- `hybrid`: combine spatial reachability edges and sequence-adjacent edges.

The `spatial` graph is the main formulation because it directly represents possible reachability between holds. `sequence` and `hybrid` are treated as ablations because the stored middle-hold order may not always be a reliable climbing sequence.

### Recommended Ablation Set

The full configuration space is large, so the intended comparison uses a focused set of GAT-based ablations:

| Purpose | Feature Set | Edge Mode |
|---|---|---|
| Minimal baseline | `xy` | `spatial` or `hybrid` |
| Hold-role effect | `type` | `spatial` |
| Direction effect | `direction` | `spatial` |
| Direction with sequence information | `direction` | `hybrid` |
| Auxiliary difficulty effect | `difficulty` | `spatial` |
| Full metadata | `difficulty_direction` | `spatial` |
| Full metadata with sequence information | `difficulty_direction` | `hybrid` |

The main interpretation should answer:

> With the same GAT encoder, which node feature set and edge construction method produce the most useful graph embedding for MoonBoard grade prediction?

## Running Notebook Experiments

The current notebook uses one `EXPERIMENTS` list as the only experiment coverage definition. In Colab or Jupyter, run all cells to execute every listed case and generate the summary table and plots.

The default list compares imbalance-handling strategies for the same `difficulty_direction + spatial + GAT` model:

- `unweighted`: original cross entropy loss.
- `class_weighted`: balanced cross entropy using weights computed from the training split.
- `moonboardrnn_v1` / `moonboardrnn_v2`: hand-tuned class weights used as MoonBoardRNN-inspired settings.
- `balanced_replacement`: MoonGen-style train-set sampling with replacement, using `train_samples_per_class`.

To run feature/edge ablations or change the sampling coverage, edit `EXPERIMENTS` instead of using a separate ablation cell.

The notebook saves results under `outputs_experiments/` and displays:

- per-epoch loss and validation metrics,
- test metric comparisons across all listed experiments,
- majority baseline and reported MoonBoardRNN GradeNet reference lines.

Each `result.json` stores the default test metrics selected by validation `+/-1` accuracy and `test_metrics_by_selection`, which also includes the checkpoint selected by validation macro-F1.

## Notes

Initial commit intentionally excludes generated output folders and preliminary result logs. Those can be added in later commits after the experiment setup is finalized.

For notes on possible training-method improvements such as class weighting, ordinal loss, and rare-class handling, see `docs/model_improvement_notes.md`.

## Related Work

This project was influenced by the following MoonBoardRNN work:

- Yi-Shiou Duh and Ray Chang. "Recurrent Neural Network for MoonBoard Climbing Route Classification and Generation." arXiv:2102.01788, 2021. https://arxiv.org/abs/2102.01788
