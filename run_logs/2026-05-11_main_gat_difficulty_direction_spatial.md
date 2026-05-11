# Run Log: Main GAT Difficulty + Direction Spatial

Date: 2026-05-11

Source folder: `run_result_260511`

## Configuration

- Dataset: MoonBoardRNN raw MoonBoard 2016 data
- Number of problems: `25096`
- Split mode: fixed size
- Train / validation / test: `20157 / 2442 / 2497`
- Model: `gat`
- Node feature set: `difficulty_direction`
- Edge mode: `spatial`
- Order mode: `source`
- Reach threshold: `0.4`
- Epochs: `20`
- Batch size: `128`
- Hidden dimension: `64`
- Learning rate: `0.005`
- Seed: `471`
- Runtime: `174.98` seconds

## Label Distribution

| Class | Count |
|---:|---:|
| 0 | 8335 |
| 1 | 6905 |
| 2 | 3311 |
| 3 | 2532 |
| 4 | 2625 |
| 5 | 877 |
| 6 | 305 |
| 7 | 141 |
| 8 | 41 |
| 9 | 24 |

The dataset is strongly imbalanced toward easier grades. This should be considered when interpreting macro-F1.

## Test Results

| Model | Exact Acc | +/-1 Acc | Macro-F1 |
|---|---:|---:|---:|
| Majority baseline | 0.332 | 0.607 | 0.050 |
| GAT graph model | 0.461 | 0.800 | 0.172 |
| MoonBoardRNN GradeNet, reported | 0.467 | 0.847 | 0.255 |

## Comparison

Compared with the majority baseline, the GAT graph model improved:

- Exact accuracy by `+0.129`
- +/-1 accuracy by `+0.193`
- Macro-F1 by `+0.122`

Compared with the reported MoonBoardRNN GradeNet baseline, the GAT graph model was lower by:

- Exact accuracy: `-0.006`
- +/-1 accuracy: `-0.048`
- Macro-F1: `-0.082`

## Validation Trend

Best validation epochs:

| Criterion | Epoch | Val Exact | Val +/-1 | Val Macro-F1 | Loss |
|---|---:|---:|---:|---:|---:|
| Best exact | 16 | 0.452 | 0.794 | 0.152 | 1.367 |
| Best +/-1 | 19 | 0.448 | 0.803 | 0.168 | 1.363 |
| Best macro-F1 | 20 | 0.437 | 0.794 | 0.177 | 1.357 |

Training loss decreased from `1.591` to `1.357`, and validation macro-F1 was highest at the final epoch. This suggests the model may not be fully converged at 20 epochs.

## Interpretation

Under a matched data-scale setting, the GAT-based graph model achieved exact accuracy close to the reported MoonBoardRNN GradeNet result. The exact accuracy gap was only about `0.006`.

However, the graph model still underperformed the sequence-based baseline on +/-1 accuracy and macro-F1. This suggests that the graph representation captures useful grade-related structure, but it currently struggles more with ordinal consistency and minority grade classes.

This is a meaningful preliminary result for the project: it does not show that the GNN outperforms MoonBoardRNN, but it does show that a graph formulation can approach the sequence baseline on exact grade classification while remaining conceptually different and more directly tied to hold reachability.

## Next Runs

Recommended next ablations:

- `direction + spatial`: tests graph performance without provenance-uncertain hold difficulty features.
- `difficulty + spatial`: isolates the contribution of hold difficulty without direction.
- `difficulty_direction + hybrid`: checks whether adding sequence-adjacent edges improves or hurts the current best configuration.
