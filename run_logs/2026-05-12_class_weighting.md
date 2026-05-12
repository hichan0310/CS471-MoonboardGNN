# Run Log: Class-Weighting Comparison

Date: 2026-05-12

Source folder: `run_result_260512_class_weight`

## Purpose

This run tests whether class-weighted cross entropy improves grade prediction for the imbalanced MoonBoard grade distribution.

The representation is fixed to the strongest stable configuration from the previous experiments:

- Node feature set: `difficulty_direction`
- Edge construction: `spatial`
- Model: `gat`

Only the loss weighting changes between runs.

## Configuration

All runs use:

- Dataset: MoonBoardRNN raw MoonBoard 2016 data
- Number of problems: `25096`
- Train / validation / test: `20157 / 2442 / 2497`
- Feature set: `difficulty_direction`
- Edge mode: `spatial`
- Model: `gat`
- Edge threshold for spatial reachability: `0.4`
- Order mode: `source`
- Epochs: `20`
- Batch size: `128`
- Seed: `471`

## Implementation Note

The code now supports class weighting through `class_weight_mode`.

- `none`: uses standard unweighted cross entropy.
- `balanced`: computes class weights from the training split and passes them to `nn.CrossEntropyLoss(weight=class_weights)`.

When `class_weight_mode = "none"`, `class_weights` is `None`, so the loss function follows the same logic as the previous unweighted code.

For the balanced run, the computed class weights were:

| Class | Weight |
|---:|---:|
| 0 | 0.301 |
| 1 | 0.363 |
| 2 | 0.758 |
| 3 | 0.991 |
| 4 | 0.956 |
| 5 | 2.859 |
| 6 | 8.227 |
| 7 | 17.838 |
| 8 | 61.082 |
| 9 | 106.089 |

The weights are very large for rare high-grade classes, which is important for interpreting the result.

## Test Results

| Run | Class Weight Mode | Exact Acc | +/-1 Acc | Macro-F1 | Runtime |
|---|---|---:|---:|---:|---:|
| `dd_spatial_unweighted` | `none` | 0.460 | 0.799 | 0.166 | 209.4s |
| `dd_spatial_class_weighted` | `balanced` | 0.377 | 0.801 | 0.179 | 184.9s |
| Majority baseline | - | 0.332 | 0.607 | 0.050 | - |
| MoonBoardRNN GradeNet, reported | class-weighted RNN baseline | 0.467 | 0.847 | 0.255 | - |

## Interpretation

Class weighting changes the model behavior in the expected direction, but the current balanced weighting is too aggressive to call it a clear improvement.

Observed tradeoff:

- Exact accuracy drops substantially: `0.460 -> 0.377`
- `+/-1` accuracy is nearly unchanged: `0.799 -> 0.801`
- Macro-F1 improves modestly: `0.166 -> 0.179`

This means the weighted loss likely makes the model pay more attention to rare classes, but it also hurts the dominant low/mid-grade classes enough to reduce exact accuracy.

The class weights explain this tradeoff. Since class 8 and class 9 are extremely rare, balanced weighting assigns them very large loss weights. This can destabilize the decision boundary unless the weight strength is controlled.

## Validation Behavior

The current training loop selects the best model by validation `+/-1` accuracy.

For the weighted run:

- Best validation `+/-1`: epoch `16`, macro-F1 `0.169`
- Best validation macro-F1: epoch `13`, macro-F1 `0.190`

Therefore, the current saved model may not be the best model for the main goal of class weighting, which is usually macro-F1 or rare-class recall improvement.

## Points To Elaborate

- The class-weighting implementation is a clean extension of the previous code path.
- Turning weighting off with `class_weight_mode = "none"` recovers the previous unweighted loss behavior.
- Full balanced weighting improves macro-F1 slightly, but the exact-accuracy cost is too large.
- The result supports class imbalance as a real issue, but not full balanced weighting as the final solution.
- The experiment should not be presented as outperforming MoonBoardRNN. It is better presented as an analysis of imbalance handling in the graph formulation.

## Suggested Next Experiments

1. Add `selection_metric` so the best checkpoint can be selected by either `relaxed_acc` or `macro_f1`.
2. Add less aggressive weighting modes:
   - `sqrt_balanced`
   - `clipped_balanced`
   - possibly `log_balanced`
3. Add per-class precision, recall, F1, and confusion matrix output.
4. Add ordinal metrics such as mean absolute grade-index error.
5. After choosing a better weighting mode, rerun the best feature/edge configuration with more epochs.

