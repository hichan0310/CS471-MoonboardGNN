# Run Log: Balanced Replacement Sampling and Classwise Diagnostics

Date: 2026-05-13

Source folders:

- `run_result_260513_sample_replacement`
- `run_result_260513_sample_replacement_with_model_selection`
- `run_result_260513_sample_replacement_with_classwise_metric`

## Purpose

This branch tests whether MoonGen-style balanced sampling with replacement can improve rare-grade prediction in the MoonBoard GNN model.

The previous class-weighting run showed that the grade distribution is highly imbalanced and that macro-F1 remains low. This branch therefore compares three imbalance-handling choices while keeping the graph representation fixed:

- no imbalance correction,
- class-weighted cross entropy,
- balanced training sampling with replacement.

The representation is fixed to:

- Node feature set: `difficulty_direction`
- Edge construction: `spatial`
- Model: `gat`

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

The original train-label distribution is:

| Class | Train Count |
|---:|---:|
| 0 | 6695 |
| 1 | 5546 |
| 2 | 2659 |
| 3 | 2034 |
| 4 | 2108 |
| 5 | 705 |
| 6 | 245 |
| 7 | 113 |
| 8 | 33 |
| 9 | 19 |

For `balanced_replacement`, the effective training set is resampled to `2000` examples per class, for `20000` effective training examples total. Validation and test splits are not resampled.

## Implementation Changes

This branch adds three experiment features.

1. Balanced replacement sampling

   - Added `train_sampling_mode` with values `none` and `balanced_replacement`.
   - Added `train_samples_per_class`.
   - Sampling is applied only to the training split.
   - Validation and test distributions remain unchanged.

2. Multiple checkpoint selection criteria

   The training loop now stores two best checkpoints from the same training run:

   - checkpoint selected by validation `+/-1` accuracy,
   - checkpoint selected by validation macro-F1.

   The default `test_metrics` remains selected by validation `+/-1` accuracy for backward compatibility. The full result is stored in `test_metrics_by_selection`.

3. Classwise diagnostics

   Each test result now stores:

   - per-class precision,
   - per-class recall,
   - per-class F1,
   - per-class support,
   - confusion matrix.

   The notebook also displays classwise tables and confusion-matrix heatmaps for both checkpoint selection criteria.

## Scalar Test Results

The table below uses the final run with classwise diagnostics: `run_result_260513_sample_replacement_with_classwise_metric`.

| Run | Imbalance Method | Checkpoint Selected By | Exact Acc | +/-1 Acc | Macro-F1 |
|---|---|---|---:|---:|---:|
| `dd_spatial_unweighted` | none | val +/-1 | 0.4573 | 0.7990 | 0.1655 |
| `dd_spatial_unweighted` | none | val macro-F1 | **0.4618** | **0.8030** | 0.1764 |
| `dd_spatial_balanced` | computed balanced class weights | val +/-1 | 0.4097 | 0.7889 | 0.1915 |
| `dd_spatial_balanced` | computed balanced class weights | val macro-F1 | 0.3825 | 0.7201 | 0.1837 |
| `dd_spatial_moonboardrnn_v1` | MoonBoardRNN weight v1 | val +/-1 | 0.4301 | 0.7974 | 0.1780 |
| `dd_spatial_moonboardrnn_v1` | MoonBoardRNN weight v1 | val macro-F1 | 0.4037 | 0.7958 | 0.1935 |
| `dd_spatial_moonboardrnn_v2` | MoonBoardRNN weight v2 | val +/-1 | 0.4165 | 0.7773 | 0.1497 |
| `dd_spatial_moonboardrnn_v2` | MoonBoardRNN weight v2 | val macro-F1 | 0.3849 | 0.7369 | 0.1683 |
| `dd_spatial_balanced_replacement` | balanced replacement sampling | val +/-1 | 0.4077 | 0.7569 | 0.1953 |
| `dd_spatial_balanced_replacement` | balanced replacement sampling | val macro-F1 | 0.3981 | 0.7245 | **0.2022** |

## Rare-Class Results

Rare classes are defined here as classes `5-9`. These correspond to high grades and are sparse in the test split.

| Run | Checkpoint Selected By | Rare F1 Avg, Classes 5-9 | Rare Recall Avg, Classes 5-9 | Classes With Nonzero Recall |
|---|---|---:|---:|---|
| `unweighted` | val +/-1 | 0.0044 | 0.0023 | 0, 1, 3, 4, 5 |
| `unweighted` | val macro-F1 | 0.0155 | 0.0092 | 0, 1, 3, 4, 5 |
| `balanced` | val +/-1 | 0.0897 | 0.0950 | 0, 1, 3, 4, 5, 6, 7 |
| `balanced` | val macro-F1 | 0.0568 | 0.1347 | 0, 1, 2, 3, 4, 5, 7, 8 |
| `moonboardrnn_v1` | val macro-F1 | 0.0809 | 0.1695 | 0, 1, 2, 3, 5, 8 |
| `moonboardrnn_v2` | val macro-F1 | 0.0751 | 0.0713 | 0, 1, 3, 5, 6, 7 |
| `balanced_replacement` | val +/-1 | **0.0898** | 0.2200 | 0, 1, 3, 4, 5, 6, 7, 8, 9 |
| `balanced_replacement` | val macro-F1 | 0.0736 | **0.2598** | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |

## Interpretation

The classwise diagnostics clarify the tradeoff.

The unweighted model has the best overall exact and `+/-1` accuracy, but it mostly ignores rare high-grade classes. In the unweighted run, classes `6-9` have F1 score `0`. Therefore, the high scalar accuracy is partly caused by the imbalanced test distribution.

Balanced replacement sampling changes the model behavior in the intended direction. With the macro-F1 selected checkpoint, every class from `0` through `9` has nonzero recall. This is the strongest evidence so far that resampling can improve rare-class coverage in the graph model.

However, full class balancing is too aggressive. Resampling class `8` from 33 original training examples and class `9` from 19 original training examples up to 2000 examples each increases rare-class recall, but it also causes many false positives. This is why macro-F1 improves while exact and `+/-1` accuracy drop.

The MoonBoardRNN v1 class-weighting scheme is a less aggressive compromise. It improves macro-F1 to `0.1935` while keeping `+/-1` accuracy at `0.7958`, close to the unweighted run. It does not cover every rare class, but the overall tradeoff is more stable than full balanced replacement.

## Checkpoint Selection Note

The checkpoint labels refer to validation-set selection, not test-set selection.

For example, in the unweighted run:

- epoch `16` had the best validation `+/-1` accuracy: `0.80999`, with test `+/-1` accuracy `0.79896`.
- epoch `20` had the best validation macro-F1: `0.17414`, with test `+/-1` accuracy `0.80296`.

Therefore, it is possible for the checkpoint selected by validation macro-F1 to have better test `+/-1` accuracy than the checkpoint selected by validation `+/-1` accuracy. This is not data leakage or a selection bug; it reflects normal validation/test variation.

Future result tables should name these columns explicitly as `selected_by_val_relaxed_acc` and `selected_by_val_macro_f1` to avoid ambiguity.

## Conclusions From This Branch

- Balanced replacement sampling is useful as evidence that rare-class recall can be improved.
- Full replacement to 2000 samples per class is not yet a good final setting because it sacrifices too much exact and `+/-1` accuracy.
- MoonBoardRNN v1 class weights are currently the strongest stable compromise among the tested imbalance methods.
- The project should report both overall metrics and classwise metrics; scalar accuracy alone hides severe rare-class failure.
- Confusion matrices are necessary for interpreting macro-F1 changes, because some settings improve recall by overpredicting rare classes.

## Suggested Next Experiments

1. Add balanced replacement intensity ablations:

   - `train_samples_per_class = 500`
   - `train_samples_per_class = 1000`
   - `train_samples_per_class = 1500`
   - `train_samples_per_class = 2000`

2. Keep per-class precision, recall, F1, and confusion matrix in all future runs.

3. Add repeated seeds for the strongest candidates:

   - unweighted, selected by validation macro-F1,
   - MoonBoardRNN v1 class weights, selected by validation macro-F1,
   - balanced replacement with reduced sampling intensity.

4. Consider a composite checkpoint rule after the intensity ablation, for example macro-F1 with a minimum validation `+/-1` threshold. This may avoid selecting checkpoints that improve rare-class recall while damaging ordinal accuracy too much.

5. Add ordinal metrics such as mean absolute grade-index error and within-2 accuracy before final reporting.
