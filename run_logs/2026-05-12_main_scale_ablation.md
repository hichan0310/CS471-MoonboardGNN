# Run Log: Main-Scale GAT Ablation

Date: 2026-05-12

Source folder: `run_result_260512_1`

## Configuration

All runs use:

- Dataset: MoonBoardRNN raw MoonBoard 2016 data
- Number of problems: `25096`
- Train / validation / test: `20157 / 2442 / 2497`
- Model: `gat`
- Edge threshold for spatial reachability: `0.4`
- Order mode: `source`
- Epochs: `20`
- Batch size: `128`
- Seed: `471`

## Test Results

| Run | Feature Set | Edge Mode | Exact Acc | +/-1 Acc | Macro-F1 |
|---|---|---|---:|---:|---:|
| `gat_difficulty_direction_hybrid` | `difficulty_direction` | `hybrid` | 0.462 | 0.792 | 0.174 |
| `gat_difficulty_spatial` | `difficulty` | `spatial` | 0.455 | 0.793 | 0.167 |
| `gat_direction_spatial` | `direction` | `spatial` | 0.414 | 0.750 | 0.139 |
| `gat_type_spatial` | `type` | `spatial` | 0.376 | 0.728 | 0.116 |
| `gat_xy_spatial` | `xy` | `spatial` | 0.360 | 0.681 | 0.100 |
| Majority baseline | - | - | 0.332 | 0.607 | 0.050 |
| MoonBoardRNN GradeNet, reported | sequence | - | 0.467 | 0.847 | 0.255 |

## Interpretation

The ablation shows a clear feature-quality ladder:

1. `xy` is only slightly above the majority baseline.
2. Adding start/middle/end type improves all metrics.
3. Adding hold direction gives a larger improvement.
4. Adding hold difficulty gives the largest jump among tested feature changes.
5. Adding sequence-adjacent edges through `hybrid` slightly improves exact accuracy and macro-F1 over `difficulty_spatial`, but does not improve +/-1 accuracy.

The strongest tested ablation is `difficulty_direction + hybrid` by exact accuracy and macro-F1. However, compared with the previous `difficulty_direction + spatial` run from 2026-05-11, the difference is very small:

| Run | Exact Acc | +/-1 Acc | Macro-F1 |
|---|---:|---:|---:|
| `difficulty_direction + spatial` | 0.461 | 0.800 | 0.172 |
| `difficulty_direction + hybrid` | 0.462 | 0.792 | 0.174 |

This suggests that sequence-adjacent edges are not yet clearly beneficial. The main value still appears to come from node features, especially hold difficulty and hold direction.

## Points To Elaborate

- The GNN formulation is meaningful because even `xy/type/direction` graph features improve over majority baseline without using the RNN sequence generator.
- The current best graph models approach MoonBoardRNN on exact accuracy, but remain below it on +/-1 accuracy and macro-F1.
- Macro-F1 remains low because the grade distribution is highly imbalanced and high-grade classes are rare.
- Difficulty features improve performance substantially, but their provenance is less clear; results without difficulty features should be reported separately.
- Hybrid edges need careful wording: they do not strongly validate the stored hold order as a true climbing sequence.

## Suggested Next Experiments

1. Run `difficulty_direction + spatial` in the same ablation batch for direct side-by-side comparison.
2. Add `direction + hybrid` to test whether sequence edges help when no difficulty feature is used.
3. Add `type + hybrid` or `xy + hybrid` only if we want a fuller edge-construction ablation.
4. Consider class weighting or ordinal loss to improve macro-F1 and +/-1 accuracy.
5. Run at least one repeated seed for the top two configurations to check whether the small `spatial` vs `hybrid` gap is noise.
