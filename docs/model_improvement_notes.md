# Model Improvement Notes

This note summarizes candidate improvements after the first main-scale GAT ablation runs.

## Current Observation

The GAT graph model approaches the reported MoonBoardRNN GradeNet baseline on exact accuracy, but it remains lower on `+/-1` accuracy and macro-F1.

Current best graph results are around:

- Exact accuracy: `0.46`
- `+/-1` accuracy: `0.79` to `0.80`
- Macro-F1: `0.17`

Reported MoonBoardRNN GradeNet test results are:

- Exact accuracy: `0.467`
- `+/-1` accuracy: `0.847`
- Macro-F1: `0.255`

The gap is therefore small for exact accuracy, but larger for ordinal consistency and class-balanced performance.

## Why More Epochs May Not Be Enough

Increasing epochs may still improve the GNN slightly because validation macro-F1 often peaks near later epochs. However, the main remaining weakness is likely not only undertraining.

The dataset is highly imbalanced:

- Low-grade classes have thousands of examples.
- High-grade classes have very few examples.

As a result, a model can improve overall accuracy while still performing poorly on rare high-grade classes. This explains why macro-F1 is much lower than exact accuracy.

## Class Weighting

Class weighting gives larger loss penalties to rare classes and smaller penalties to frequent classes.

Instead of treating every class equally in the loss, weighted cross entropy can use:

```python
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

Expected benefit:

- Better macro-F1
- Better rare-class recall

Risk:

- Exact accuracy may drop if rare classes are overemphasized.

This is the most reasonable next training-method improvement because MoonBoardRNN GradeNet also used class weights.

Implementation note:

- `run_moonboard_gnn.py` and the notebooks support `class_weight_mode`.
- `none` keeps the original unweighted cross entropy.
- `balanced` computes weights from the training split only:

```text
weight[class] = number_of_training_samples / (number_of_present_classes * class_count)
```

The current notebook has a single `EXPERIMENTS` list. The default list runs the same `difficulty_direction + spatial + GAT` setting twice, once with `none` and once with `balanced`, then saves `outputs_experiments/experiment_summary.json` and plots the listed results against the majority baseline and reported MoonBoardRNN GradeNet metrics. Feature/edge ablations should be added to the same list rather than run through a separate optional cell.

## Ordinal Loss

MoonBoard grades are ordered labels, not purely nominal labels:

```text
6B < 6C < 7A < 7A+ < ... < 8B
```

Standard cross entropy treats all wrong predictions equally. For example, predicting one grade away and predicting five grades away are both simply wrong.

An ordinal-aware loss would penalize farther mistakes more strongly, for example by combining cross entropy with an expected grade-distance penalty.

Expected benefit:

- Better `+/-1` accuracy
- Better mean absolute error over grade indices

Risk:

- Requires an extra loss hyperparameter.
- Slightly more complex to explain than class weighting.

## Rare-Class Handling

Rare-class handling changes the training data exposure rather than the loss function.

Possible methods:

- Oversampling rare classes
- Balanced batch sampling
- Grouping extremely rare high-grade classes

Expected benefit:

- Better high-grade class recall
- Better macro-F1

Risk:

- Higher overfitting risk
- Possible mismatch between training distribution and real data distribution

## Recommended Next Order

1. Add class-weighted cross entropy.
2. Add mean absolute error as an additional metric.
3. If time allows, test ordinal-aware loss.
4. Use rare-class sampling only if class weighting is not enough.

For the report, the safest claim is:

> More epochs may improve exact accuracy slightly, but closing the macro-F1 and `+/-1` accuracy gap likely requires addressing class imbalance and the ordinal nature of climbing grades.
