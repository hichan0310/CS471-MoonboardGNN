from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = []
    for result_path in sorted(root.glob("outputs*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = result["config"]
        test = result["test_metrics"]
        majority = result["majority_baseline"]
        rows.append(
            {
                "run": result_path.parent.name,
                "model": config["model"],
                "feature_set": config["feature_set"],
                "edge_mode": config["edge_mode"],
                "order_mode": config["order_mode"],
                "max_samples": config["max_samples"],
                "epochs": config["epochs"],
                "test_exact": f"{test['exact_acc']:.6f}",
                "test_relaxed": f"{test['relaxed_acc']:.6f}",
                "test_macro_f1": f"{test['macro_f1']:.6f}",
                "majority_exact": f"{majority['exact_acc']:.6f}",
                "majority_relaxed": f"{majority['relaxed_acc']:.6f}",
                "runtime_seconds": f"{result['runtime_seconds']:.3f}",
            }
        )

    out_path = root / "experiment_summary.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {out_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
