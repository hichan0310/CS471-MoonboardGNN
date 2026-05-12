from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


GRADE_MAP = {
    "6B": 0,
    "6B+": 0,
    "6C": 1,
    "6C+": 1,
    "7A": 2,
    "7A+": 3,
    "7B": 4,
    "7B+": 4,
    "7C": 5,
    "7C+": 6,
    "8A": 7,
    "8A+": 8,
    "8B": 9,
}

HOLD_DIRECTION_RAW = """
1 - H7 - SE
2 - J14 - NW
3 - K7 - N
4 - D8 - N
5 - A16 - NW
6 - F6 - E
7 - K6 - N
8 - C9 - W
9 - A10 - SW
10 - I8 - N
11 - K12 - N
12 - A11 - SE
13 - B7 - S
14 - D16 - N
15 - K10 - S
16 - G16 - N
17 - F8 - N
18 - F15 - N
19 - G7 - SW
20 - H15 - NE
21 - C12 - N
22 - D6 - N
23 - D3 - S
24 - D13 - N
25 - J11 - N
26 - A13 - N
27 - G11 - E
28 - H18 - N
29 - B4 - SW
30 - K8 - N
31 - C15 - NW
32 - H9 - E
33 - D10 - NW
34 - H14 - W
35 - I5 - N
36 - I12 - SW
37 - K13 - N
38 - C7 - N
39 - C18 - N
40 - F9 - N
50 - C14 - N
51 - D17 - N
52 - D9 - NE
53 - F7 - NW
54 - F12 - E
55 - G12 - NE
56 - B11 - NW
57 - J10 - NE
58 - J2 - SE
59 - E13 - N
60 - I6 - NE
61 - J9 - SE
62 - F14 - NW
63 - I13 - E
64 - E10 - NW
65 - F10 - NE
66 - E15 - NW
67 - B8 - N
68 - A12 - E
69 - I16 - NE
70 - I11 - N
71 - B16 - NW
72 - E11 - N
73 - H11 - W
74 - E7 - S
75 - D12 - N
76 - J8 - N
77 - B13 - NW
78 - B9 - NE
79 - C10 - NE
80 - B3 - SW
81 - G2 - N
82 - G18 - W
83 - I4 - NE
84 - K11 - NW
85 - A5 - N
86 - K5 - N
87 - K18 - W
88 - G8 - N
89 - F5 - N
90 - G13 - N
91 - E18 - N
92 - J6 - S
93 - D14 - N
94 - C11 - W
95 - C6 - S
96 - F16 - S
97 - D5 - NW
98 - A15 - N
99 - B18 - SE
100 - H16 - N
101 - B15 - N
102 - J12 - NE
103 - J13 - N
104 - K16 - N
105 - F13 - NW
106 - E16 - NW
107 - I7 - NE
108 - I15 - NW
109 - I9 - SE
110 - E12 - NE
111 - H5 - NW
112 - G15 - NW
113 - J7 - N
114 - H12 - NW
115 - G17 - N
116 - E9 - NE
117 - J16 - E
118 - F11 - NE
119 - D11 - SW
120 - I10 - N
121 - K9 - N
122 - E8 - N
123 - A14 - NW
124 - I14 - NW
125 - C5 - N
126 - D15 - NW
127 - E14 - E
128 - G9 - NE
129 - E6 - NW
130 - J5 - NW
131 - H8 - NE
132 - I18 - NE
133 - A9 - NW
134 - G6 - SW
135 - C8 - NW
136 - D18 - N
137 - G14 - E
138 - C13 - NW
139 - A18 - N
140 - H10 - NE
141 - G4 - N
142 - B12 - SE
143 - C16 - N
144 - K14 - NE
145 - G10 - NE
146 - D7 - S
147 - B6 - NW
148 - B10 - SE
149 - H13 - SW
"""

DIRECTION_VECTORS = {
    "N": (0.0, 1.0),
    "NE": (0.707, 0.707),
    "E": (1.0, 0.0),
    "SE": (0.707, -0.707),
    "S": (0.0, -1.0),
    "SW": (-0.707, -0.707),
    "W": (-1.0, 0.0),
    "NW": (-0.707, 0.707),
}


def build_hold_direction_map() -> dict[tuple[int, int], tuple[float, float]]:
    result = {}
    for line in HOLD_DIRECTION_RAW.strip().splitlines():
        parts = [p.strip() for p in line.split(" - ")]
        if len(parts) != 3:
            continue
        hold = parts[1]
        direction = parts[2]
        x = ord(hold[0].upper()) - ord("A")
        # MoonBoardRNN raw data stores y as zero-based board coordinates,
        # while hold strings such as A5 are one-based.
        y = int(hold[1:]) - 1
        result[(x, y)] = DIRECTION_VECTORS[direction]
    return result


HOLD_DIRECTIONS = build_hold_direction_map()


@dataclass(frozen=True)
class Problem:
    key: str
    grade: str
    label: int
    start: tuple[tuple[int, int], ...]
    mid: tuple[tuple[int, int], ...]
    end: tuple[tuple[int, int], ...]
    is_benchmark: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_hold_difficulties(path: Path) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("X_coord") or not row.get("Y_coord"):
                continue
            x = int(float(row["X_coord"]))
            y = int(float(row["Y_coord"]))
            diff = float(row["Difficulties"])
            values[(x, y)] = diff
    return values


def load_problems(
    raw_path: Path,
    dataset: str,
    max_samples: int | None,
    seed: int,
) -> list[Problem]:
    raw = pickle.load(raw_path.open("rb"))
    problems: list[Problem] = []
    for key, item in raw.items():
        grade = item.get("grade")
        if grade not in GRADE_MAP:
            continue
        if dataset == "benchmark" and not item.get("is_benchmark"):
            continue
        if dataset == "nonbenchmark" and item.get("is_benchmark"):
            continue
        start = tuple(tuple(map(int, hold)) for hold in item.get("start", []))
        mid = tuple(tuple(map(int, hold)) for hold in item.get("mid", []))
        end = tuple(tuple(map(int, hold)) for hold in item.get("end", []))
        if not start or not end:
            continue
        problems.append(
            Problem(
                key=str(key),
                grade=grade,
                label=GRADE_MAP[grade],
                start=start,
                mid=mid,
                end=end,
                is_benchmark=bool(item.get("is_benchmark")),
            )
        )

    rng = random.Random(seed)
    rng.shuffle(problems)
    if max_samples is not None:
        problems = problems[:max_samples]
    return problems


def ordered_holds(problem: Problem, order_mode: str) -> list[tuple[tuple[int, int], str]]:
    start = sorted(problem.start, key=lambda xy: (xy[1], xy[0]))
    mid = list(problem.mid)
    end = sorted(problem.end, key=lambda xy: (xy[1], xy[0]))

    if order_mode == "source":
        # Raw scrape stores start/mid/end separately. Preserve mid order.
        mid_ordered = mid
    elif order_mode == "height":
        mid_ordered = sorted(mid, key=lambda xy: (xy[1], xy[0]))
    elif order_mode == "none":
        mid_ordered = sorted(mid, key=lambda xy: (xy[0], xy[1]))
    else:
        raise ValueError(f"unknown order_mode: {order_mode}")

    return (
        [(xy, "start") for xy in start]
        + [(xy, "mid") for xy in mid_ordered]
        + [(xy, "end") for xy in end]
    )


def node_feature(
    xy: tuple[int, int],
    hold_type: str,
    position: int,
    n_holds: int,
    feature_set: str,
    left_diff: dict[tuple[int, int], float],
    right_diff: dict[tuple[int, int], float],
) -> list[float]:
    x, y = xy
    feat = [x / 10.0, y / 17.0]

    if feature_set in {
        "type",
        "type_order",
        "direction",
        "direction_order",
        "difficulty",
        "difficulty_order",
        "difficulty_direction",
        "difficulty_direction_order",
    }:
        feat.extend(
            [
                1.0 if hold_type == "start" else 0.0,
                1.0 if hold_type == "mid" else 0.0,
                1.0 if hold_type == "end" else 0.0,
            ]
        )

    if feature_set in {
        "direction",
        "direction_order",
        "difficulty_direction",
        "difficulty_direction_order",
    }:
        feat.extend(HOLD_DIRECTIONS.get(xy, (0.0, 1.0)))

    if feature_set in {
        "difficulty",
        "difficulty_order",
        "difficulty_direction",
        "difficulty_direction_order",
    }:
        feat.extend(
            [
                left_diff.get(xy, 0.0) / 10.0,
                right_diff.get(xy, 0.0) / 10.0,
            ]
        )

    if feature_set in {
        "type_order",
        "direction_order",
        "difficulty_order",
        "difficulty_direction_order",
    }:
        denom = max(1, n_holds - 1)
        feat.append(position / denom)

    return feat


def create_edges(
    holds: list[tuple[tuple[int, int], str]],
    edge_mode: str,
    reach_threshold: float,
) -> torch.Tensor:
    edge_set: set[tuple[int, int]] = set()
    coords = [xy for xy, _ in holds]

    if edge_mode in {"spatial", "hybrid"}:
        for i, (xi, yi) in enumerate(coords):
            for j, (xj, yj) in enumerate(coords):
                if i == j:
                    continue
                dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5 / ((10**2 + 17**2) ** 0.5)
                if dist <= reach_threshold:
                    edge_set.add((i, j))

    if edge_mode in {"sequence", "hybrid"}:
        for i in range(len(holds) - 1):
            edge_set.add((i, i + 1))
            # Add reverse edge so message passing is not one-way brittle.
            edge_set.add((i + 1, i))

    if not edge_set:
        for i in range(len(holds)):
            edge_set.add((i, i))

    edge_index = torch.tensor(sorted(edge_set), dtype=torch.long).t().contiguous()
    return edge_index


def problem_to_graph(
    problem: Problem,
    feature_set: str,
    edge_mode: str,
    order_mode: str,
    reach_threshold: float,
    left_diff: dict[tuple[int, int], float],
    right_diff: dict[tuple[int, int], float],
) -> Data:
    holds = ordered_holds(problem, order_mode)
    features = [
        node_feature(
            xy=xy,
            hold_type=hold_type,
            position=i,
            n_holds=len(holds),
            feature_set=feature_set,
            left_diff=left_diff,
            right_diff=right_diff,
        )
        for i, (xy, hold_type) in enumerate(holds)
    ]
    x = torch.tensor(features, dtype=torch.float32)
    edge_index = create_edges(holds, edge_mode=edge_mode, reach_threshold=reach_threshold)
    y = torch.tensor([problem.label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.problem_key = problem.key
    data.grade = problem.grade
    return data


class GATClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_max_pool(x, batch)
        return self.mlp(x)


class GCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_max_pool(x, batch)
        return self.mlp(x)


class SAGEClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_max_pool(x, batch)
        return self.mlp(x)


def make_model(model_name: str, in_channels: int, hidden_channels: int, num_classes: int) -> nn.Module:
    if model_name == "gat":
        return GATClassifier(in_channels, hidden_channels, num_classes)
    if model_name == "gcn":
        return GCNClassifier(in_channels, hidden_channels, num_classes)
    if model_name == "sage":
        return SAGEClassifier(in_channels, hidden_channels, num_classes)
    raise ValueError(f"unknown model: {model_name}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
            y_true.extend(batch.y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    exact = accuracy_score(y_true, y_pred)
    relaxed = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 1))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"exact_acc": exact, "relaxed_acc": relaxed, "macro_f1": macro_f1}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_val = -1.0
    best_state = None

    model.to(device)
    epoch_iter = range(1, epochs + 1)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, total=epochs, unit="epoch", desc="Training", dynamic_ncols=True)
    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch.num_graphs

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if val_metrics["relaxed_acc"] > best_val:
            best_val = val_metrics["relaxed_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        progress_text = (
            f"loss {train_loss:.4f}, "
            f"val_exact {val_metrics['exact_acc']:.4f}, "
            f"val_+/-1 {val_metrics['relaxed_acc']:.4f}, "
            f"val_f1 {val_metrics['macro_f1']:.4f}"
        )
        if tqdm is not None and hasattr(epoch_iter, "set_description"):
            epoch_iter.set_description(progress_text)
        else:
            print(f"epoch={epoch:03d} {progress_text}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    test_metrics = evaluate(model, test_loader, device)
    return test_metrics, history


def majority_baseline(labels: list[int], test_labels: list[int]) -> dict[str, float]:
    majority = Counter(labels).most_common(1)[0][0]
    pred = np.full(len(test_labels), majority)
    true = np.array(test_labels)
    return {
        "exact_acc": float(np.mean(pred == true)),
        "relaxed_acc": float(np.mean(np.abs(pred - true) <= 1)),
        "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="MoonBoardRNN raw-data GNN grade prediction experiment.")
    parser.add_argument("--raw", type=Path, default=Path("moonGen_scrape_2016_final"))
    parser.add_argument("--left-diff", type=Path, default=Path("HoldFeature2016LeftHand.csv"))
    parser.add_argument("--right-diff", type=Path, default=Path("HoldFeature2016RightHand.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset", choices=["all", "benchmark", "nonbenchmark"], default="all")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument(
        "--feature-set",
        choices=[
            "xy",
            "type",
            "type_order",
            "direction",
            "direction_order",
            "difficulty",
            "difficulty_order",
            "difficulty_direction",
            "difficulty_direction_order",
        ],
        default="difficulty",
    )
    parser.add_argument("--edge-mode", choices=["spatial", "sequence", "hybrid"], default="hybrid")
    parser.add_argument("--order-mode", choices=["source", "height", "none"], default="source")
    parser.add_argument("--model", choices=["gat", "gcn", "sage"], default="gat")
    parser.add_argument("--reach-threshold", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=471)
    args = parser.parse_args()

    total_start = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    left_diff = load_hold_difficulties(args.left_diff)
    right_diff = load_hold_difficulties(args.right_diff)
    problems = load_problems(args.raw, args.dataset, args.max_samples, args.seed)
    labels = [p.label for p in problems]

    train_val_idx, test_idx = train_test_split(
        np.arange(len(problems)),
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=args.seed,
        stratify=[labels[i] for i in train_val_idx],
    )

    graphs = [
        problem_to_graph(
            problem=p,
            feature_set=args.feature_set,
            edge_mode=args.edge_mode,
            order_mode=args.order_mode,
            reach_threshold=args.reach_threshold,
            left_diff=left_diff,
            right_diff=right_diff,
        )
        for p in problems
    ]

    train_set = [graphs[i] for i in train_idx]
    val_set = [graphs[i] for i in val_idx]
    test_set = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    in_channels = graphs[0].num_node_features
    num_classes = max(labels) + 1
    model = make_model(args.model, in_channels, args.hidden, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    majority = majority_baseline([labels[i] for i in train_idx], [labels[i] for i in test_idx])
    print("config:", vars(args))
    print("device:", device)
    print("num problems:", len(problems))
    print("label distribution:", dict(sorted(Counter(labels).items())))
    print("node features:", in_channels)
    print("majority baseline:", majority)

    test_metrics, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    runtime = time.perf_counter() - total_start
    result = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "runtime_seconds": runtime,
        "num_problems": len(problems),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "node_features": in_channels,
        "majority_baseline": majority,
        "test_metrics": test_metrics,
    }
    write_json(args.output_dir / "result.json", result)
    write_json(args.output_dir / "history.json", history)

    print("test metrics:", test_metrics)
    print(f"runtime seconds: {runtime:.3f}")
    print(f"saved: {args.output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
