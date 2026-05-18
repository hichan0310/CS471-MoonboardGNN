#!/usr/bin/env python3
"""
Standalone no-user-grade MoonBoard graph/relation stack.

Requires only:
  - this script
  - moonGen_scrape_2016_final

Hard rule:
  - user_grade is not loaded, not encoded, not used for routing/override/diagnostics.

Pipeline:
  1) fixed train/val/test split
  2) train-only relation/pair/setter encodings
  3) graph/relation sparse + dense features
  4) base learners: graph KNN, sparse SGD, HGB, ExtraTrees
  5) optional custom edge-aware GAT branch using node/edge graph tensors
  6) validation meta stack + exact-bias calibration + probability blend
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Dataset = object
    DataLoader = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

warnings.filterwarnings("ignore")

GRADE_MAP = {
    "6B": 0, "6B+": 0,
    "6C": 1, "6C+": 1,
    "7A": 2,
    "7A+": 3,
    "7B": 4, "7B+": 4,
    "7C": 5,
    "7C+": 6,
    "8A": 7,
    "8A+": 8,
    "8B": 9,
}
NUM_CLASSES = 10
BOARD_X = 11
BOARD_Y = 18
N_POS = BOARD_X * BOARD_Y


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Problem:
    key: str
    label: int
    grade: str
    start: tuple
    mid: tuple
    end: tuple
    repeats: float
    is_benchmark: bool
    is_master: bool
    setter_id: str

    @property
    def holds(self):
        return self.start + self.mid + self.end

    @property
    def typed(self):
        return tuple([(h, 's') for h in self.start] + [(h, 'm') for h in self.mid] + [(h, 'e') for h in self.end])


def load_problems(path: Path) -> List[Problem]:
    raw = pickle.load(Path(path).open('rb'))
    out: List[Problem] = []
    items = raw.items() if hasattr(raw, 'items') else enumerate(raw)
    for k, v in items:
        if not isinstance(v, dict):
            continue
        grade = v.get('grade')
        if grade not in GRADE_MAP:
            continue
        setter = v.get('setter') or {}
        if not isinstance(setter, dict):
            setter = {}
        out.append(Problem(
            key=str(k),
            label=int(GRADE_MAP[grade]),
            grade=str(grade),
            start=tuple(tuple(map(int, h)) for h in v.get('start', []) or []),
            mid=tuple(tuple(map(int, h)) for h in v.get('mid', []) or []),
            end=tuple(tuple(map(int, h)) for h in v.get('end', []) or []),
            repeats=float(v.get('repeats') or 0.0),
            is_benchmark=bool(v.get('is_benchmark')),
            is_master=bool(v.get('is_master')),
            setter_id=str(setter.get('Id') or setter.get('Nickname') or 'NONE'),
        ))
    return out


def hold_col(h) -> int:
    return int(h[0]) * BOARD_Y + int(h[1])


def role_id(t: str) -> int:
    return {'s': 0, 'm': 1, 'e': 2}[t]


def phash_int(vals, mod: int) -> int:
    x = 2166136261
    for v in vals:
        x ^= int(v) & 0xffffffff
        x = (x * 16777619) & 0xffffffff
    return x % mod


def pair_keys(ha, ta, hb, tb):
    dx = hb[0] - ha[0]
    dy = hb[1] - ha[1]
    dist = math.hypot(dx, dy)
    return [
        ('dir', ha, hb),
        ('undir', tuple(sorted([ha, hb]))),
        ('typed', ta, tb, ha, hb),
        ('geom', ta, tb, min(6, abs(dx)), max(0, min(16, dy + 8)), min(10, int(round(dist)))),
    ]


def build_tables(problems: List[Problem], train_idx, smoothing: float = 12.0):
    log('build train-only relation/setter encodings')
    y = np.array([problems[int(i)].label for i in train_idx], dtype=np.int64)
    global_dist = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    global_dist /= max(1.0, global_dist.sum())
    global_mean = float(y.mean()) if len(y) else 0.0

    setter_count = defaultdict(int)
    setter_sum = defaultdict(float)
    setter_dist = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.float64))
    pair_count = defaultdict(int)
    pair_sum = defaultdict(float)
    pair_dist = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.float64))

    it = train_idx
    if tqdm is not None:
        it = tqdm(list(train_idx), desc='tables', dynamic_ncols=True)
    for ii in it:
        p = problems[int(ii)]
        setter_count[p.setter_id] += 1
        setter_sum[p.setter_id] += p.label
        setter_dist[p.setter_id][p.label] += 1
        typed = p.typed
        for i, (ha, ta) in enumerate(typed):
            for j, (hb, tb) in enumerate(typed):
                if i == j:
                    continue
                dx = hb[0] - ha[0]
                dy = hb[1] - ha[1]
                dist = math.hypot(dx, dy)
                if dist == 0 or dist > 10:
                    continue
                for key in pair_keys(ha, ta, hb, tb):
                    pair_count[key] += 1
                    pair_sum[key] += p.label
                    pair_dist[key][p.label] += 1

    def mean(sumtab, cnttab, key):
        c = cnttab.get(key, 0)
        return (sumtab.get(key, 0.0) + smoothing * global_mean) / (c + smoothing)

    def dist(tab, cnttab, key):
        c = cnttab.get(key, 0)
        return (tab.get(key, np.zeros(NUM_CLASSES, dtype=np.float64)) + smoothing * global_dist) / (c + smoothing)

    return dict(
        gm=global_mean,
        gd=global_dist,
        smoothing=smoothing,
        sc=dict(setter_count),
        ss=dict(setter_sum),
        sd=dict(setter_dist),
        pc=dict(pair_count),
        ps=dict(pair_sum),
        pdist=dict(pair_dist),
        mean=mean,
        dist=dist,
    )


def safe_mean(vals, default=0.0):
    return float(np.mean(vals)) if len(vals) else float(default)


def summarize(vals, default: float):
    if not vals:
        vals = [default]
    a = np.asarray(vals, dtype=np.float32)
    top = np.sort(a)[::-1]
    return [
        float(a.mean()), float(a.std()), float(a.min()), float(a.max()),
        float(np.quantile(a, 0.10)), float(np.quantile(a, 0.25)), float(np.quantile(a, 0.50)),
        float(np.quantile(a, 0.75)), float(np.quantile(a, 0.90)), float(np.quantile(a, 0.95)),
        float(top[:2].mean()), float(top[:3].mean()), float(top[:5].mean() if len(top) >= 5 else top.mean()),
    ]


def build_no_usergrade_features(problems: List[Problem], tables, pair_hash_dim=32768, setter_hash_dim=4096):
    """Build graph/relation features. user_grade is intentionally not read anywhere here."""
    log('build no-user-grade sparse+dense graph/relation features')
    rows, cols, vals = [], [], []
    dense = []

    typed_off = 0
    roleless_off = typed_off + 3 * N_POS
    pair_off = roleless_off + N_POS
    geom_pair_off = pair_off + pair_hash_dim
    setter_off = geom_pair_off + pair_hash_dim
    dim = setter_off + setter_hash_dim

    iterator = range(len(problems))
    if tqdm is not None:
        iterator = tqdm(iterator, desc='features(no_user_grade)', dynamic_ncols=True)

    for r in iterator:
        p = problems[r]

        def add(c, v=1.0):
            rows.append(r)
            cols.append(int(c))
            vals.append(float(v))

        for h in p.start:
            add(typed_off + hold_col(h)); add(roleless_off + hold_col(h))
        for h in p.mid:
            add(typed_off + N_POS + hold_col(h)); add(roleless_off + hold_col(h))
        for h in p.end:
            add(typed_off + 2 * N_POS + hold_col(h)); add(roleless_off + hold_col(h))

        sid_hash = phash_int([ord(ch) for ch in str(p.setter_id)[:32]], setter_hash_dim)
        add(setter_off + sid_hash, 1.0)

        typed = p.typed
        rel_vals, geom_vals, conf_vals, edge_dy, edge_dx, dist_vals, adj_vals = [], [], [], [], [], [], []
        for i, (ha, ta) in enumerate(typed):
            for j, (hb, tb) in enumerate(typed):
                if i == j:
                    continue
                dx = hb[0] - ha[0]
                dy = hb[1] - ha[1]
                dist = math.hypot(dx, dy)
                if dist == 0 or dist > 10:
                    continue
                salt = 19*role_id(ta) + 37*role_id(tb) + 7*min(6, abs(dx)) + 13*max(0, min(16, dy+8)) + 23*min(10, int(round(dist)))
                add(pair_off + phash_int([ha[0], ha[1], hb[0], hb[1], salt], pair_hash_dim), 1.0)
                uh = sorted([ha, hb])
                add(geom_pair_off + phash_int([uh[0][0], uh[0][1], uh[1][0], uh[1][1], salt], pair_hash_dim), 1.0)

                keys = pair_keys(ha, ta, hb, tb)
                encs = [tables['mean'](tables['ps'], tables['pc'], k) for k in keys]
                cnts = [tables['pc'].get(k, 0) for k in keys]
                enc = float(np.mean(encs))
                conf = float(np.log1p(max(cnts)))
                geom = 0.13 * abs(dx) + 0.11 * max(0, dy) + 0.04 * dist
                rel_vals.append(enc + geom)
                geom_vals.append(geom)
                conf_vals.append(conf)
                edge_dx.append(dx); edge_dy.append(dy); dist_vals.append(dist)
                if abs(i - j) == 1:
                    adj_vals.append(enc + geom)

        holds = p.holds
        xs = [h[0] for h in holds] or [0]
        ys = [h[1] for h in holds] or [0]
        c = tables['sc'].get(p.setter_id, 0)
        setter_mean = (tables['ss'].get(p.setter_id, 0.0) + tables['smoothing'] * tables['gm']) / (c + tables['smoothing'])
        setter_dist = tables['dist'](tables['sd'], tables['sc'], p.setter_id)

        dense.append([
            math.log1p(float(p.repeats)), float(p.is_benchmark), float(p.is_master),
            float(c), float(setter_mean), *setter_dist.tolist(),
            len(p.start), len(p.mid), len(p.end), len(holds),
            np.mean(xs), np.std(xs), min(xs), max(xs), np.mean(ys), np.std(ys), min(ys), max(ys),
            max(xs) - min(xs), max(ys) - min(ys),
            *summarize(rel_vals, tables['gm']),
            *summarize(geom_vals, 0.0),
            *summarize(conf_vals, 0.0),
            *summarize(adj_vals, tables['gm']),
            safe_mean([abs(x) for x in edge_dx]),
            safe_mean([max(0, y) for y in edge_dy]),
            safe_mean(dist_vals),
            float(np.mean([d > 4 for d in dist_vals])) if dist_vals else 0.0,
            float(np.mean([y > 3 for y in edge_dy])) if edge_dy else 0.0,
        ])

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(problems), dim), dtype=np.float32)
    D = np.asarray(dense, dtype=np.float32)
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    return X, D


def sanitize_probs(P: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    if P.ndim != 2 or P.shape[1] != NUM_CLASSES:
        raise ValueError(f"expected [n,{NUM_CLASSES}], got {P.shape}")
    P[P < 0] = 0.0
    s = P.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(s[:, 0])) | (s[:, 0] <= eps)
    if np.any(bad):
        P[bad] = 1.0 / NUM_CLASSES
        s = P.sum(axis=1, keepdims=True)
    return P / np.clip(s, eps, None)


def probs_align(clf, X) -> np.ndarray:
    raw = clf.predict_proba(X)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros((raw.shape[0], NUM_CLASSES), dtype=np.float32)
    for j, c in enumerate(clf.classes_):
        ci = int(c)
        if 0 <= ci < NUM_CLASSES:
            out[:, ci] = raw[:, j]
    return sanitize_probs(out)


def metric(y, pred, prob=None):
    y = np.asarray(y)
    pred = np.asarray(pred)
    out = {
        'exact_acc': float(accuracy_score(y, pred)),
        'micro_f1': float(f1_score(y, pred, average='micro', zero_division=0)),
        'relaxed_acc': float(np.mean(np.abs(y - pred) <= 1)),
        'within2_acc': float(np.mean(np.abs(y - pred) <= 2)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, pred, average='weighted', zero_division=0)),
        'mse_argmax': float(np.mean((y - pred) ** 2)),
        'mae_argmax': float(np.mean(np.abs(y - pred))),
    }
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=list(range(NUM_CLASSES)), zero_division=0)
    rare = [5, 6, 7, 8, 9]
    out['rare_f1_5_9'] = float(np.mean([f1[i] for i in rare]))
    out['rare_recall_5_9'] = float(np.mean([rc[i] for i in rare]))
    out['rare_precision_5_9'] = float(np.mean([pr[i] for i in rare]))
    if prob is not None:
        exp = sanitize_probs(prob) @ np.arange(NUM_CLASSES)
        out['mse_expected'] = float(np.mean((exp - y) ** 2))
        out['mae_expected'] = float(np.mean(np.abs(exp - y)))
    return out


def graph_knn_probs(Xtr, ytr, Xev, k, batch=2048):
    nnm = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=-1).fit(Xtr)
    out = np.zeros((Xev.shape[0], NUM_CLASSES), dtype=np.float32)
    rng = range(0, Xev.shape[0], batch)
    if tqdm is not None:
        rng = tqdm(list(rng), desc=f'knn{k}', dynamic_ncols=True)
    for s in rng:
        e = min(s + batch, Xev.shape[0])
        dist, ind = nnm.kneighbors(Xev[s:e], return_distance=True)
        w = np.clip(1.0 - dist, 0, 1) + 1e-4
        for r in range(e - s):
            labs = ytr[ind[r]]
            for jj, lab in enumerate(labs):
                out[s + r, int(lab)] += float(w[r, jj])
            out[s + r] /= max(1e-9, out[s + r].sum())
    return sanitize_probs(out)


def exact_bias_search(Pv, yv, Pt, trials=30000, seed=471, scale=0.28):
    rng = np.random.default_rng(seed)
    lv = np.log(np.clip(sanitize_probs(Pv), 1e-9, 1.0))
    lt = np.log(np.clip(sanitize_probs(Pt), 1e-9, 1.0))
    best_score = -1.0
    best_bias = np.zeros(NUM_CLASSES, dtype=np.float32)
    it = range(trials)
    if tqdm is not None:
        it = tqdm(it, desc='bias', dynamic_ncols=True)
    for _ in it:
        b = rng.normal(0.0, scale, size=NUM_CLASSES).astype(np.float32)
        score = accuracy_score(yv, (lv + b).argmax(axis=1))
        if score > best_score:
            best_score = float(score)
            best_bias = b
    Pt2 = np.exp(lt + best_bias)
    return best_score, sanitize_probs(Pt2)


def random_blend(bank: Dict[str, Dict[str, np.ndarray]], yv: np.ndarray, trials: int, seed: int):
    names = list(bank.keys())
    if len(names) < 2 or trials <= 0:
        return None
    rng = np.random.default_rng(seed)
    best = None
    it = range(trials)
    if tqdm is not None:
        it = tqdm(it, desc='blend', dynamic_ncols=True)
    for _ in it:
        m = int(rng.integers(2, min(9, len(names)) + 1))
        chosen = list(rng.choice(names, size=m, replace=False))
        w = rng.dirichlet(np.ones(m) * 0.45)
        Pv = sum(float(a) * bank[n]['val'] for a, n in zip(w, chosen))
        score = accuracy_score(yv, Pv.argmax(axis=1))
        if best is None or score > best['val_exact']:
            Pt = sum(float(a) * bank[n]['test'] for a, n in zip(w, chosen))
            best = {'val_exact': float(score), 'members': chosen, 'weights': w.tolist(), 'test': sanitize_probs(Pt)}
    return best


# ----------------------------- custom edge-aware GAT -----------------------------

if TORCH_AVAILABLE:
    @dataclass
    class GraphItem:
        x: torch.Tensor
        edge_attr: torch.Tensor
        edge_mask: torch.Tensor
        dense: np.ndarray
        y: int
        idx: int

    class GraphDataset(Dataset):
        def __init__(self, graphs, indices):
            self.graphs = graphs
            self.indices = list(map(int, indices))
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.graphs[self.indices[i]]

    def collate_graphs(items):
        B = len(items)
        N = max(it.x.shape[0] for it in items)
        Fd = items[0].x.shape[1]
        Ed = items[0].edge_attr.shape[-1]
        x = torch.zeros(B, N, Fd)
        ea = torch.zeros(B, N, N, Ed)
        em = torch.zeros(B, N, N, dtype=torch.bool)
        nm = torch.zeros(B, N, dtype=torch.bool)
        dense = torch.tensor(np.stack([it.dense for it in items]), dtype=torch.float32)
        y = torch.tensor([it.y for it in items], dtype=torch.long)
        for b, it in enumerate(items):
            n = it.x.shape[0]
            x[b, :n] = it.x
            ea[b, :n, :n] = it.edge_attr
            em[b, :n, :n] = it.edge_mask
            nm[b, :n] = True
        return {'x': x, 'edge_attr': ea, 'edge_mask': em, 'node_mask': nm, 'dense': dense, 'y': y}

    def build_graphs(problems, tables, dense_scaled, max_dist=10.0):
        log('build custom no-user-grade edge-aware GAT graphs')
        graphs = []
        edge_dim = 25
        it = enumerate(problems)
        if tqdm is not None:
            it = tqdm(list(it), desc='gat_graphs_no_user_grade', dynamic_ncols=True)
        for idx, p in it:
            typed = list(p.typed)
            xs = []
            for h, t in typed:
                typ = role_id(t)
                xs.append([
                    h[0] / 10.0, h[1] / 17.0,
                    float(typ == 0), float(typ == 1), float(typ == 2),
                    len(p.start) / 4.0, len(p.mid) / 12.0, len(p.end) / 4.0,
                ])
            if not xs:
                xs = [[0.0] * 8]
                typed = [((0, 0), 'm')]
            n = len(xs)
            edge_attr = np.zeros((n, n, edge_dim), dtype=np.float32)
            edge_mask = np.zeros((n, n), dtype=bool)
            for i, (ha, ta) in enumerate(typed):
                for j, (hb, tb) in enumerate(typed):
                    dx = hb[0] - ha[0]
                    dy = hb[1] - ha[1]
                    distv = math.hypot(dx, dy)
                    if i == j:
                        attr = [0.0] * edge_dim
                        attr[7] = 1.0
                        edge_attr[i, j] = attr
                        edge_mask[i, j] = True
                        continue
                    if distv <= 0 or distv > max_dist:
                        continue
                    keys = pair_keys(ha, ta, hb, tb)
                    means = [tables['mean'](tables['ps'], tables['pc'], k) for k in keys]
                    dists = [tables['dist'](tables['pdist'], tables['pc'], k) for k in keys]
                    enc = float(np.mean(means)) / 9.0
                    conf = float(np.mean([min(1.0, tables['pc'].get(k, 0) / 20.0) for k in keys]))
                    geom_hard = min(1.0, (0.13 * abs(dx) + 0.11 * max(0, dy) + 0.04 * distv) / 2.2)
                    st = role_id(ta)
                    dt = role_id(tb)
                    dist_mean = np.mean(dists, axis=0)
                    attr = [
                        dx / 10.0, dy / 17.0, abs(dx) / 10.0, abs(dy) / 17.0, distv / max_dist,
                        float(dy > 0), float(dy < 0), float(abs(dx) >= 4),
                        float(st == 0), float(st == 1), float(st == 2), float(dt == 0),
                        enc, conf, geom_hard,
                        *dist_mean.tolist(),
                    ]
                    edge_attr[i, j] = np.asarray(attr, dtype=np.float32)
                    edge_mask[i, j] = True
            graphs.append(GraphItem(
                torch.tensor(xs, dtype=torch.float32),
                torch.tensor(edge_attr, dtype=torch.float32),
                torch.tensor(edge_mask, dtype=torch.bool),
                dense_scaled[idx].astype(np.float32),
                int(p.label),
                idx,
            ))
        return graphs, graphs[0].x.shape[1], graphs[0].edge_attr.shape[-1]

    class EdgeAwareGATLayer(nn.Module):
        def __init__(self, hidden, edge_dim, dropout):
            super().__init__()
            self.lin = nn.Linear(hidden, hidden, bias=False)
            self.edge = nn.Linear(edge_dim, hidden, bias=False)
            self.a_src = nn.Parameter(torch.empty(hidden))
            self.a_dst = nn.Parameter(torch.empty(hidden))
            self.a_edge = nn.Parameter(torch.empty(hidden))
            self.out = nn.Linear(hidden, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.drop = nn.Dropout(dropout)
            self.reset_parameters()
        def reset_parameters(self):
            self.lin.reset_parameters(); self.edge.reset_parameters(); self.out.reset_parameters()
            nn.init.xavier_uniform_(self.a_src.view(1, -1))
            nn.init.xavier_uniform_(self.a_dst.view(1, -1))
            nn.init.xavier_uniform_(self.a_edge.view(1, -1))
        def forward(self, h, ea, em, nm):
            z = self.lin(h)
            e = self.edge(ea)
            score = (z.unsqueeze(2) * self.a_src).sum(-1) + (z.unsqueeze(1) * self.a_dst).sum(-1) + (e * self.a_edge).sum(-1)
            score = F.leaky_relu(score, 0.2).masked_fill(~em, -1e9)
            alpha = torch.softmax(score, dim=1).masked_fill(~em, 0.0)
            msg = z.unsqueeze(2) + e
            out = (alpha.unsqueeze(-1) * msg).sum(dim=1)
            out = self.out(self.drop(out))
            return self.norm(h + out) * nm.unsqueeze(-1).float()

    class CustomGAT(nn.Module):
        def __init__(self, node_dim, edge_dim, dense_dim, hidden, dropout):
            super().__init__()
            self.node = nn.Linear(node_dim, hidden)
            self.g1 = EdgeAwareGATLayer(hidden, edge_dim, dropout)
            self.g2 = EdgeAwareGATLayer(hidden, edge_dim, dropout)
            self.graph_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.dense_head = nn.Sequential(nn.Linear(dense_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
            self.cls = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, NUM_CLASSES))
        def forward(self, b, device):
            x = b['x'].to(device)
            ea = b['edge_attr'].to(device)
            em = b['edge_mask'].to(device)
            nm = b['node_mask'].to(device)
            d = b['dense'].to(device)
            h = F.elu(self.node(x)) * nm.unsqueeze(-1).float()
            h = F.elu(self.g1(h, ea, em, nm))
            h = F.elu(self.g2(h, ea, em, nm))
            denom = nm.sum(dim=1, keepdim=True).clamp_min(1).float()
            mean = h.sum(dim=1) / denom
            mx = h.masked_fill(~nm.unsqueeze(-1), -1e9).max(dim=1).values
            mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
            gh = self.graph_head(torch.cat([mean, mx], dim=1))
            dh = self.dense_head(d)
            return self.cls(torch.cat([gh, dh], dim=1))

    def eval_gat_model(model, loader, device):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in loader:
                logits = model(b, device)
                ps.append(F.softmax(logits, dim=1).cpu().numpy())
                ys.extend(b['y'].numpy().tolist())
        return np.array(ys, dtype=np.int64), sanitize_probs(np.vstack(ps))

    def train_gat(name, graphs, tr, va, te, node_dim, edge_dim, dense_dim, args, seed):
        log(f'train {name}')
        seed_all(seed)
        device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
        model = CustomGAT(node_dim, edge_dim, dense_dim, args.gat_hidden, args.gat_dropout).to(device)
        ytr = np.array([graphs[int(i)].y for i in tr])
        counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float32)
        w = np.ones(NUM_CLASSES, dtype=np.float32)
        present = counts > 0
        w[present] = np.sqrt(len(ytr) / (present.sum() * counts[present]))
        w[present] /= max(1e-9, w[present].mean())
        ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=args.gat_lr, weight_decay=args.gat_weight_decay)
        tr_loader = DataLoader(GraphDataset(graphs, tr), batch_size=args.gat_batch_size, shuffle=True, collate_fn=collate_graphs)
        va_loader = DataLoader(GraphDataset(graphs, va), batch_size=args.gat_batch_size, shuffle=False, collate_fn=collate_graphs)
        te_loader = DataLoader(GraphDataset(graphs, te), batch_size=args.gat_batch_size, shuffle=False, collate_fn=collate_graphs)
        best = None
        it = range(1, args.gat_epochs + 1)
        if tqdm is not None:
            it = tqdm(it, desc=name, dynamic_ncols=True)
        for ep in it:
            model.train()
            total = 0.0
            nobs = 0
            for b in tr_loader:
                opt.zero_grad(set_to_none=True)
                logits = model(b, device)
                yy = b['y'].to(device)
                prob = F.softmax(logits, dim=1)
                exp = (prob * torch.arange(NUM_CLASSES, device=device).float()).sum(1)
                loss = ce(logits, yy) + args.ordinal_weight * F.mse_loss(exp, yy.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total += float(loss.item()) * len(yy)
                nobs += len(yy)
            yv, pv = eval_gat_model(model, va_loader, device)
            vexact = accuracy_score(yv, pv.argmax(1))
            if best is None or vexact > best[0]:
                yt, pt = eval_gat_model(model, te_loader, device)
                best = (vexact, ep, pv.copy(), pt.copy())
            if ep % max(1, args.gat_log_every) == 0:
                log(f'{name} ep={ep} loss={total/max(1,nobs):.4f} val_exact={vexact:.4f}')
        return {'name': name, 'val_exact': best[0], 'epoch': best[1], 'val_probs': best[2], 'test_probs': best[3]}


def add_row(rows: List[dict], name: str, ytest: np.ndarray, Ptest: np.ndarray, **extra) -> None:
    Ptest = sanitize_probs(Ptest)
    row = {'model': name, **extra, **metric(ytest, Ptest.argmax(axis=1), Ptest)}
    rows.append(row)
    log(f"{name} test_exact={row['exact_acc']:.4f} relaxed={row['relaxed_acc']:.4f} macro={row['macro_f1']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('moonGen_scrape_2016_final'))
    ap.add_argument('--out', type=Path, default=Path('runs_final_no_usergrade_standalone'))
    ap.add_argument('--seed', type=int, default=471)
    ap.add_argument('--target-exact', type=float, default=0.60)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--test-size', type=float, default=0.20)
    ap.add_argument('--val-size-in-trainval', type=float, default=0.20)
    ap.add_argument('--pair-hash-dim', type=int, default=32768)
    ap.add_argument('--setter-hash-dim', type=int, default=4096)
    ap.add_argument('--svd-dim', type=int, default=512)
    ap.add_argument('--extra-trees-estimators', type=int, default=1200)
    ap.add_argument('--hgb-iters', type=int, default=900)
    ap.add_argument('--gat-epochs', type=int, default=100)
    ap.add_argument('--gat-hidden', type=int, default=160)
    ap.add_argument('--gat-seeds', type=int, default=3)
    ap.add_argument('--gat-batch-size', type=int, default=128)
    ap.add_argument('--gat-lr', type=float, default=8e-4)
    ap.add_argument('--gat-weight-decay', type=float, default=1e-4)
    ap.add_argument('--gat-dropout', type=float, default=0.25)
    ap.add_argument('--ordinal-weight', type=float, default=0.02)
    ap.add_argument('--gat-log-every', type=int, default=10)
    ap.add_argument('--bias-trials', type=int, default=70000)
    ap.add_argument('--blend-trials', type=int, default=120000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--skip-hgb', action='store_true')
    ap.add_argument('--skip-gat', action='store_true')
    ap.add_argument('--skip-knn', action='store_true')
    args = ap.parse_args()

    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    log('NO USER_GRADE: disabled as feature, hard override, routing, and diagnostics')
    log(f'load {args.data}')
    problems = load_problems(args.data)
    if args.max_samples and args.max_samples < len(problems):
        y0 = np.array([p.label for p in problems])
        try:
            _, sub = train_test_split(np.arange(len(problems)), test_size=args.max_samples, random_state=args.seed, stratify=y0)
        except ValueError:
            rng = np.random.default_rng(args.seed)
            sub = rng.choice(len(problems), size=args.max_samples, replace=False)
        problems = [problems[int(i)] for i in sub]
        log(f'max_samples={len(problems)}')

    y = np.array([p.label for p in problems], dtype=np.int64)
    all_idx = np.arange(len(problems))
    try:
        trainval, test = train_test_split(all_idx, test_size=args.test_size, random_state=args.seed, stratify=y)
    except ValueError:
        log('stratified test split failed; using unstratified split')
        trainval, test = train_test_split(all_idx, test_size=args.test_size, random_state=args.seed, stratify=None)
    try:
        train, val = train_test_split(trainval, test_size=args.val_size_in_trainval, random_state=args.seed + 1, stratify=y[trainval])
    except ValueError:
        log('stratified val split failed; using unstratified split')
        train, val = train_test_split(trainval, test_size=args.val_size_in_trainval, random_state=args.seed + 1, stratify=None)
    log(f'split train={len(train)} val={len(val)} test={len(test)} labels={dict(Counter(y.tolist()))}')

    tables = build_tables(problems, train)
    Xs, D = build_no_usergrade_features(problems, tables, args.pair_hash_dim, args.setter_hash_dim)
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(D[train])
    Dscaled = scaler.transform(D).astype(np.float32)
    X = sparse.hstack([Xs, sparse.csr_matrix(Dscaled)], format='csr')
    ytr, yv, yt = y[train], y[val], y[test]
    log(f'features sparse={Xs.shape} dense={D.shape} combined={X.shape}')

    rows: List[dict] = []
    bank: Dict[str, Dict[str, np.ndarray]] = {}

    def register(name: str, Pv: np.ndarray, Pt: np.ndarray, **extra):
        Pv = sanitize_probs(Pv)
        Pt = sanitize_probs(Pt)
        bank[name] = {'val': Pv, 'test': Pt}
        add_row(rows, name, yt, Pt, val_exact=float(accuracy_score(yv, Pv.argmax(axis=1))), **extra)

    for k in ([] if args.skip_knn else [3, 5, 11, 25, 51]):
        log(f'fit graph_knn{k}')
        Pv = graph_knn_probs(Xs[train], ytr, Xs[val], k)
        Pt = graph_knn_probs(Xs[train], ytr, Xs[test], k)
        register(f'graph_knn{k}', Pv, Pt)

    for alpha in [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
        name = f'sgd_l2_alpha{alpha:g}'
        log(f'fit {name}')
        clf = SGDClassifier(loss='log_loss', alpha=alpha, penalty='l2', max_iter=3500, tol=1e-4,
                            random_state=args.seed, n_jobs=-1, class_weight=None)
        clf.fit(X[train], ytr)
        register(name, probs_align(clf, X[val]), probs_align(clf, X[test]))

    ncomp = max(16, min(args.svd_dim, X[train].shape[1] - 1, X[train].shape[0] - 1))
    log(f'fit TruncatedSVD n_components={ncomp}')
    svd = TruncatedSVD(n_components=ncomp, random_state=args.seed)
    Ztr = svd.fit_transform(X[train])
    Zv = svd.transform(X[val])
    Zt = svd.transform(X[test])
    Ztr = np.hstack([Ztr, Dscaled[train]])
    Zv = np.hstack([Zv, Dscaled[val]])
    Zt = np.hstack([Zt, Dscaled[test]])
    zscaler = StandardScaler().fit(Ztr)
    Ztr_s = zscaler.transform(Ztr)
    Zv_s = zscaler.transform(Zv)
    Zt_s = zscaler.transform(Zt)

    for lr, leaf, l2 in ([] if args.skip_hgb else [(0.03, 31, 0.01), (0.04, 31, 0.02), (0.03, 63, 0.01), (0.025, 127, 0.03)]):
        name = f'hgb_lr{lr}_leaf{leaf}_l2{l2}'
        log(f'fit {name}')
        clf = HistGradientBoostingClassifier(max_iter=args.hgb_iters, learning_rate=lr,
                                             max_leaf_nodes=leaf, l2_regularization=l2,
                                             random_state=args.seed, early_stopping=False)
        clf.fit(Ztr, ytr)
        register(name, probs_align(clf, Zv), probs_align(clf, Zt))

    for leaf in [1, 2, 4]:
        name = f'extra_trees{args.extra_trees_estimators}_leaf{leaf}'
        log(f'fit {name}')
        clf = ExtraTreesClassifier(n_estimators=args.extra_trees_estimators, max_features='sqrt',
                                   min_samples_leaf=leaf, n_jobs=-1, random_state=args.seed,
                                   class_weight=None)
        clf.fit(X[train], ytr)
        register(name, probs_align(clf, X[val]), probs_align(clf, X[test]))

    if args.skip_gat or args.gat_seeds <= 0:
        log('skip custom edge-aware GAT branch')
    elif not TORCH_AVAILABLE:
        log('torch is not available; skip custom edge-aware GAT branch')
    else:
        graphs, nd, ed = build_graphs(problems, tables, Dscaled)
        for i in range(args.gat_seeds):
            seed = args.seed + 101 * i
            gat = train_gat(f'edge_gat_seed{seed}', graphs, train, val, test,
                            nd, ed, Dscaled.shape[1], args, seed)
            register(gat['name'], gat['val_probs'], gat['test_probs'], epoch=gat['epoch'])

    log('fit validation meta stacks')
    names = list(bank.keys())
    MetaV = np.nan_to_num(np.hstack([bank[n]['val'] for n in names]), nan=0.0, posinf=0.0, neginf=0.0)
    MetaT = np.nan_to_num(np.hstack([bank[n]['test'] for n in names]), nan=0.0, posinf=0.0, neginf=0.0)
    for C in [0.05, 0.15, 0.35, 0.7, 1.5]:
        name = f'val_meta_C{C}'
        clf = LogisticRegression(C=C, max_iter=3000, solver='lbfgs')
        clf.fit(MetaV, yv)
        Pv = probs_align(clf, MetaV)
        Pt = probs_align(clf, MetaT)
        register(name, Pv, Pt)
        bv, Ptb = exact_bias_search(Pv, yv, Pt, trials=args.bias_trials, seed=args.seed + int(C * 1000))
        register(name + '_bias', Pv, Ptb, bias_val_exact=bv)

    log(f'random blend trials={args.blend_trials}')
    blend = random_blend(bank, yv, args.blend_trials, args.seed)
    if blend is not None:
        Pt = blend['test']
        row = {
            'model': 'random_prob_blend',
            'val_exact': blend['val_exact'],
            'members': ';'.join(blend['members']),
            'weights': ';'.join(f'{w:.4f}' for w in blend['weights']),
            **metric(yt, Pt.argmax(axis=1), Pt),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['exact_acc', 'relaxed_acc', 'macro_f1'], ascending=False)
    df['target_exact_met'] = df['exact_acc'] >= args.target_exact
    out_csv = args.out / 'leaderboard_final_no_usergrade_standalone.csv'
    df.to_csv(out_csv, index=False)
    best = df.iloc[0].to_dict()
    (args.out / 'best_summary.json').write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding='utf-8')
    (args.out / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str, ensure_ascii=False), encoding='utf-8')
    log('leaderboard top 30:')
    with pd.option_context('display.max_columns', None, 'display.width', 260):
        print(df.head(30).to_string(index=False), flush=True)
    log(f"target {'met' if bool(best['target_exact_met']) else 'not met'}: best exact={best['exact_acc']:.4f} target={args.target_exact:.4f}")
    log(f'saved {out_csv}')


if __name__ == '__main__':
    main()
