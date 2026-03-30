"""
Build a spatial tissue graph for PyTorch Geometric.

Each tissue spot is a node; edges connect each spot to its K=6
nearest spatial neighbors (Euclidean distance on x/y coordinates).
"""

import numpy as np
from pathlib import Path

# Try torch_geometric; fall back to plain dict if unavailable
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[GRAPH] torch_geometric not available — using sparse-matrix fallback")


def build_knn_edges(coords: np.ndarray, k: int = 6):
    """
    Compute KNN edges from 2-D spatial coordinates.

    Returns
    -------
    edge_index : (2, E) int64 array
    edge_weight: (E,) float32 array   (1 / (1 + distance))
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    # k+1 because the nearest neighbour of a point is itself
    distances, indices = tree.query(coords, k=k + 1)

    src_list, dst_list, w_list = [], [], []
    for i, (dists, nbrs) in enumerate(zip(distances, indices)):
        for d, j in zip(dists[1:], nbrs[1:]):   # skip self (index 0)
            src_list.append(i)
            dst_list.append(j)
            w_list.append(1.0 / (1.0 + d))
            # Add reverse edge for undirected graph
            src_list.append(j)
            dst_list.append(i)
            w_list.append(1.0 / (1.0 + d))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(w_list, dtype=np.float32)

    # Deduplicate
    pairs = set()
    keep = []
    for idx in range(edge_index.shape[1]):
        key = (edge_index[0, idx], edge_index[1, idx])
        if key not in pairs:
            pairs.add(key)
            keep.append(idx)
    edge_index = edge_index[:, keep]
    edge_weight = edge_weight[keep]

    return edge_index, edge_weight


def build_graph(
    expression: np.ndarray,
    metal_targets: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> "Data | dict":
    """
    Build the tissue spatial graph.

    Parameters
    ----------
    expression   : (N, G) gene expression matrix (already normalised)
    metal_targets: (N, 5) metal abundance targets
    coords       : (N, 2) spatial coordinates
    k            : number of nearest neighbours per spot

    Returns
    -------
    PyG Data object (or plain dict if PyG unavailable)
    """
    n_spots = expression.shape[0]
    print(f"[GRAPH] Building KNN graph: {n_spots} nodes, k={k} …")

    edge_index, edge_weight = build_knn_edges(coords, k=k)
    print(f"[GRAPH] Edges: {edge_index.shape[1]:,}")

    if HAS_PYG:
        import torch
        data = Data(
            x=torch.tensor(expression, dtype=torch.float32),
            y=torch.tensor(metal_targets, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_weight, dtype=torch.float32).unsqueeze(1),
            pos=torch.tensor(coords, dtype=torch.float32),
        )
        data.num_nodes = n_spots
        print("[GRAPH] PyG Data object created")
        return data
    else:
        # Fallback: return plain dict with numpy arrays
        return {
            "x": expression,
            "y": metal_targets,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "pos": coords,
            "num_nodes": n_spots,
        }


def train_val_test_split(
    coords: np.ndarray,
    n_spots: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Spatial block split: divide tissue into quadrants, assign
    train/val/test spatially rather than randomly.
    """
    rng = np.random.default_rng(seed)

    # Assign each spot to a quadrant based on its position
    x_med = np.median(coords[:, 0])
    y_med = np.median(coords[:, 1])

    quadrant = (
        (coords[:, 0] >= x_med).astype(int) * 2
        + (coords[:, 1] >= y_med).astype(int)
    )  # 0,1,2,3

    indices = np.arange(n_spots)
    q_shuffled = []
    for q in range(4):
        q_idx = indices[quadrant == q]
        rng.shuffle(q_idx)
        q_shuffled.append(q_idx)

    all_idx = np.concatenate(q_shuffled)
    n_train = int(n_spots * train_ratio)
    n_val = int(n_spots * val_ratio)

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train: n_train + n_val]
    test_idx = all_idx[n_train + n_val:]

    # Return as boolean masks
    train_mask = np.zeros(n_spots, dtype=bool)
    val_mask = np.zeros(n_spots, dtype=bool)
    test_mask = np.zeros(n_spots, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask
