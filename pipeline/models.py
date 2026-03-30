"""
Model definitions:
  - SpatialGNN   : 2-layer GraphSAGE for spatial transcriptomics regression
  - MLPBaseline  : 3-layer MLP (ignores graph structure)
  - RandomForestBaseline : sklearn MultiOutputRegressor wrapper
  - LinearBaseline: sklearn MultiOutputRegressor with Ridge regression
"""

import numpy as np

# ── PyTorch / PyG imports ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# ── Sklearn baselines ─────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


# ── GNN ───────────────────────────────────────────────────────────────────────

if HAS_TORCH and HAS_PYG:

    class SpatialGNN(nn.Module):
        """
        2-layer GraphSAGE for node-level multi-task regression.

        Architecture
        ------------
        SAGEConv(in → 256) → BN → ReLU → Dropout(0.3)
        SAGEConv(256 → 128) → BN → ReLU → Dropout(0.3)
        Linear(128 → out_channels)
        """

        def __init__(self, in_channels: int, out_channels: int = 5,
                     hidden: int = 256, dropout: float = 0.3):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden)
            self.bn1 = nn.BatchNorm1d(hidden)
            self.conv2 = SAGEConv(hidden, hidden // 2)
            self.bn2 = nn.BatchNorm1d(hidden // 2)
            self.head = nn.Linear(hidden // 2, out_channels)
            self.dropout = dropout

        def forward(self, x, edge_index, edge_attr=None):
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            return self.head(x)

    class MLPBaseline(nn.Module):
        """
        3-layer MLP: treats each spot independently (no spatial context).

        2000 → 512 → 256 → 5
        """

        def __init__(self, in_channels: int, out_channels: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_channels, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, out_channels),
            )

        def forward(self, x, edge_index=None, edge_attr=None):
            return self.net(x)

elif HAS_TORCH:
    # PyG unavailable — pure-PyTorch GCN via sparse multiplication

    class SpatialGNN(nn.Module):  # type: ignore[no-redef]
        """Sparse-matmul GCN fallback (no torch_geometric required)."""

        def __init__(self, in_channels: int, out_channels: int = 5,
                     hidden: int = 256, dropout: float = 0.3):
            super().__init__()
            self.W1 = nn.Linear(in_channels, hidden)
            self.bn1 = nn.BatchNorm1d(hidden)
            self.W2 = nn.Linear(hidden, hidden // 2)
            self.bn2 = nn.BatchNorm1d(hidden // 2)
            self.head = nn.Linear(hidden // 2, out_channels)
            self.dropout = dropout

        def _propagate(self, x, adj):
            """Normalised adjacency × feature matrix."""
            import torch
            if adj is not None:
                return torch.sparse.mm(adj, x)
            return x

        def forward(self, x, edge_index=None, edge_attr=None):
            import torch
            # Build sparse adj on first call
            if edge_index is not None:
                n = x.shape[0]
                vals = torch.ones(edge_index.shape[1], device=x.device)
                adj = torch.sparse_coo_tensor(edge_index, vals, (n, n))
                # Row-normalise
                deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)
                vals = vals / deg[edge_index[0]]
                adj = torch.sparse_coo_tensor(edge_index, vals, (n, n))
            else:
                adj = None

            h = self._propagate(x, adj)
            h = F.relu(self.bn1(self.W1(h)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self._propagate(h, adj)
            h = F.relu(self.bn2(self.W2(h)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            return self.head(h)

    class MLPBaseline(nn.Module):  # type: ignore[no-redef]
        def __init__(self, in_channels: int, out_channels: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_channels, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, out_channels),
            )

        def forward(self, x, edge_index=None, edge_attr=None):
            return self.net(x)

else:
    # No torch at all — stub classes so imports don't break
    class SpatialGNN:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise RuntimeError("PyTorch is required for SpatialGNN")

    class MLPBaseline:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise RuntimeError("PyTorch is required for MLPBaseline")


# ── Sklearn baselines ─────────────────────────────────────────────────────────

class RandomForestBaseline:
    """Multi-output Random Forest (sklearn)."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, seed: int = 42):
        base = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        )
        self.model = MultiOutputRegressor(base, n_jobs=1)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).astype(np.float32)


class LinearBaseline:
    """Multi-output Ridge regression."""

    def __init__(self, alpha: float = 1.0):
        base = Ridge(alpha=alpha)
        self.model = MultiOutputRegressor(base)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).astype(np.float32)
