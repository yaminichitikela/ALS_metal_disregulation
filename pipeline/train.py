"""
Steps 5 & 6 from Hands-On ML (Géron):

  Step 5 — Shortlist Promising Models
    "Train many quick-and-dirty models from different categories using
     standard parameters. Measure and compare their performance using
     N-fold cross-validation." (cross_val_score)

  Step 6 — Fine-Tune the System
    "Fine-tune the hyperparameters using cross-validation.
     Treat data transformation choices as hyperparameters.
     Use RandomizedSearchCV for large search spaces." (RandomizedSearchCV)
    "Analyze the best models and their errors."
    "Evaluate on the test set — estimate the generalisation error."

Saves:
  results/best_gnn.pt           GNN checkpoint
  results/best_mlp.pt           MLP checkpoint
  results/cv_scores.json        Step 5 cross-validation results
  results/comparison.json       Final test-set R² (all models)
  results/predictions.npz       Test predictions for Streamlit
  results/loss_curves.json      GNN/MLP epoch loss history
  results/feature_importance.json  RF feature importances (Step 6)
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import randint

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

RESULTS_DIR = Path(__file__).parent.parent / "results"
METALS = ["Fe", "Zn", "Cu", "Mn", "Mg"]

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r2_per_metal(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    scores = {}
    for i, m in enumerate(METALS):
        scores[m] = float(round(r2_score(y_true[:, i], y_pred[:, i]), 4))
    scores["mean"] = float(round(np.mean(list(scores.values())), 4))
    return scores


def _to_numpy(t):
    if HAS_TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# ── Book Step 4: sklearn Preparation Pipeline ─────────────────────────────────
# "Use Pipeline objects to prepare data consistently for train and test."

def make_prep_pipeline() -> Pipeline:
    """
    Numeric pipeline following the book's pattern:
      SimpleImputer(median) → StandardScaler

    Applied to the expression matrix before feeding into sklearn models.
    (GNN does its own normalisation via BatchNorm layers.)
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # handle any NaN
        ("scaler",  StandardScaler()),                    # zero mean, unit var
    ])


# ── Book Step 5: Cross-Validation ─────────────────────────────────────────────

def cross_validate_models(
    expression: np.ndarray,
    metal_targets: np.ndarray,
    train_mask: np.ndarray,
    fast_mode: bool = False,
) -> dict:
    """
    Book Step 5: "Train many quick-and-dirty models from different
    categories and compare them using N-fold cross-validation."

    Uses KFold(n_splits=5) on the training set only.
    Reports mean R² ± std for each model, averaged over all 5 metals.
    """
    n_splits = 3 if fast_mode else 5
    print(f"\n[CV] Step 5: {n_splits}-fold cross-validation on training set …")

    X_train = expression[train_mask]
    y_train = metal_targets[train_mask]

    # Book: fit the pipeline on training folds only (no data leakage)
    prep = make_prep_pipeline()
    X_train_prep = prep.fit_transform(X_train)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Models to compare (book step 5: "diverse categories")
    candidates = {
        "RandomForest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=50 if fast_mode else 100,
                                  max_depth=10, random_state=42, n_jobs=-1)
        ),
        "Ridge": MultiOutputRegressor(Ridge(alpha=1.0)),
    }

    cv_results = {}
    for name, model in candidates.items():
        # cross_val_score returns one score per fold
        # We score per-output then average (book: use neg_mean_squared_error
        # for regression; we use r2 here for interpretability)
        fold_r2s = []
        for train_idx, val_idx in kf.split(X_train_prep):
            model.fit(X_train_prep[train_idx], y_train[train_idx])
            preds = model.predict(X_train_prep[val_idx])
            # Mean R² across all 5 metals for this fold
            per_metal = [r2_score(y_train[val_idx, m], preds[:, m]) for m in range(5)]
            fold_r2s.append(np.mean(per_metal))

        mean_r2 = float(round(np.mean(fold_r2s), 4))
        std_r2  = float(round(np.std(fold_r2s),  4))
        cv_results[name] = {"mean_r2": mean_r2, "std_r2": std_r2, "folds": fold_r2s}

        # Book format: "Scores: mean ± std"
        print(f"  {name:<20}  R² = {mean_r2:.4f} ± {std_r2:.4f}")

    # Book: "save cross-validation scores so you can compare later"
    with open(RESULTS_DIR / "cv_scores.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"[CV] Saved → results/cv_scores.json")

    return cv_results


# ── Book Step 6a: Fine-Tune with RandomizedSearchCV ───────────────────────────

def fine_tune_rf(
    expression: np.ndarray,
    metal_targets: np.ndarray,
    train_mask: np.ndarray,
    fast_mode: bool = False,
) -> tuple:
    """
    Book Step 6: "Fine-tune hyperparameters using cross-validation.
    Use RandomizedSearchCV when the hyperparameter search space is large."

    Returns (best_rf_pipeline, best_params, feature_importances)
    """
    print("\n[TUNE] Step 6: RandomizedSearchCV for Random Forest …")

    X_train = expression[train_mask]
    y_train = metal_targets[train_mask]

    # Book pattern: include preprocessing inside the pipeline so
    # CV never leaks test-fold statistics into training
    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   MultiOutputRegressor(
            RandomForestRegressor(random_state=42, n_jobs=-1)
        )),
    ])

    # Book: "specify hyperparameter distributions, not discrete grids"
    param_dist = {
        "model__estimator__n_estimators": randint(50, 300),
        "model__estimator__max_depth":    [5, 10, 20, None],
        "model__estimator__min_samples_split": randint(2, 20),
        "model__estimator__max_features": ["sqrt", "log2", 0.3, 0.5],
    }

    n_iter = 5 if fast_mode else 15
    n_cv   = 3 if fast_mode else 5

    rnd_search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=n_cv,
        scoring="r2",          # book uses neg_rmse; we use r2 for clarity
        random_state=42,
        n_jobs=-1,
        error_score="raise",
    )

    # Book note: RandomizedSearchCV scores multi-output as average R² across outputs
    rnd_search.fit(X_train, y_train)

    best_params = rnd_search.best_params_
    best_score  = float(round(rnd_search.best_score_, 4))
    print(f"  Best CV R²: {best_score}")
    print(f"  Best params: {best_params}")

    # Book Step 6: "Analyse the best models — feature importances"
    # Extract from the fitted RandomForest inside the pipeline
    best_pipeline = rnd_search.best_estimator_
    try:
        importances = np.mean(
            [est.feature_importances_
             for est in best_pipeline.named_steps["model"].estimators_],
            axis=0
        )
    except Exception:
        importances = np.zeros(X_train.shape[1])

    return best_pipeline, best_params, importances, best_score


# ── GNN / MLP training (PyTorch) ──────────────────────────────────────────────

def _train_torch_model(
    model,
    graph_data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    model_name: str = "GNN",
    use_edges: bool = True,
) -> tuple:
    """Train GNN or MLP. Returns (test_preds, train_losses, val_losses)."""
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if isinstance(graph_data, dict):
        x          = torch.tensor(graph_data["x"],          dtype=torch.float32).to(device)
        y          = torch.tensor(graph_data["y"],          dtype=torch.float32).to(device)
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long).to(device)
        edge_attr  = torch.tensor(graph_data["edge_weight"],dtype=torch.float32).unsqueeze(1).to(device)
    else:
        x          = graph_data.x.to(device)
        y          = graph_data.y.to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr  = graph_data.edge_attr.to(device) if graph_data.edge_attr is not None else None

    tm  = torch.tensor(train_mask).to(device)
    vm  = torch.tensor(val_mask).to(device)
    tsm = torch.tensor(test_mask).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    train_losses, val_losses = [], []

    print(f"[TRAIN] {model_name}: {n_epochs} epochs on {device}")
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(x, edge_index, edge_attr) if use_edges else model(x)
        loss = criterion(out[tm], y[tm])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out      = model(x, edge_index, edge_attr) if use_edges else model(x)
            val_loss = criterion(out[vm], y[vm]).item()

        scheduler.step(val_loss)
        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[TRAIN] {model_name}: early stop at epoch {epoch}")
            break

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d} | train={loss.item():.4f} | val={val_loss:.4f} | {elapsed:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out   = model(x, edge_index, edge_attr) if use_edges else model(x)
        preds = _to_numpy(out[tsm])

    return preds, train_losses, val_losses


# ── Main entry point ──────────────────────────────────────────────────────────

def train_all(
    expression: np.ndarray,
    metal_targets: np.ndarray,
    coords: np.ndarray,
    gene_names: list,
    graph_data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    n_epochs: int = 100,
    fast_mode: bool = False,
) -> dict:
    """
    Full Steps 5 + 6 pipeline:
      1. Cross-validate sklearn models        (Step 5)
      2. Fine-tune RF with RandomizedSearchCV  (Step 6)
      3. Train GNN + MLP with early stopping  (deep learning path)
      4. Evaluate ALL models on the held-out test set   (Step 6 final eval)
      5. Save results
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if fast_mode:
        n_epochs = min(n_epochs, 50)

    y_test = metal_targets[test_mask]
    n_genes = expression.shape[1]
    all_results = {}
    all_preds   = {}
    loss_curves = {}

    # ── Step 5: Cross-validate ────────────────────────────────────────────────
    cross_validate_models(expression, metal_targets, train_mask, fast_mode)

    # ── Step 6a: Fine-tune RF ─────────────────────────────────────────────────
    best_rf_pipeline, best_params, importances, _ = fine_tune_rf(
        expression, metal_targets, train_mask, fast_mode
    )

    # Evaluate fine-tuned RF on test set
    # Book: "evaluate on the test set to estimate the generalisation error"
    preds_rf = best_rf_pipeline.predict(expression[test_mask])
    all_results["RandomForest"] = _r2_per_metal(y_test, preds_rf)
    all_preds["RandomForest"]   = preds_rf

    # Save feature importances (Step 6: "analyse the best models")
    top20_idx  = np.argsort(importances)[::-1][:20]
    fi_report  = [
        {"gene": gene_names[i], "importance": float(round(importances[i], 6))}
        for i in top20_idx
    ]
    with open(RESULTS_DIR / "feature_importance.json", "w") as f:
        json.dump(fi_report, f, indent=2)
    print(f"\n[TUNE] Top RF gene: {fi_report[0]['gene']} "
          f"(importance={fi_report[0]['importance']:.4f})")
    print(f"[TUNE] Feature importances saved → results/feature_importance.json")
    print(f"[TUNE] Best params saved: {best_params}")

    # ── Step 6b: Ridge (baseline linear) ─────────────────────────────────────
    lin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   MultiOutputRegressor(Ridge(alpha=1.0))),
    ])
    lin_pipeline.fit(expression[train_mask], metal_targets[train_mask])
    preds_lin = lin_pipeline.predict(expression[test_mask])
    all_results["LinearRegression"] = _r2_per_metal(y_test, preds_lin)
    all_preds["LinearRegression"]   = preds_lin
    print("\n[TRAIN] LinearRegression done")

    # ── Step 6c: GNN (deep learning with spatial graph) ───────────────────────
    if HAS_TORCH:
        import torch
        from pipeline.models import SpatialGNN
        gnn = SpatialGNN(in_channels=n_genes, out_channels=5, hidden=256)
        preds_gnn, tl_gnn, vl_gnn = _train_torch_model(
            gnn, graph_data, train_mask, val_mask, test_mask,
            n_epochs=n_epochs, model_name="GNN", use_edges=True,
        )
        all_results["GNN"] = _r2_per_metal(y_test, preds_gnn)
        all_preds["GNN"]   = preds_gnn
        loss_curves["GNN"] = {"train": tl_gnn, "val": vl_gnn}
        torch.save(gnn.state_dict(), RESULTS_DIR / "best_gnn.pt")
        print(f"[TRAIN] GNN checkpoint saved → results/best_gnn.pt")

    # ── Step 6d: MLP (no graph) ───────────────────────────────────────────────
    if HAS_TORCH:
        from pipeline.models import MLPBaseline
        mlp = MLPBaseline(in_channels=n_genes, out_channels=5)
        preds_mlp, tl_mlp, vl_mlp = _train_torch_model(
            mlp, graph_data, train_mask, val_mask, test_mask,
            n_epochs=n_epochs, model_name="MLP", use_edges=False,
        )
        all_results["MLP"] = _r2_per_metal(y_test, preds_mlp)
        all_preds["MLP"]   = preds_mlp
        loss_curves["MLP"] = {"train": tl_mlp, "val": vl_mlp}
        torch.save(mlp.state_dict(), RESULTS_DIR / "best_mlp.pt")
        print(f"[TRAIN] MLP checkpoint saved → results/best_mlp.pt")

    # ── Book Step 6 final: save generalisation error on test set ─────────────
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[TRAIN] Test-set results saved → results/comparison.json")

    # Save predictions for Streamlit
    save_dict = {
        "y_test":      y_test,
        "coords_test": coords[test_mask],
        "coords_all":  coords,
        "gene_names":  np.array(gene_names),
    }
    for name, preds in all_preds.items():
        save_dict[f"pred_{name}"] = preds

    # Full-dataset GNN predictions for tissue map
    if HAS_TORCH and "GNN" in all_preds:
        import torch
        from pipeline.models import SpatialGNN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gnn2 = SpatialGNN(in_channels=n_genes, out_channels=5, hidden=256)
        gnn2.load_state_dict(torch.load(RESULTS_DIR / "best_gnn.pt", map_location=device))
        gnn2 = gnn2.to(device).eval()

        x_all  = (graph_data.x if not isinstance(graph_data, dict)
                  else torch.tensor(graph_data["x"], dtype=torch.float32)).to(device)
        ei_all = (graph_data.edge_index if not isinstance(graph_data, dict)
                  else torch.tensor(graph_data["edge_index"], dtype=torch.long)).to(device)
        with torch.no_grad():
            save_dict["pred_GNN_all"] = gnn2(x_all, ei_all).cpu().numpy()

    np.savez(RESULTS_DIR / "predictions.npz", **save_dict)

    with open(RESULTS_DIR / "loss_curves.json", "w") as f:
        json.dump(loss_curves, f)

    return all_results


# ── Pretty-print ──────────────────────────────────────────────────────────────

def print_comparison(results: dict):
    w = 65
    print("\n" + "=" * w)
    print(f"{'Model':<20} {'Fe':>7} {'Zn':>7} {'Cu':>7} {'Mn':>7} {'Mg':>7} {'Mean':>7}")
    print("-" * w)
    for name, scores in results.items():
        row = f"{name:<20}"
        for m in METALS + ["mean"]:
            row += f" {scores.get(m, 0):>7.4f}"
        print(row)
    print("=" * w)
