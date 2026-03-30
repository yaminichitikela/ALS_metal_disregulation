"""
Step 3 from Hands-On ML (Géron): Explore the Data to Gain Insights.

The book says:
  - Create a copy of the data for exploration (don't touch training set)
  - Study each attribute: type, missing values, distribution, correlations
  - Compute a correlation matrix between features and targets
  - Visualize distributions
  - Document key findings

Saves plots + a JSON report to results/eda/
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
EDA_DIR = RESULTS_DIR / "eda"
METALS = ["Fe", "Zn", "Cu", "Mn", "Mg"]


def explore_data(
    expression: np.ndarray,
    metal_targets: np.ndarray,
    coords: np.ndarray,
    gene_names: list,
) -> dict:
    """
    Run the full EDA pipeline from the book. Returns a summary dict
    and saves plots to results/eda/.

    Book quote: "Create a copy of the data for exploration
    (never touch the training set for exploration)."
    """
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Work on a copy (book step 3 rule #1) ─────────────────────────────────
    expr    = expression.copy()
    targets = metal_targets.copy()
    xy      = coords.copy()

    n_spots, n_genes = expr.shape
    report = {}

    print("\n[EDA] ── Step 3: Explore the Data ──────────────────────────────")

    # ── 1. Attribute overview (book: study each attribute) ────────────────────
    print(f"[EDA] Dataset: {n_spots} spots × {n_genes} genes, {targets.shape[1]} metal targets")

    attr_summary = []
    for i, m in enumerate(METALS):
        v = targets[:, i]
        attr_summary.append({
            "metal":   m,
            "mean":    float(round(v.mean(), 4)),
            "std":     float(round(v.std(), 4)),
            "min":     float(round(v.min(), 4)),
            "max":     float(round(v.max(), 4)),
            "missing": int(np.isnan(v).sum()),
        })
        print(f"  {m}: mean={v.mean():.3f}  std={v.std():.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}  missing={np.isnan(v).sum()}")

    report["attribute_summary"] = attr_summary

    # ── 2. Distribution histograms (book: generate visualisations) ────────────
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    fig.suptitle("Metal Target Distributions (training data copy)", fontsize=12)
    for i, (ax, m) in enumerate(zip(axes, METALS)):
        ax.hist(targets[:, i], bins=40, color="#1D9E75", edgecolor="white", linewidth=0.3)
        ax.set_title(m)
        ax.set_xlabel("Normalised abundance")
        ax.set_ylabel("Spot count")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "metal_distributions.png", dpi=120)
    plt.close(fig)
    print("[EDA] Saved: results/eda/metal_distributions.png")

    # ── 3. Spatial distribution (tissue map for each metal) ───────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    fig.suptitle("Spatial Distribution of Metal Targets", fontsize=12)
    for i, (ax, m) in enumerate(zip(axes, METALS)):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=targets[:, i],
                        cmap="RdYlGn_r", s=4, alpha=0.7)
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(m)
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "spatial_metal_maps.png", dpi=120)
    plt.close(fig)
    print("[EDA] Saved: results/eda/spatial_metal_maps.png")

    # ── 4. Correlation matrix (book: "look for correlations") ────────────────
    # Between metals (are they co-regulated?)
    metal_df = pd.DataFrame(targets, columns=METALS)
    metal_corr = metal_df.corr()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(metal_corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(5)); ax.set_xticklabels(METALS)
    ax.set_yticks(range(5)); ax.set_yticklabels(METALS)
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f"{metal_corr.values[r,c]:.2f}",
                    ha="center", va="center", fontsize=9)
    ax.set_title("Metal–Metal Correlation Matrix")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "metal_correlation_matrix.png", dpi=120)
    plt.close(fig)
    print("[EDA] Saved: results/eda/metal_correlation_matrix.png")

    report["metal_correlations"] = metal_corr.round(4).to_dict()

    # ── 5. Gene–metal correlations (book: identify valuable features) ─────────
    # Book: "Look for correlations between each feature and the label"
    print("[EDA] Computing gene–metal Pearson correlations …")
    gene_fe_corr = np.array([
        float(np.corrcoef(expr[:, j], targets[:, 0])[0, 1])
        for j in range(min(n_genes, 500))   # cap at 500 for speed
    ])
    top10_pos = np.argsort(gene_fe_corr)[::-1][:10]
    top10_neg = np.argsort(gene_fe_corr)[:10]

    top_genes_fe = {
        "top_positive": [
            {"gene": gene_names[j], "correlation": round(float(gene_fe_corr[j]), 4)}
            for j in top10_pos
        ],
        "top_negative": [
            {"gene": gene_names[j], "correlation": round(float(gene_fe_corr[j]), 4)}
            for j in top10_neg
        ],
    }
    report["top_genes_fe_correlation"] = top_genes_fe

    print("  Top genes positively correlated with Fe:")
    for g in top_genes_fe["top_positive"][:5]:
        print(f"    {g['gene']:<12}  r={g['correlation']:+.4f}")
    print("  Top genes negatively correlated with Fe:")
    for g in top_genes_fe["top_negative"][:5]:
        print(f"    {g['gene']:<12}  r={g['correlation']:+.4f}")

    # Bar chart of top correlated genes
    names_pos = [g["gene"] for g in top_genes_fe["top_positive"]]
    vals_pos  = [g["correlation"] for g in top_genes_fe["top_positive"]]
    names_neg = [g["gene"] for g in top_genes_fe["top_negative"]]
    vals_neg  = [g["correlation"] for g in top_genes_fe["top_negative"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(names_pos[::-1], vals_pos[::-1], color="#1D9E75")
    axes[0].set_title("Top 10 genes: positive correlation with Fe")
    axes[0].set_xlabel("Pearson r")
    axes[1].barh(names_neg, vals_neg, color="#e05c5c")
    axes[1].set_title("Top 10 genes: negative correlation with Fe")
    axes[1].set_xlabel("Pearson r")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "gene_fe_correlations.png", dpi=120)
    plt.close(fig)
    print("[EDA] Saved: results/eda/gene_fe_correlations.png")

    # ── 6. Gene expression statistics (book: study each attribute) ────────────
    expr_means  = expr.mean(axis=0)
    expr_vars   = expr.var(axis=0)
    zeros_frac  = (expr == 0).mean(axis=0)   # sparsity per gene

    report["expression_stats"] = {
        "mean_expression_mean": float(round(expr_means.mean(), 4)),
        "mean_expression_std":  float(round(expr_means.std(), 4)),
        "sparsity_mean":        float(round(zeros_frac.mean(), 4)),
        "n_genes_gt50pct_zero": int((zeros_frac > 0.5).sum()),
    }

    # Variance vs mean plot (mean-variance relationship — characteristic of RNA-seq)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(expr_means[:500], expr_vars[:500], s=4, alpha=0.5, color="#1D9E75")
    ax.set_xlabel("Mean expression (log1p-normalised)")
    ax.set_ylabel("Variance")
    ax.set_title("Mean–Variance Relationship (gene expression)")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "mean_variance.png", dpi=120)
    plt.close(fig)
    print("[EDA] Saved: results/eda/mean_variance.png")

    # ── 7. Missing value report (book: "handle missing features") ────────────
    n_missing_expr    = int(np.isnan(expr).sum())
    n_missing_targets = int(np.isnan(targets).sum())
    report["missing_values"] = {
        "expression_matrix": n_missing_expr,
        "metal_targets":     n_missing_targets,
    }
    print(f"[EDA] Missing values — expression: {n_missing_expr}, targets: {n_missing_targets}")

    # ── 8. Save summary report ────────────────────────────────────────────────
    with open(RESULTS_DIR / "eda_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[EDA] Full report saved → results/eda_report.json")
    print("[EDA] ─────────────────────────────────────────────────────────────\n")

    return report
