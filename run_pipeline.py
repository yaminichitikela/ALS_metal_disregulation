"""
run_pipeline.py — One-click pipeline: data → graph → train → explain → results

Usage:
    python run_pipeline.py            # full run (~10 min)
    python run_pipeline.py --fast     # fast mode: 500 spots, 50 epochs (~3 min)
"""

import sys
import time
import argparse
from pathlib import Path

# ── Fast-mode flag ────────────────────────────────────────────────────────────
FAST_MODE = "--fast" in sys.argv or "-f" in sys.argv

parser = argparse.ArgumentParser(description="ALS Metal-Gene Biomarker Pipeline")
parser.add_argument("--fast", action="store_true", help="Fast demo mode (500 spots, 50 epochs)")
parser.add_argument("--no-cache", action="store_true", help="Ignore cached processed data")
args = parser.parse_args()
FAST_MODE = args.fast

N_SPOTS = 500 if FAST_MODE else 2000
N_EPOCHS = 50 if FAST_MODE else 100

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Clear cache if requested
if args.no_cache:
    cache = DATA_DIR / "processed.npz"
    if cache.exists():
        cache.unlink()
        print("[PIPELINE] Cache cleared")

print("=" * 60)
print("  ALS Metal-Gene Biomarker Discovery Pipeline")
print("  ODU CSGS Hackathon 2026")
print(f"  Mode: {'FAST' if FAST_MODE else 'FULL'} | spots={N_SPOTS} | epochs={N_EPOCHS}")
print("=" * 60)

t_start = time.time()

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("\n[STEP 1/5] Loading / generating data …")
from pipeline.data_loader import load_data

data = load_data(fast_mode=FAST_MODE, n_spots=N_SPOTS, n_genes=500)
expression = data["expression"]
metal_targets = data["metal_targets"]
coords = data["coords"]
gene_names = data["gene_names"]
source = data["source"]

print(f"  Source : {source}")
print(f"  Shape  : {expression.shape[0]:,} spots × {expression.shape[1]:,} genes")
print(f"  Metals : {metal_targets.shape[1]} ({', '.join(['Fe','Zn','Cu','Mn','Mg'])})")

# ── Step 3: Explore the data (Hands-On ML Ch.2 Step 3) ───────────────────────
print("\n[STEP 2/5] Exploring data (EDA) …")
from pipeline.explore import explore_data
eda_report = explore_data(expression, metal_targets, coords, gene_names)
print(f"  Metal–Fe top gene: "
      f"{eda_report['top_genes_fe_correlation']['top_positive'][0]['gene']}")

# ── Step 2: Build graph ───────────────────────────────────────────────────────
print("\n[STEP 3/5] Building spatial tissue graph …")
from pipeline.graph_builder import build_graph, train_val_test_split

graph_data = build_graph(expression, metal_targets, coords, k=6)

train_mask, val_mask, test_mask = train_val_test_split(
    coords, n_spots=expression.shape[0]
)
print(f"  Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

# ── Step 3: Train models ──────────────────────────────────────────────────────
print("\n[STEP 4/5] Training models (CV → fine-tune → final eval) …")
from pipeline.train import train_all, print_comparison

results = train_all(
    expression=expression,
    metal_targets=metal_targets,
    coords=coords,
    gene_names=gene_names,
    graph_data=graph_data,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    n_epochs=N_EPOCHS,
    fast_mode=FAST_MODE,
)

# ── Step 4: SHAP analysis ─────────────────────────────────────────────────────
print("\n[STEP 5/5] Gene importance analysis (iron / Fe) …")
from pipeline.explain import run_shap_analysis

shap_results = run_shap_analysis(
    expression=expression,
    metal_targets=metal_targets,
    gene_names=gene_names,
)
print(f"  Top gene: {shap_results[0]['gene']} (SHAP={shap_results[0]['shap']:.4f})")

# ── Step 5: Print summary ─────────────────────────────────────────────────────
print("\n[RESULTS] Final test-set comparison")
print_comparison(results)

elapsed = time.time() - t_start
print(f"\n  Total time: {elapsed/60:.1f} min")
print(f"  Data source: {source}")
print()
print("=" * 60)
print("  Done!")
print("  Run:  streamlit run app.py")
print("=" * 60)
