"""
Data loading and preprocessing for ALS spatial transcriptomics.

Attempts to download real GEO data (GSE224364), falls back to
biologically grounded synthetic data if unavailable.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Biological anchors ────────────────────────────────────────────────────────
ALS_IRON_GENES = [
    "FTH1", "FTL", "TFRC", "SLC40A1", "HAMP", "BMP6", "SMAD4", "TF", "CP", "STEAP3",
]
ALS_ZINC_GENES = [
    "SOD1", "MT1A", "MT2A", "ZIP8", "ZnT3", "SLC30A3", "PARK7", "ATP7A", "COMMD1",
]
ALS_COPPER_GENES = [
    "SOD1", "ATP7A", "ATP7B", "CP", "ATOX1", "CCS", "COX17", "SCO1", "COX11",
]
ALS_MN_GENES = [
    "SOD2", "SLC30A10", "SLC39A14", "PARK7", "DJ1", "ATP13A2",
]
ALS_MG_GENES = [
    "SLC41A1", "TRPM7", "CNNM2", "MRS2", "SLC41A2", "ACDP2",
]
ALS_PATHWAY_GENES = [
    "TARDBP", "FUS", "C9orf72", "OPTN", "TBK1", "UBQLN2", "VCP", "HNRNPA1",
    "SETX", "ALS2", "DCTN1", "NEFH", "CHAT", "SMN1", "HSPB1", "HSPA5",
    "NEFL", "NEFM", "MAP2", "TUBB3", "GFAP", "AIF1", "MBP", "MOG",
]

ALL_BIOLOGICAL_GENES = list(
    dict.fromkeys(
        ALS_IRON_GENES + ALS_ZINC_GENES + ALS_COPPER_GENES
        + ALS_MN_GENES + ALS_MG_GENES + ALS_PATHWAY_GENES
    )
)

METAL_NAMES = ["Fe", "Zn", "Cu", "Mn", "Mg"]

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Synthetic data generation ─────────────────────────────────────────────────

def _make_tissue_coords(n_spots: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 2-D coordinates that resemble a cortical tissue section."""
    # Simulate an irregular ellipse with noise
    angles = rng.uniform(0, 2 * np.pi, n_spots)
    radii = rng.beta(2, 2, n_spots) * 50
    x = radii * np.cos(angles) * 1.4 + rng.normal(0, 2, n_spots)
    y = radii * np.sin(angles) + rng.normal(0, 2, n_spots)
    return np.stack([x, y], axis=1)


def _spatial_smooth(values: np.ndarray, coords: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Apply Gaussian spatial smoothing to add autocorrelation."""
    from scipy.spatial.distance import cdist
    D = cdist(coords, coords)
    W = np.exp(-D**2 / (2 * sigma**2))
    W /= W.sum(axis=1, keepdims=True)
    return W @ values


def generate_synthetic_data(
    n_spots: int = 2000,
    n_genes: int = 500,
    seed: int = 42,
) -> dict:
    """
    Build a biologically grounded synthetic spatial transcriptomics dataset.

    Returns
    -------
    dict with keys:
        expression   : (n_spots, n_genes) float32 array (log1p-normalised)
        metal_targets: (n_spots, 5) float32 array
        coords       : (n_spots, 2) float32 array
        gene_names   : list of str (length n_genes)
    """
    print("[DATA] Generating biologically grounded synthetic dataset …")
    rng = np.random.default_rng(seed)

    # ── 1. Gene universe ──────────────────────────────────────────────────────
    # Biological anchors first, then random "background" genes
    n_background = max(0, n_genes - len(ALL_BIOLOGICAL_GENES))
    bg_genes = [f"GENE{i:04d}" for i in range(n_background)]
    gene_names = ALL_BIOLOGICAL_GENES[:n_genes] + bg_genes
    gene_names = gene_names[:n_genes]
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    # ── 2. Tissue coordinates ─────────────────────────────────────────────────
    coords = _make_tissue_coords(n_spots, rng).astype(np.float32)

    # ── 3. Base expression (negative-binomial-like counts) ────────────────────
    # Each gene has a mean expression drawn from a log-normal
    gene_means = rng.lognormal(mean=0.5, sigma=1.2, size=n_genes).astype(np.float32)
    raw_counts = rng.negative_binomial(
        n=2, p=0.3, size=(n_spots, n_genes)
    ).astype(np.float32)
    raw_counts = raw_counts * gene_means[None, :]

    # ── 4. Library-size normalise → log1p ─────────────────────────────────────
    lib_sizes = raw_counts.sum(axis=1, keepdims=True).clip(1)
    expression = np.log1p(raw_counts / lib_sizes * 1e4).astype(np.float32)

    # ── 5. Metal targets (spatially autocorrelated, gene-driven) ─────────────
    metal_targets = np.zeros((n_spots, 5), dtype=np.float32)
    gene_groups = [ALS_IRON_GENES, ALS_ZINC_GENES, ALS_COPPER_GENES, ALS_MN_GENES, ALS_MG_GENES]

    for m, genes in enumerate(gene_groups):
        present = [g for g in genes if g in gene_idx]
        if present:
            idxs = [gene_idx[g] for g in present]
            signal = expression[:, idxs].mean(axis=1)
        else:
            signal = rng.standard_normal(n_spots).astype(np.float32)

        # Add spatial autocorrelation
        spatial_noise = rng.standard_normal(n_spots).astype(np.float32)
        spatial_noise = _spatial_smooth(spatial_noise, coords, sigma=8.0).astype(np.float32)

        # Combine: 60% gene signal, 40% spatial structure
        metal_targets[:, m] = 0.6 * signal + 0.4 * spatial_noise

    # Standardize each metal
    for m in range(5):
        v = metal_targets[:, m]
        metal_targets[:, m] = (v - v.mean()) / (v.std() + 1e-8)

    print(f"[DATA] Synthetic dataset: {n_spots} spots × {n_genes} genes, 5 metal targets")
    return {
        "expression": expression,
        "metal_targets": metal_targets,
        "coords": coords,
        "gene_names": gene_names,
        "source": "synthetic",
    }


# ── GEO download ──────────────────────────────────────────────────────────────

# GSE153960: NYGC ALS Consortium — ALS + control post-mortem motor cortex
# RSEM gene count matrix (supplementary file, ~71 MB compressed)
_COUNT_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153960/suppl/"
    "GSE153960_Gene_counts_matrix_RSEM_Prudencio_et_al_2020.txt.gz"
)
_COUNT_MATRIX_FILE = "GSE153960_Gene_counts_matrix_RSEM_Prudencio_et_al_2020.txt.gz"


def _try_geo_download() -> dict | None:
    """
    Download and parse GSE153960 (NYGC ALS Consortium) RSEM count matrix.
    Falls back gracefully to None on any error.
    """
    import requests, gzip

    count_path = DATA_DIR / _COUNT_MATRIX_FILE

    # ── Download if not cached ─────────────────────────────────────────────
    if not count_path.exists():
        print(f"[DATA] Downloading GSE153960 count matrix (~71 MB) …")
        try:
            resp = requests.get(_COUNT_MATRIX_URL, stream=True, timeout=180)
            resp.raise_for_status()
            total      = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(count_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r       {downloaded/total*100:.0f}%  "
                              f"({downloaded>>20} MB / {total>>20} MB)",
                              end="", flush=True)
            print()
            print(f"[DATA] Download complete → {count_path}")
        except Exception as e:
            print(f"[DATA] Download failed: {e}")
            if count_path.exists():
                count_path.unlink()
            return None
    else:
        print(f"[DATA] Using cached count matrix: {count_path}")

    # ── Parse count matrix ─────────────────────────────────────────────────
    # File structure (confirmed from inspection):
    #   col 0 (index_col=0) = row numbers (1, 2, 3 …)
    #   col 1 "EnsemblID"   = Ensembl gene IDs  (ENSG00000000003.14 …)
    #   cols 2-1660          = per-sample integer counts
    try:
        print("[DATA] Parsing count matrix …")
        with gzip.open(count_path, "rt") as fh:
            df = pd.read_csv(fh, sep="\t", index_col=1)   # EnsemblID as row index

        # Drop the leading row-number column (now the first data column)
        df = df.drop(columns=df.columns[0])
        print(f"[DATA] Raw matrix: {df.shape[0]} genes × {df.shape[1]} samples")

        # Strip Ensembl version numbers: ENSG00000000003.14 → ENSG00000000003
        df.index = df.index.str.replace(r"\.\d+$", "", regex=True)

        # Map Ensembl IDs → HGNC gene symbols for our ALS anchor genes
        ENSEMBL_TO_SYMBOL = {
            "ENSG00000167996": "FTH1",    "ENSG00000087086": "FTL",
            "ENSG00000072274": "TFRC",    "ENSG00000091513": "TF",
            "ENSG00000047457": "CP",      "ENSG00000138449": "SLC40A1",
            "ENSG00000166033": "HAMP",    "ENSG00000112175": "BMP6",
            "ENSG00000141736": "SMAD4",   "ENSG00000115341": "STEAP3",
            "ENSG00000142168": "SOD1",    "ENSG00000112096": "SOD2",
            "ENSG00000205362": "MT1A",    "ENSG00000125144": "MT2A",
            "ENSG00000138028": "ZIP8",    "ENSG00000158106": "SLC30A3",
            "ENSG00000090274": "PARK7",   "ENSG00000165240": "ATP7A",
            "ENSG00000123191": "ATP7B",   "ENSG00000169925": "ATOX1",
            "ENSG00000173992": "CCS",     "ENSG00000128714": "COX17",
            "ENSG00000171533": "SLC30A10","ENSG00000104067": "SLC39A14",
            "ENSG00000159363": "ATP13A2", "ENSG00000120948": "TARDBP",
            "ENSG00000089280": "FUS",     "ENSG00000147894": "C9orf72",
            "ENSG00000123240": "OPTN",    "ENSG00000183735": "TBK1",
            "ENSG00000103064": "UBQLN2",  "ENSG00000165280": "VCP",
            "ENSG00000137309": "HNRNPA1", "ENSG00000107290": "SETX",
            "ENSG00000115266": "ALS2",    "ENSG00000204843": "DCTN1",
            "ENSG00000171246": "NEFH",    "ENSG00000131095": "GFAP",
            "ENSG00000197746": "HSPA5",   "ENSG00000106211": "HSPB1",
            "ENSG00000163794": "SLC41A1", "ENSG00000092439": "TRPM7",
            "ENSG00000119397": "CNNM2",   "ENSG00000158077": "NEFL",
            "ENSG00000130203": "APOE",    "ENSG00000114854": "TNNC1",
            "ENSG00000157542": "KCNJ6",   "ENSG00000148053": "NTRK2",
        }
        df.index = [ENSEMBL_TO_SYMBOL.get(g, g) for g in df.index]

        # Transpose: rows = samples, cols = genes; fill any NaN
        df     = df.T.fillna(0)

        # Normalise: raw counts → CPM → log1p
        counts = df.values.astype(np.float32)
        lib    = counts.sum(axis=1, keepdims=True).clip(1)
        vals   = np.log1p(counts / lib * 1e6)

        gene_names = list(df.columns.astype(str))
        n_spots    = vals.shape[0]
        print(f"[DATA] GSE153960 loaded: {n_spots} samples × {len(gene_names)} genes")

    except Exception as e:
        print(f"[DATA] Parse failed: {e} — falling back to synthetic")
        return None

    # ── Spatial grid (bulk RNA-seq has no x/y coords) ────────────────────
    rng  = np.random.default_rng(42)
    side = int(np.ceil(np.sqrt(n_spots)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    gx = gx.ravel()[:n_spots].astype(np.float32)
    gy = gy.ravel()[:n_spots].astype(np.float32)
    gx += rng.uniform(-0.3, 0.3, n_spots).astype(np.float32)
    gy += rng.uniform(-0.3, 0.3, n_spots).astype(np.float32)
    coords = np.stack([gx, gy], axis=1)

    # ── Metal targets: derived from real anchor gene expression ──────────
    gene_idx_map = {g: i for i, g in enumerate(gene_names)}
    gene_groups  = [ALS_IRON_GENES, ALS_ZINC_GENES, ALS_COPPER_GENES,
                    ALS_MN_GENES,   ALS_MG_GENES]
    metal_targets = np.zeros((n_spots, 5), dtype=np.float32)

    for m, genes in enumerate(gene_groups):
        present = [g for g in genes if g in gene_idx_map]
        if present:
            idxs   = [gene_idx_map[g] for g in present]
            signal = vals[:, idxs].mean(axis=1)
            print(f"       {METAL_NAMES[m]}: {len(present)} anchor genes "
                  f"({', '.join(present[:4])}{'…' if len(present) > 4 else ''})")
        else:
            signal = rng.standard_normal(n_spots).astype(np.float32)
            print(f"       {METAL_NAMES[m]}: no anchor genes found — noise fallback")
        metal_targets[:, m] = (signal - signal.mean()) / (signal.std() + 1e-8)

    return {
        "expression":    vals,
        "metal_targets": metal_targets,
        "coords":        coords,
        "gene_names":    gene_names,
        "source":        "GEO:GSE153960 (NYGC ALS Consortium, Prudencio et al. 2020)",
    }


# ── QC + feature selection ────────────────────────────────────────────────────

def _qc_filter(data: dict, min_genes: int = 200, min_spots: int = 10) -> dict:
    expr = data["expression"]
    coords = data["coords"]
    metals = data["metal_targets"]
    gene_names = data["gene_names"]

    # Spots: must express ≥ min_genes genes
    genes_per_spot = (expr > 0).sum(axis=1)
    keep_spots = genes_per_spot >= min_genes
    # Genes: expressed in ≥ min_spots spots
    spots_per_gene = (expr > 0).sum(axis=0)
    keep_genes = spots_per_gene >= min_spots

    expr = expr[keep_spots][:, keep_genes]
    coords = coords[keep_spots]
    metals = metals[keep_spots]
    gene_names = [g for g, k in zip(gene_names, keep_genes) if k]

    print(f"[QC] After filtering: {expr.shape[0]} spots × {expr.shape[1]} genes")
    return {**data, "expression": expr, "coords": coords,
            "metal_targets": metals, "gene_names": gene_names}


def _select_hvg(data: dict, n_top: int = 2000) -> dict:
    """Keep top highly-variable genes by variance."""
    expr = data["expression"]
    gene_names = data["gene_names"]

    variances = expr.var(axis=0)
    n_top = min(n_top, len(gene_names))
    top_idx = np.argsort(variances)[::-1][:n_top]
    top_idx = np.sort(top_idx)

    # Always include biological anchor genes (if present)
    bio_set = set(ALL_BIOLOGICAL_GENES)
    bio_mask = np.array([g in bio_set for g in gene_names])
    bio_idx = np.where(bio_mask)[0]

    combined = np.union1d(top_idx, bio_idx)
    combined = combined[:n_top]  # respect cap

    expr = expr[:, combined]
    gene_names = [gene_names[i] for i in combined]

    print(f"[HVG] Selected {len(gene_names)} highly variable genes")
    return {**data, "expression": expr, "gene_names": gene_names}


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(
    fast_mode: bool = False,
    n_spots: int = 2000,
    n_genes: int = 500,
) -> dict:
    """
    Load (or generate) the ALS spatial transcriptomics dataset.

    Returns a dict with keys:
        expression, metal_targets, coords, gene_names, source
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = DATA_DIR / "processed.npz"

    if cache_path.exists():
        print("[DATA] Loading cached processed data …")
        npz = np.load(cache_path, allow_pickle=True)
        return {
            "expression": npz["expression"],
            "metal_targets": npz["metal_targets"],
            "coords": npz["coords"],
            "gene_names": list(npz["gene_names"]),
            "source": str(npz["source"]),
        }

    if fast_mode:
        n_spots = min(n_spots, 500)

    # Try real data first
    data = _try_geo_download()

    # Fall back to synthetic
    if data is None:
        data = generate_synthetic_data(n_spots=n_spots, n_genes=n_genes)

    # QC + feature selection
    data = _qc_filter(data)
    data = _select_hvg(data, n_top=min(2000, data["expression"].shape[1]))

    # Cache
    np.savez(
        cache_path,
        expression=data["expression"],
        metal_targets=data["metal_targets"],
        coords=data["coords"],
        gene_names=np.array(data["gene_names"]),
        source=np.array(data["source"]),
    )
    print(f"[DATA] Saved processed data → {cache_path}")

    # Save gene list for Streamlit
    gene_meta = pd.DataFrame({"gene": data["gene_names"]})
    gene_meta.to_csv(RESULTS_DIR / "gene_names.csv", index=False)

    return data
