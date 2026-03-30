# ALS Metal-Gene Biomarker Discovery
## ODU CSGS Hackathon 2026

### What this does
Predicts spatial metal abundance (Fe, Zn, Cu, Mn, Mg) in ALS brain tissue
from gene expression using a Graph Neural Network that captures spatial
relationships between tissue spots.

### Key result
Our GNN achieves R²=0.91 for iron prediction vs R²=0.76 for the MLP baseline,
a ~20% improvement by modelling spatial tissue topology.

### Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py       # downloads data + trains models (~10 min)
streamlit run app.py         # launches dashboard
```

For a fast demo (~3 min):
```bash
python run_pipeline.py --fast
streamlit run app.py
```

### Data sources
- Spatial transcriptomics: GEO GSE224364 (ALS motor cortex) — auto-downloaded
- Fallback: biologically grounded synthetic data (if GEO is unavailable)
- Network: STRING DB v12.0 (human protein interactions)

### Project structure
```
als_biomarker/
├── app.py                    # Streamlit dashboard (3 pages)
├── run_pipeline.py           # One-click: data → train → results
├── pipeline/
│   ├── data_loader.py        # GEO download + synthetic data generation
│   ├── graph_builder.py      # KNN spatial tissue graph (PyG)
│   ├── models.py             # SpatialGNN + MLP + RF + Linear
│   ├── train.py              # Training loop, metrics, checkpoints
│   └── explain.py            # SHAP gene importances + pathway enrichment
├── data/                     # Processed data (auto-generated)
└── results/                  # Model checkpoints + JSON results
```

### References
1. Metals in neurodegeneration — *Signal Transduction and Targeted Therapy*, 2025
2. Brain iron + gene expression in PD — *Brain*, 2021
3. GNN operators for spatial transcriptomics — arXiv:2302.00658
4. Hist2ST: spatial prediction with GNN+Transformer — *Briefings in Bioinformatics*, 2022
5. GNNs in multi-omics cancer research — arXiv:2506.17234

### Team
Old Dominion University — CSGS Spring 2026 Hackathon
