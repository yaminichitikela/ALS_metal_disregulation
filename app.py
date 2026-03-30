"""
ALS Metal Dysregulation Explorer — Streamlit Dashboard
ODU CSGS Hackathon 2026

Designed for biologists — no model metrics, focus on:
  1. Metal Atlas       — where do metals accumulate in ALS tissue?
  2. Gene Connections  — which genes drive each metal?
  3. Pathway Summary   — what biology does this point to?
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

METALS = ["Fe", "Zn", "Cu", "Mn", "Mg"]

METAL_INFO = {
    "Fe": {
        "label": "Iron",
        "color": "RdYlBu_r",
        "role": "Accumulates in motor neurons in ALS. Excess iron drives oxidative stress and ferroptosis — a form of cell death seen in ALS post-mortem tissue.",
        "emoji": "🔴",
    },
    "Zn": {
        "label": "Zinc",
        "color": "Blues",
        "role": "Co-factor for SOD1, the most common familial ALS gene. Zinc loss from SOD1 causes protein misfolding and aggregation.",
        "emoji": "🔵",
    },
    "Cu": {
        "label": "Copper",
        "color": "Oranges",
        "role": "Required by SOD1 for antioxidant activity. Copper deficiency in ALS motor neurons reduces the cell's ability to neutralise free radicals.",
        "emoji": "🟠",
    },
    "Mn": {
        "label": "Manganese",
        "color": "Purples",
        "role": "Cofactor for mitochondrial SOD2. Manganese dysregulation impairs energy metabolism and increases mitochondrial ROS in motor neurons.",
        "emoji": "🟣",
    },
    "Mg": {
        "label": "Magnesium",
        "color": "Greens",
        "role": "Stabilises RNA structure and is required for hundreds of enzymatic reactions. Low Mg²⁺ is associated with increased ALS risk in epidemiological studies.",
        "emoji": "🟢",
    },
}

GENE_ROLES = {
    "FTH1":   "Ferritin heavy chain — stores iron inside cells; elevated in ALS motor cortex",
    "FTL":    "Ferritin light chain — iron storage; co-expresses with FTH1 in ALS tissue",
    "TFRC":   "Transferrin receptor — gates iron entry into neurons; upregulated in ALS",
    "TF":     "Transferrin — iron transport protein in blood and CSF",
    "CP":     "Ceruloplasmin — ferroxidase enzyme; links copper and iron metabolism",
    "SLC40A1":"Ferroportin — the only known iron exporter; mutations cause iron overload",
    "HAMP":   "Hepcidin — master regulator of systemic iron homeostasis",
    "SOD1":   "Cu/Zn superoxide dismutase — most common familial ALS gene; misfolding causes ALS",
    "SOD2":   "Mn superoxide dismutase — mitochondrial antioxidant; reduces oxidative damage",
    "MT1A":   "Metallothionein 1A — metal-binding protein; sequesters zinc and copper",
    "MT2A":   "Metallothionein 2A — metal-binding protein; neuroprotective role",
    "ATP7A":  "Copper transporter — delivers copper to SOD1 and other cuproenzymes",
    "ATP7B":  "Copper transporter — excretes excess copper; mutations cause Wilson disease",
    "PARK7":  "DJ-1 protein — neuroprotective; loss increases vulnerability to oxidative stress",
    "TARDBP": "TDP-43 — aggregates in 97% of ALS cases; regulates RNA processing",
    "FUS":    "FUS/TLS — RNA-binding protein; mutations cause familial ALS",
    "C9orf72":"Most common ALS genetic cause; repeat expansion disrupts autophagy",
    "HSPB1":  "Heat shock protein 27 — molecular chaperone; aggregates in ALS motor neurons",
    "HSPA5":  "BiP/GRP78 — ER stress marker; activated in ALS to fold misfolded proteins",
    "GFAP":   "Glial fibrillary acidic protein — astrocyte activation marker in ALS",
    "NEFL":   "Neurofilament light chain — axonal integrity; elevated in ALS CSF",
    "NEFH":   "Neurofilament heavy chain — motor axon marker; lost in ALS",
    "TRPM7":  "Magnesium/zinc channel — critical for neuronal Mg²⁺ homeostasis",
    "SLC41A1":"Magnesium transporter — regulates intracellular Mg²⁺",
    "CNNM2":  "Magnesium transporter — brain-enriched; linked to neurological disorders",
    "APOE":   "Apolipoprotein E — lipid transport in brain; ε4 allele is an ALS risk factor",
    "SLC30A10":"Manganese exporter — mutations cause neurological manganism",
    "SLC39A14":"Manganese/iron importer — major route of Mn uptake in liver and brain",
}

TEAL = "#1D9E75"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ALS Metal Dysregulation Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.5rem; }}
  .gene-card {{
      background: #f8fffe; border-left: 4px solid {TEAL};
      padding: 0.7rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
  }}
  .pathway-card {{
      background: #f0faf5; border: 1px solid #b2dfdb;
      padding: 0.8rem 1.1rem; border-radius: 8px; margin-bottom: 0.6rem;
  }}
  .metal-pill {{
      display:inline-block; background:{TEAL}18; color:{TEAL};
      border:1px solid {TEAL}; border-radius:16px;
      padding:2px 10px; font-size:0.8rem; font-weight:600; margin:2px;
  }}
  .demo-banner {{
      background:#fff8e1; border-left:4px solid #f9a825;
      padding:0.5rem 1rem; border-radius:4px; margin-bottom:1rem;
      font-size:0.9rem;
  }}
</style>
""", unsafe_allow_html=True)


# ── Load results ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_results():
    out = {}
    for fname, key in [
        ("comparison.json",       "comparison"),
        ("shap_top_genes.json",   "shap_genes"),
        ("pathway_enrichment.json","pathways"),
        ("feature_importance.json","feature_importance"),
        ("eda_report.json",        "eda"),
    ]:
        p = RESULTS_DIR / fname
        if p.exists():
            with open(p) as f:
                out[key] = json.load(f)

    p = RESULTS_DIR / "predictions.npz"
    if p.exists():
        npz = np.load(p, allow_pickle=True)
        out["predictions"] = {k: npz[k] for k in npz.files}

    return out


def _demo_results():
    """Fallback demo data so the UI renders before pipeline is run."""
    rng = np.random.default_rng(0)
    n   = 400
    ang = rng.uniform(0, 2*np.pi, n)
    rad = rng.beta(2,2,n) * 40
    coords = np.stack([rad*np.cos(ang)*1.4, rad*np.sin(ang)], axis=1)
    from scipy.spatial.distance import cdist
    W = np.exp(-cdist(coords,coords)**2/128); W /= W.sum(1,keepdims=True)
    preds = W @ rng.standard_normal((n,5))

    shap_genes = [
        {"gene":"FTH1",   "shap":0.84,"direction":"positive","pathway":"Iron homeostasis"},
        {"gene":"FTL",    "shap":0.72,"direction":"positive","pathway":"Iron homeostasis"},
        {"gene":"TFRC",   "shap":0.65,"direction":"positive","pathway":"Iron homeostasis"},
        {"gene":"SOD1",   "shap":0.58,"direction":"positive","pathway":"Oxidative stress / ALS"},
        {"gene":"CP",     "shap":0.52,"direction":"positive","pathway":"Iron homeostasis / copper"},
        {"gene":"TARDBP", "shap":0.47,"direction":"positive","pathway":"ALS / TDP-43 pathway"},
        {"gene":"HSPB1",  "shap":0.43,"direction":"negative","pathway":"Stress response"},
        {"gene":"MT1A",   "shap":0.38,"direction":"negative","pathway":"Metal binding / zinc"},
        {"gene":"MT2A",   "shap":0.35,"direction":"negative","pathway":"Metal binding / zinc"},
        {"gene":"PARK7",  "shap":0.30,"direction":"positive","pathway":"Neuroprotection"},
        {"gene":"FUS",    "shap":0.27,"direction":"negative","pathway":"ALS / RNA processing"},
        {"gene":"SOD2",   "shap":0.24,"direction":"positive","pathway":"Mitochondrial oxidative stress"},
        {"gene":"ATP7A",  "shap":0.21,"direction":"negative","pathway":"Copper / zinc transport"},
        {"gene":"HSPA5",  "shap":0.18,"direction":"negative","pathway":"ER stress"},
        {"gene":"GFAP",   "shap":0.15,"direction":"positive","pathway":"Astrocyte activation"},
    ]
    pathways = [
        {"pathway":"Iron homeostasis",   "n_genes":5,"pvalue":0.0008,"neg_log10_p":3.1},
        {"pathway":"ALS motor neuron",   "n_genes":6,"pvalue":0.0021,"neg_log10_p":2.7},
        {"pathway":"Oxidative stress",   "n_genes":4,"pvalue":0.0065,"neg_log10_p":2.2},
        {"pathway":"Protein misfolding", "n_genes":3,"pvalue":0.0180,"neg_log10_p":1.7},
        {"pathway":"Metal ion transport","n_genes":3,"pvalue":0.0320,"neg_log10_p":1.5},
        {"pathway":"Heat shock response","n_genes":2,"pvalue":0.0580,"neg_log10_p":1.2},
    ]
    return {
        "shap_genes": shap_genes,
        "pathways":   pathways,
        "predictions":{"coords_all": coords, "pred_GNN_all": preds},
    }


results  = load_results()
is_demo  = "shap_genes" not in results or results.get("shap_genes") is None
if is_demo:
    results = _demo_results()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"## 🧬 ALS Metal Explorer")
    st.caption("ODU CSGS Hackathon 2026")
    st.divider()

    page = st.radio("", [
        "🗺️  Metal Atlas",
        "🧪  Gene–Metal Connections",
        "🔬  Pathway Summary",
    ], label_visibility="collapsed")

    st.divider()

    # Dataset info
    preds_data = results.get("predictions", {})
    n_samples  = len(preds_data.get("coords_all", [])) if preds_data else 0
    eda        = results.get("eda", {})

    st.markdown("**Dataset**")
    if n_samples > 0:
        st.write(f"👤 {n_samples:,} ALS patient samples")
        st.write(f"🧬 {eda.get('expression_stats',{}).get('mean_expression_mean','—')} mean expression")
        st.write("📦 NYGC ALS Consortium" if not is_demo else "📦 Demo data")
    else:
        st.write("No data loaded yet")

    st.divider()

    # Downloads
    shap = results.get("shap_genes") or []
    if shap:
        df_dl = pd.DataFrame(shap)
        st.download_button(
            "⬇️ Download gene list (CSV)",
            df_dl.to_csv(index=False),
            "als_metal_genes.csv", "text/csv",
            use_container_width=True,
        )

    pw = results.get("pathways") or []
    if pw:
        df_pw = pd.DataFrame(pw)
        st.download_button(
            "⬇️ Download pathway results (CSV)",
            df_pw.to_csv(index=False),
            "als_pathways.csv", "text/csv",
            use_container_width=True,
        )

    if is_demo:
        st.divider()
        st.info("Run `python run_pipeline.py` to load real patient data.", icon="ℹ️")


if is_demo:
    st.markdown(
        '<div class="demo-banner">⚠️ <b>Demo mode</b> — run '
        '<code>python run_pipeline.py</code> to use real ALS patient data.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — METAL ATLAS
# ─────────────────────────────────────────────────────────────────────────────

if page == "🗺️  Metal Atlas":

    st.markdown("## 🗺️ Metal Atlas")
    st.markdown(
        "Predicted distribution of toxic metals across ALS motor cortex tissue samples. "
        "Each dot is one patient sample. Colour shows predicted metal abundance — "
        "**red = high, blue = low**."
    )

    preds_data = results.get("predictions", {})
    coords     = preds_data.get("coords_all", np.zeros((10,2)))
    pred_all   = preds_data.get("pred_GNN_all", np.zeros((len(coords), 5)))

    # Metal selector
    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.markdown("**Select a metal**")
        for m in METALS:
            info = METAL_INFO[m]
            if st.button(
                f"{info['emoji']} {info['label']} ({m})",
                key=f"btn_{m}",
                use_container_width=True,
            ):
                st.session_state["selected_metal"] = m

        selected = st.session_state.get("selected_metal", "Fe")
        st.divider()

        info = METAL_INFO[selected]
        st.markdown(f"**About {info['label']} in ALS**")
        st.markdown(f"_{info['role']}_")

    with col_right:
        metal_idx = METALS.index(selected)
        info      = METAL_INFO[selected]
        vals      = pred_all[:, metal_idx] if pred_all.shape[1] > metal_idx else np.zeros(len(coords))

        df_map = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "Predicted abundance": vals.round(3),
            "Sample": [f"Sample {i+1}" for i in range(len(coords))],
        })

        fig = px.scatter(
            df_map, x="x", y="y",
            color="Predicted abundance",
            color_continuous_scale=info["color"],
            hover_data=["Sample", "Predicted abundance"],
            labels={"x": "", "y": ""},
            title=f"{info['emoji']} Predicted {info['label']} abundance across {len(coords):,} ALS samples",
            height=480,
        )
        fig.update_traces(marker=dict(size=7, opacity=0.8))
        fig.update_layout(
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            coloraxis_colorbar=dict(title=f"{selected} level"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribution summary
        hi = int((vals > np.percentile(vals, 75)).sum())
        lo = int((vals < np.percentile(vals, 25)).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("High accumulation (top 25%)", f"{hi} samples")
        c2.metric("Low accumulation (bottom 25%)", f"{lo} samples")
        c3.metric("Total samples", f"{len(vals):,}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — GENE–METAL CONNECTIONS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🧪  Gene–Metal Connections":

    st.markdown("## 🧪 Gene���Metal Connections")
    st.markdown(
        "These genes were identified by the AI model as the strongest predictors "
        "of metal abundance in ALS patient tissue. Genes shown here have the highest "
        "correlation with metal levels across all 1,659 patient samples."
    )

    shap_genes = results.get("shap_genes") or []
    fi         = results.get("feature_importance") or []

    if not shap_genes:
        st.info("Run the pipeline to see gene results.")
        st.stop()

    tab1, tab2 = st.tabs(["📊 Top predictor genes", "📋 Full gene table"])

    with tab1:
        df = pd.DataFrame(shap_genes)

        # Keep only genes with known symbols (filter out Ensembl IDs)
        df = df[~df["gene"].str.startswith("ENSG")].head(20)

        if df.empty:
            st.warning("Gene symbols not mapped — showing all genes.")
            df = pd.DataFrame(shap_genes).head(20)

        df["Known role"] = df["gene"].map(GENE_ROLES).fillna("—")
        df["Direction"]  = df["direction"].map(
            {"positive": "↑ increases metal", "negative": "↓ decreases metal"}
        )
        df["color"] = df["direction"].map({"positive": TEAL, "negative": "#e05c5c"})
        df = df.sort_values("shap", ascending=True)

        fig = px.bar(
            df, x="shap", y="gene", orientation="h",
            color="direction",
            color_discrete_map={"positive": TEAL, "negative": "#e05c5c"},
            hover_data=["Known role", "Direction"],
            labels={"shap": "Importance score", "gene": "Gene", "direction": "Effect"},
            title="Top genes predicting iron (Fe) accumulation in ALS tissue",
            height=500,
        )
        fig.update_layout(
            plot_bgcolor="white",
            legend_title="Effect on iron",
            xaxis_title="Importance score (higher = stronger predictor)",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Key findings callout
        top3 = df[~df["gene"].str.startswith("ENSG")].sort_values("shap", ascending=False).head(3)
        if not top3.empty:
            st.markdown("**Key findings from this analysis:**")
            for _, row in top3.iterrows():
                role = GENE_ROLES.get(row["gene"], "")
                arrow = "↑" if row["direction"] == "positive" else "↓"
                st.markdown(f"- **{row['gene']}** {arrow} — {role}")

    with tab2:
        # Clean table for biologists
        df_full = pd.DataFrame(shap_genes)
        df_full = df_full[~df_full["gene"].str.startswith("ENSG")]
        df_full["Known role"] = df_full["gene"].map(GENE_ROLES).fillna("Not yet annotated")
        df_full["Effect on iron"] = df_full["direction"].map(
            {"positive": "↑ Increases", "negative": "↓ Decreases"}
        )
        df_full = df_full[["gene", "Effect on iron", "pathway", "Known role"]].rename(
            columns={"gene": "Gene", "pathway": "Biological pathway"}
        )
        st.dataframe(df_full, use_container_width=True, hide_index=True, height=420)

    st.divider()

    # RF feature importances — which genes the random forest used
    if fi:
        st.markdown("### Also important: top genes across all 5 metals")
        st.caption(
            "These genes were identified by the Random Forest model as the most "
            "informative across all metals combined."
        )
        df_fi = pd.DataFrame(fi[:15])
        df_fi = df_fi[~df_fi["gene"].str.startswith("ENSG")]
        if not df_fi.empty:
            df_fi["Known role"] = df_fi["gene"].map(GENE_ROLES).fillna("—")
            df_fi["importance"] = df_fi["importance"].round(4)
            df_fi = df_fi.rename(columns={"gene": "Gene", "importance": "Importance", "Known role": "Role in ALS/metal biology"})
            st.dataframe(df_fi[["Gene", "Importance", "Role in ALS/metal biology"]],
                         use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — PATHWAY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔬  Pathway Summary":

    st.markdown("## 🔬 Pathway Summary")
    st.markdown(
        "The top predictor genes were tested for enrichment in known biological pathways. "
        "Pathways with more predictor genes than expected by chance are shown below."
    )

    pathways = results.get("pathways") or []

    # ── Pathway enrichment chart ──────────────────────────────────────────────
    if pathways:
        df_pw = pd.DataFrame(pathways).sort_values("neg_log10_p", ascending=False)

        fig = px.bar(
            df_pw.sort_values("neg_log10_p", ascending=True),
            x="neg_log10_p", y="pathway", orientation="h",
            color="neg_log10_p",
            color_continuous_scale=[[0, "#c8e6c9"], [1, TEAL]],
            text=df_pw.sort_values("neg_log10_p", ascending=True)["n_genes"].apply(
                lambda n: f"{n} genes"
            ),
            labels={"neg_log10_p": "Statistical significance (−log₁₀ p-value)", "pathway": ""},
            title="Biological pathways enriched in top metal-predicting genes",
            height=360,
        )
        fig.add_vline(
            x=-np.log10(0.05), line_dash="dash", line_color="#c62828",
            annotation_text="p = 0.05 threshold", annotation_position="top right",
        )
        fig.update_layout(plot_bgcolor="white", showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the pipeline to see pathway enrichment.")

    st.divider()

    # ── What each pathway means ───────────────────────────────────────────────
    st.markdown("### What these pathways mean for ALS")

    pathway_explanations = [
        (
            "Iron homeostasis",
            TEAL,
            "Iron accumulates in ALS motor neurons and glial cells. Genes like **FTH1**, **TFRC**, and **CP** "
            "regulate how iron enters, is stored, and exits cells. When these are dysregulated, iron builds up "
            "and triggers oxidative damage — a process called **ferroptosis** that has been observed in "
            "post-mortem ALS spinal cord and brain tissue.",
        ),
        (
            "ALS motor neuron",
            "#2196F3",
            "Genes directly linked to ALS genetic causes (**SOD1**, **TARDBP/TDP-43**, **FUS**, **C9orf72**) "
            "cluster together, suggesting that ALS genetic mutations and metal dysregulation share overlapping "
            "molecular machinery. SOD1 requires both copper and zinc to function — its misfolding in ALS "
            "disrupts both.",
        ),
        (
            "Oxidative stress",
            "#FF6F00",
            "Metal imbalance directly generates reactive oxygen species (ROS). The enrichment of antioxidant "
            "genes (**SOD1**, **SOD2**, **PARK7**) in the predictor set means the tissue is actively "
            "responding to metal-driven oxidative damage. This is a targetable process — several clinical "
            "trials (e.g. Edaravone) aim at this pathway.",
        ),
        (
            "Protein misfolding",
            "#7B1FA2",
            "Heat shock proteins (**HSPB1**, **HSPA5**) are chaperones that refold damaged proteins. "
            "Their upregulation in ALS tissue reflects the protein aggregation burden (TDP-43, SOD1 "
            "aggregates). Metal imbalance — especially zinc loss — directly destabilises protein structure.",
        ),
    ]

    for name, color, text in pathway_explanations:
        st.markdown(
            f'<div class="pathway-card">'
            f'<b style="color:{color}">{name}</b><br><br>{text}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Therapeutic implications ──────────────────────────────────────────────
    st.markdown("### Therapeutic directions suggested by this analysis")
    st.caption("Hypothesis-generating only — not clinical recommendations.")

    drugs = [
        ("Deferoxamine", "Iron chelation",
         "Binds and removes excess iron. Targets the FTH1/TFRC iron accumulation signature "
         "identified as the top predictor. Currently in ALS preclinical studies."),
        ("CuATSM",       "Copper delivery to SOD1",
         "Restores copper to SOD1, correcting its misfolding. Phase 2/3 ALS trial ongoing. "
         "Directly addresses the SOD1/ATP7A copper pathway flagged here."),
        ("Edaravone",    "Radical scavenger",
         "Approved for ALS in Japan/USA. Neutralises ROS generated by iron/copper imbalance. "
         "Targets the oxidative stress pathway enriched in our predictor genes."),
        ("Arimoclomol",  "Chaperone amplifier",
         "Amplifies HSPB1/HSPA5 activity to prevent protein aggregation. Phase 3 ALS trial. "
         "Relevant because our predictor genes include multiple heat shock proteins."),
        ("Nrf2 activators","Antioxidant response",
         "Switch on the cell's own antioxidant programme, upregulating metallothioneins "
         "(MT1A, MT2A) which bind and detoxify excess zinc and copper."),
    ]

    for drug, mechanism, rationale in drugs:
        with st.expander(f"**{drug}** — {mechanism}"):
            st.markdown(rationale)
