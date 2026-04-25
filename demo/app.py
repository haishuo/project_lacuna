"""Lacuna live demo — Streamlit UI for mechanism classification.

Drag a CSV in, hit Analyze, get a green/yellow/red verdict alongside the
model's intermediate work. Pure-Python helpers live in `demo/pipeline.py`
so this file can stay focused on UI.

Run:
    streamlit run demo/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo.pipeline import (
    ACTION_LABELS,
    CHECKPOINT_ENV,
    CLASS_NAMES,
    MODEL_DEFAULTS,
    build_model,
    cross_column_corr,
    csv_to_dataset,
    load_metadata,
    missingness_overview,
    resolve_checkpoint,
    run_model,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "demo" / "model.pt"

ACTION_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}
ACTION_HEADLINES = {
    0: "Safe — missingness looks random",
    1: "Use caution — missingness depends on observed data",
    2: "Stop — missingness likely depends on the missing values",
}
ACTION_BODIES = {
    0: (
        "Complete-case analysis is valid. Simple imputation "
        "(mean/median, listwise deletion) is acceptable."
    ),
    1: (
        "Use multiple imputation (MICE, Amelia II) or likelihood-based "
        "methods (EM, full-information maximum likelihood)."
    ),
    2: (
        "Use sensitivity analysis, selection models (Heckman), or "
        "pattern-mixture models. Standard imputation may be biased."
    ),
}


@st.cache_resource(show_spinner=False)
def _load_model_cached(path: str):
    return build_model(path)


# =============================================================================
# UI helpers
# =============================================================================

TRAFFIC_LIGHT_CSS = """
<style>
.lacuna-light-card {
    border-radius: 18px;
    padding: 28px 32px;
    margin: 12px 0;
    color: white;
    box-shadow: 0 6px 24px rgba(0,0,0,0.18);
}
.lacuna-light-label {
    font-size: 14px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    opacity: 0.85;
    margin-bottom: 6px;
}
.lacuna-light-headline {
    font-size: 32px;
    font-weight: 700;
    margin: 0 0 10px 0;
    line-height: 1.15;
}
.lacuna-light-body {
    font-size: 16px;
    opacity: 0.95;
    line-height: 1.45;
}
.lacuna-confidence {
    margin-top: 14px;
    font-size: 13px;
    opacity: 0.85;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
</style>
"""


def render_traffic_light(result: dict) -> None:
    action_id = result["action_id"]
    color = ACTION_COLORS[action_id]
    label = ACTION_LABELS[action_id]
    headline = ACTION_HEADLINES[action_id]
    body = ACTION_BODIES[action_id]
    conf = result["confidence"] * 100
    pred = result["predicted_class_name"]

    st.markdown(TRAFFIC_LIGHT_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="lacuna-light-card" style="background: {color};">
            <div class="lacuna-light-label">Recommendation · {label}</div>
            <div class="lacuna-light-headline">{headline}</div>
            <div class="lacuna-light-body">{body}</div>
            <div class="lacuna-confidence">
                predicted mechanism: {pred} · confidence: {conf:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Page
# =============================================================================

st.set_page_config(
    page_title="Lacuna — Missing Data Mechanism Classifier",
    page_icon="🚦",
    layout="wide",
)

st.title("Lacuna")
st.caption(
    "Classify the missingness mechanism of a tabular dataset — "
    "MCAR (random), MAR (depends on observed), or MNAR (depends on the missing values themselves)."
)

# ----- Sidebar: model info -----
with st.sidebar:
    st.subheader("Model")
    ckpt_path = resolve_checkpoint(DEFAULT_CHECKPOINT)
    if not ckpt_path.exists():
        st.error(
            f"No checkpoint at `{ckpt_path}`. Set `{CHECKPOINT_ENV}` or "
            f"copy a `.pt` file to `demo/model.pt`."
        )
        st.stop()

    try:
        model, n_params = _load_model_cached(str(ckpt_path))
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    meta = load_metadata(ckpt_path)
    st.markdown(f"**Checkpoint:** `{ckpt_path.name}`")
    st.markdown(f"**Parameters:** {n_params:,}")
    if meta.get("trained_at"):
        st.markdown(f"**Trained:** {meta['trained_at']}")
    if meta.get("val_accuracy") is not None:
        st.markdown(f"**Val accuracy:** {meta['val_accuracy']*100:.1f}%")
    if meta.get("config"):
        st.markdown(f"**Config:** `{meta['config']}`")

    st.divider()
    st.subheader("Sample data")
    sample_dir = PROJECT_ROOT / "demo" / "sample_data"
    samples = sorted(p for p in sample_dir.glob("*.csv") if not p.name.startswith("_"))
    sample_choice = "—"
    if samples:
        st.caption("Pick one to load instead of uploading.")
        sample_choice = st.selectbox(
            "Sample dataset", ["—"] + [s.name for s in samples], label_visibility="collapsed"
        )


# ----- Main: input row -----
left, right = st.columns([1, 1])

with left:
    st.subheader("1 · Load a dataset")
    uploaded = st.file_uploader(
        "Drop a CSV here", type=["csv"],
        help="Numeric columns only. NaN cells are treated as missing.",
    )
    if uploaded is None and sample_choice != "—":
        uploaded_bytes = (sample_dir / sample_choice).read_bytes()
        upload_name = sample_choice
    elif uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        upload_name = uploaded.name
    else:
        uploaded_bytes = None
        upload_name = None

    analyze = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
        disabled=uploaded_bytes is None,
    )

with right:
    st.subheader("How it works")
    st.markdown(
        "Lacuna is a transformer-based mixture-of-experts classifier trained on "
        "millions of semi-synthetic tabular datasets. It looks at the *pattern* "
        "of missing cells — not the values — to decide which mechanism best "
        "explains them. The verdict tells you what kind of imputation strategy "
        "is statistically defensible."
    )

if not analyze or uploaded_bytes is None:
    st.info("Upload a CSV (or pick a sample) and hit **Analyze** to get a verdict.")
    st.stop()


# =============================================================================
# Pipeline
# =============================================================================

st.divider()
st.subheader("2 · Walking through the analysis")

try:
    loaded = csv_to_dataset(uploaded_bytes, upload_name, max_rows=MODEL_DEFAULTS["max_rows"])
except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

dataset = loaded.dataset

if dataset.d > MODEL_DEFAULTS["max_cols"]:
    st.warning(
        f"Dataset has {dataset.d} columns; the model supports "
        f"{MODEL_DEFAULTS['max_cols']}. Only the first will be used."
    )

overview = missingness_overview(dataset)
if overview["n_missing"] == 0:
    st.error(
        "No missing values found in the dataset. Lacuna classifies *missingness* — "
        "the data must contain NaN cells. If your CSV uses a sentinel "
        "(e.g. -999, 'NA'), convert those to NaN first."
    )
    st.stop()

with st.status("Running Lacuna…", expanded=True) as status:
    st.write(f"Parsed **{upload_name}** → {dataset.n} rows × {dataset.d} columns")
    if loaded.dropped_columns:
        preview = ", ".join(f"`{c}`" for c in loaded.dropped_columns[:6])
        more = "…" if len(loaded.dropped_columns) > 6 else ""
        st.write(f"Dropped {len(loaded.dropped_columns)} non-numeric column(s): {preview}{more}")
    if loaded.subsampled_from:
        st.write(f"Subsampled to {dataset.n} rows (from {loaded.subsampled_from}).")
    st.write(
        f"Missing cells: **{overview['n_missing']:,} / {overview['n_total']:,}** "
        f"({overview['pct_missing']:.1f}%)"
    )
    time.sleep(0.4)

    st.write("**Per-column missing rate** (cross-column variance is one of the model's features):")
    miss_df = pd.DataFrame({
        "column": overview["feature_names"],
        "missing_rate": overview["per_col_rate"],
    }).set_index("column")
    st.bar_chart(miss_df, height=180)
    time.sleep(0.4)

    if dataset.d >= 2:
        st.write("**Cross-column missingness correlation** "
                 "(MAR/MNAR patterns tend to show structured correlations):")
        corr = cross_column_corr(dataset)
        corr_df = pd.DataFrame(
            corr,
            index=overview["feature_names"],
            columns=overview["feature_names"],
        )
        st.dataframe(
            corr_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"),
            use_container_width=True,
            height=min(60 + 30 * dataset.d, 360),
        )
        time.sleep(0.4)

    st.write("**Tokenizing and running the model…**")
    t0 = time.time()
    result = run_model(model, dataset)
    elapsed = time.time() - t0
    st.write(f"Forward pass: **{elapsed*1000:.0f} ms** on CPU.")

    status.update(label="Analysis complete", state="complete", expanded=False)


# =============================================================================
# Verdict
# =============================================================================

st.divider()
st.subheader("3 · Verdict")
render_traffic_light(result)

prob_df = pd.DataFrame({
    "mechanism": list(CLASS_NAMES),
    "probability": result["p_class"],
}).set_index("mechanism")
st.write("**Posterior over mechanisms**")
st.bar_chart(prob_df, height=200)


# =============================================================================
# Details (collapsible)
# =============================================================================

with st.expander("Show full details", expanded=False):
    cols = st.columns(2)

    with cols[0]:
        st.markdown("**Posterior probabilities**")
        st.dataframe(
            pd.DataFrame({
                "mechanism": list(CLASS_NAMES),
                "p(mechanism | data)": [f"{p*100:.2f}%" for p in result["p_class"]],
            }).set_index("mechanism"),
            use_container_width=True,
        )
        st.markdown(
            f"**Entropy:** {result['entropy']:.3f} / {result['max_entropy']:.3f} "
            f"(normalized {result['normalized_entropy']:.2f})  \n"
            f"**Confidence:** {result['confidence']*100:.1f}%"
        )

    with cols[1]:
        st.markdown("**Bayes risk by action**")
        risks = result["expected_risks"]
        min_risk = min(risks)
        action_rows = []
        for label, risk in zip(["Green (assume MCAR)", "Yellow (assume MAR)", "Red (assume MNAR)"], risks):
            action_rows.append({
                "action": label,
                "expected_risk": f"{risk:.3f}" + ("  ← min" if risk == min_risk else ""),
            })
        st.dataframe(
            pd.DataFrame(action_rows).set_index("action"),
            use_container_width=True,
        )
        st.caption(
            "The recommended action is the one minimizing expected loss under "
            "the model's posterior."
        )

    st.markdown("**Full recommendation**")
    st.info(ACTION_BODIES[result["action_id"]])

    st.markdown("**Raw JSON**")
    st.code(json.dumps(result, indent=2), language="json")
