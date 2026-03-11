#!/usr/bin/env python3
"""
Generate the Lacuna architecture diagram as a PDF figure.

Produces: lacuna_architecture_diagram.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────
C_TOKEN   = "#7c3aed"   # purple – input / token
C_EMBED   = "#3b82f6"   # blue – embedding
C_TRANS   = "#4f46e5"   # indigo – transformer
C_POOL    = "#0d9488"   # teal – pooling
C_RECON   = "#d97706"   # amber – reconstruction
C_FEAT    = "#059669"   # emerald – missingness features
C_MOE     = "#dc2626"   # red – MoE / gating
C_DECIDE  = "#16a34a"   # green – decision
C_BG      = "#f8fafc"   # slate-50 background
C_ARROW   = "#475569"   # slate-600


def rounded_box(ax, x, y, w, h, text, color, fontsize=8, text_color="black",
                alpha=0.15, linewidth=1.5, bold=False):
    """Draw a rounded rectangle with centred text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=(*matplotlib.colors.to_rgb(color), alpha),
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, weight=weight, wrap=True,
            fontfamily="monospace" if "[" in text else "sans-serif")


def arrow(ax, x1, y1, x2, y2, label="", color=C_ARROW):
    """Draw a downward arrow with optional label."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.02, my, label, fontsize=6.5, color="#64748b",
                ha="left", va="center", style="italic")


def section_title(ax, x, y, text, color):
    """Draw a section title."""
    ax.text(x, y, text, ha="center", va="center", fontsize=10,
            color=color, weight="bold")


def dim_label(ax, x, y, text):
    """Draw a dimension annotation."""
    ax.text(x, y, text, ha="center", va="center", fontsize=6.5,
            color="#6366f1", fontfamily="monospace")


# ── Main figure ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 16))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("auto")
ax.axis("off")
fig.patch.set_facecolor("white")

# Vertical layout positions (top to bottom)
Y = {
    "title": 0.97,
    "input": 0.91,
    "embed_title": 0.855,
    "embed": 0.82,
    "trans_title": 0.745,
    "attn": 0.70,
    "ffn": 0.64,
    "trans_repeat": 0.595,
    "pool_title": 0.555,
    "row_pool": 0.52,
    "ds_pool": 0.465,
    "branch_top": 0.42,
    "recon": 0.375,
    "feat": 0.375,
    "evidence_label": 0.375,
    "gate_title": 0.30,
    "gate_concat": 0.265,
    "gate_mlp": 0.22,
    "posterior": 0.165,
    "decision": 0.105,
    "output": 0.06,
}

CX = 0.5  # centre x

# ── Title ───────────────────────────────────────────────────────────
ax.text(CX, Y["title"], "Lacuna-Base Architecture", ha="center", va="center",
        fontsize=14, weight="bold", color="#1e293b")
ax.text(CX, Y["title"] - 0.02, "901,130 parameters  |  h = 128, p = 64, L = 4, H = 4",
        ha="center", va="center", fontsize=7.5, color="#64748b")

# ── 1. Input ────────────────────────────────────────────────────────
section_title(ax, CX, Y["input"] + 0.025, "1. Input Tokens", C_TOKEN)
rounded_box(ax, CX, Y["input"], 0.70, 0.035,
            "T  =  [value, is_observed, mask_type, feature_id]  per cell",
            C_TOKEN, fontsize=8)
dim_label(ax, CX, Y["input"] - 0.025, "[B, 128, 48, 4]")

arrow(ax, CX, Y["input"] - 0.04, CX, Y["embed_title"] + 0.015)

# ── 2. Token Embedding ──────────────────────────────────────────────
section_title(ax, CX, Y["embed_title"], "2. Token Embedding", C_EMBED)

embed_boxes = [
    ("Value Proj\nLinear(1→32)", 0.18),
    ("Obs Embed\nEmbed(2, 32)", 0.39),
    ("Mask Embed\nEmbed(2, 32)", 0.61),
    ("Pos Embed\nEmbed(32, 32)", 0.82),
]
for label, bx in embed_boxes:
    rounded_box(ax, bx, Y["embed"], 0.17, 0.035, label, C_EMBED, fontsize=7)

# Concat + project
arrow(ax, CX, Y["embed"] - 0.025, CX, Y["embed"] - 0.045)
rounded_box(ax, CX, Y["embed"] - 0.06, 0.55, 0.025,
            "Concat [32,32,32,32] → Linear(128→128) → LayerNorm → Dropout",
            C_EMBED, fontsize=7)
dim_label(ax, CX, Y["embed"] - 0.08, "[B, 128, 48, 128]")

arrow(ax, CX, Y["embed"] - 0.09, CX, Y["trans_title"] + 0.015)

# ── 3. Transformer Encoder ──────────────────────────────────────────
section_title(ax, CX, Y["trans_title"], "3. Transformer Layer ×4  (row-wise)", C_TRANS)

# Attention sub-block
rounded_box(ax, CX, Y["attn"], 0.72, 0.06, "", C_TRANS, alpha=0.06)
ax.text(CX, Y["attn"] + 0.02, "Multi-Head Self-Attention (H = 4, dₖ = 32)",
        ha="center", va="center", fontsize=8, weight="bold", color=C_TRANS)

qkv_labels = [("Wq: 128→128", 0.23), ("Wk: 128→128", 0.50), ("Wv: 128→128", 0.77)]
for label, bx in qkv_labels:
    rounded_box(ax, bx, Y["attn"] - 0.005, 0.22, 0.022, label, C_TRANS, fontsize=6.5)

ax.text(CX, Y["attn"] - 0.025, "Attn(Q,K,V) = softmax(QKᵀ/√32) V  →  Wo: 128→128  →  + residual",
        ha="center", va="center", fontsize=6.5, color="#4338ca", fontfamily="monospace")

arrow(ax, CX, Y["attn"] - 0.035, CX, Y["ffn"] + 0.03)

# FFN sub-block
rounded_box(ax, CX, Y["ffn"], 0.72, 0.045, "", C_TRANS, alpha=0.06)
ax.text(CX, Y["ffn"] + 0.012, "Feed-Forward Network", ha="center", va="center",
        fontsize=8, weight="bold", color=C_TRANS)
ax.text(CX, Y["ffn"] - 0.008,
        "Linear(128→512) → GELU → Dropout(0.1) → Linear(512→128) → + residual",
        ha="center", va="center", fontsize=6.5, color="#4338ca", fontfamily="monospace")

dim_label(ax, CX, Y["ffn"] - 0.028, "[B, 128, 48, 128]  (pre-norm at each sub-layer)")

# Repeat indicator
ax.text(CX, Y["trans_repeat"], "↻ Repeat ×4 layers total",
        ha="center", va="center", fontsize=8, color=C_TRANS, weight="bold",
        style="italic")

arrow(ax, CX, Y["trans_repeat"] - 0.015, CX, Y["pool_title"] + 0.015)

# ── 4. Hierarchical Pooling ─────────────────────────────────────────
section_title(ax, CX, Y["pool_title"], "4. Hierarchical Attention Pooling", C_POOL)

# Row pooling
rounded_box(ax, CX, Y["row_pool"], 0.60, 0.035, "", C_POOL, alpha=0.08)
ax.text(CX, Y["row_pool"] + 0.008, "Row Pooling: features → rows", ha="center",
        va="center", fontsize=8, weight="bold", color=C_POOL)
ax.text(CX, Y["row_pool"] - 0.008,
        "score = Linear(64→1)(tanh(Linear(128→64)(h)))  →  softmax  →  weighted sum",
        ha="center", va="center", fontsize=6.5, color="#0f766e", fontfamily="monospace")
dim_label(ax, CX + 0.35, Y["row_pool"], "[B, 128, 48, 128] → [B, 128, 128]")

arrow(ax, CX, Y["row_pool"] - 0.022, CX, Y["ds_pool"] + 0.022)

# Dataset pooling
rounded_box(ax, CX, Y["ds_pool"], 0.60, 0.035, "", C_POOL, alpha=0.08)
ax.text(CX, Y["ds_pool"] + 0.008, "Dataset Pooling: rows → evidence", ha="center",
        va="center", fontsize=8, weight="bold", color=C_POOL)
ax.text(CX, Y["ds_pool"] - 0.008,
        "AttnPool(rows) → Linear(128→64) → LayerNorm → z",
        ha="center", va="center", fontsize=6.5, color="#0f766e", fontfamily="monospace")
dim_label(ax, CX + 0.35, Y["ds_pool"], "[B, 128, 128] → [B, 64]")

# ── 5. Three-way branch ─────────────────────────────────────────────
# Evidence (centre), Reconstruction (left), Missingness features (right)
branch_y = Y["branch_top"]

# Evidence arrow (straight down)
arrow(ax, CX, Y["ds_pool"] - 0.022, CX, Y["gate_concat"] + 0.02,
      label="evidence z ∈ ℝ⁶⁴", color=C_POOL)

# Reconstruction branch (left)
# Arrow from transformer output to recon
ax.annotate("", xy=(0.18, Y["recon"] + 0.025), xytext=(0.25, Y["trans_repeat"] - 0.015),
            arrowprops=dict(arrowstyle="-|>", color=C_RECON, lw=1.0,
                           connectionstyle="arc3,rad=0.15"))
ax.text(0.13, Y["trans_repeat"] - 0.015, "token repr\nH⁽⁴⁾", fontsize=6, color=C_RECON,
        ha="center", va="center", style="italic")

rounded_box(ax, 0.18, Y["recon"], 0.28, 0.06, "", C_RECON, alpha=0.08)
ax.text(0.18, Y["recon"] + 0.018, "Reconstruction Heads", ha="center",
        va="center", fontsize=7.5, weight="bold", color=C_RECON)
ax.text(0.18, Y["recon"] - 0.002, "MCAR: MLP(128→64→1)", ha="center",
        va="center", fontsize=6, color="#92400e")
ax.text(0.18, Y["recon"] - 0.015, "MAR: CrossAttn(Q,K from repr; V from raw vals)",
        ha="center", va="center", fontsize=5.5, color="#92400e")
ax.text(0.18, Y["recon"] - 0.026, "MNAR: MLP + censoring adjustment",
        ha="center", va="center", fontsize=6, color="#92400e")
dim_label(ax, 0.18, Y["recon"] - 0.04, "ε ∈ ℝ³ (natural errors)")

# Arrow from recon to gate
ax.annotate("", xy=(0.35, Y["gate_concat"] + 0.015), xytext=(0.18, Y["recon"] - 0.05),
            arrowprops=dict(arrowstyle="-|>", color=C_RECON, lw=1.0,
                           connectionstyle="arc3,rad=-0.15"))

# Missingness features branch (right)
# Arrow from input to features
ax.annotate("", xy=(0.82, Y["feat"] + 0.025), xytext=(0.80, Y["input"] - 0.025),
            arrowprops=dict(arrowstyle="-|>", color=C_FEAT, lw=1.0,
                           connectionstyle="arc3,rad=-0.3"))
ax.text(0.90, Y["input"] - 0.06, "raw\ntokens T", fontsize=6, color=C_FEAT,
        ha="center", va="center", style="italic")

rounded_box(ax, 0.82, Y["feat"], 0.28, 0.06, "", C_FEAT, alpha=0.08)
ax.text(0.82, Y["feat"] + 0.018, "Missingness Features", ha="center",
        va="center", fontsize=7.5, weight="bold", color=C_FEAT)
ax.text(0.82, Y["feat"] - 0.002, "4 missing rate stats", ha="center",
        va="center", fontsize=6, color="#065f46")
ax.text(0.82, Y["feat"] - 0.013, "3 point-biserial corr + 3 cross-col corr",
        ha="center", va="center", fontsize=6, color="#065f46")
ax.text(0.82, Y["feat"] - 0.024, "4 distributional + 2 Little's test",
        ha="center", va="center", fontsize=6, color="#065f46")
dim_label(ax, 0.82, Y["feat"] - 0.04, "f ∈ ℝ¹⁶ (non-learnable)")

# Arrow from features to gate
ax.annotate("", xy=(0.65, Y["gate_concat"] + 0.015), xytext=(0.82, Y["feat"] - 0.05),
            arrowprops=dict(arrowstyle="-|>", color=C_FEAT, lw=1.0,
                           connectionstyle="arc3,rad=0.15"))

# ── 6. MoE Gating ──────────────────────────────────────────────────
section_title(ax, CX, Y["gate_title"], "5. Mixture-of-Experts Gating", C_MOE)

# Concatenation
rounded_box(ax, CX, Y["gate_concat"], 0.50, 0.022,
            "Concat [z; ε; f]  =  [64 + 3 + 16]  =  83 dims",
            C_MOE, fontsize=7)

arrow(ax, CX, Y["gate_concat"] - 0.015, CX, Y["gate_mlp"] + 0.02)

# Gating MLP
rounded_box(ax, CX, Y["gate_mlp"], 0.65, 0.035, "", C_MOE, alpha=0.08)
ax.text(CX, Y["gate_mlp"] + 0.008, "Gating MLP", ha="center",
        va="center", fontsize=8, weight="bold", color=C_MOE)
ax.text(CX, Y["gate_mlp"] - 0.008,
        "Linear(83→64) → LayerNorm → GELU → Dropout → Linear(64→3)",
        ha="center", va="center", fontsize=6.5, color="#991b1b", fontfamily="monospace")

arrow(ax, CX, Y["gate_mlp"] - 0.022, CX, Y["posterior"] + 0.025)

# ── Posterior ────────────────────────────────────────────────────────
rounded_box(ax, CX, Y["posterior"], 0.55, 0.04, "", C_MOE, alpha=0.08)
ax.text(CX, Y["posterior"] + 0.01, "Temperature-Scaled Softmax  (T = 1.96)",
        ha="center", va="center", fontsize=8, weight="bold", color=C_MOE)
ax.text(CX, Y["posterior"] - 0.008,
        "p(c|D) = softmax(l / 1.96)     ECE = 0.038",
        ha="center", va="center", fontsize=7, color="#7f1d1d", fontfamily="monospace")

# Three class boxes
class_y = Y["posterior"] - 0.035
for i, (label, prob, col) in enumerate([
    ("P(MCAR)", "0.15", "#3b82f6"),
    ("P(MAR)", "0.72", "#16a34a"),
    ("P(MNAR)", "0.13", "#dc2626"),
]):
    bx = 0.28 + i * 0.22
    rounded_box(ax, bx, class_y, 0.16, 0.025,
                f"{label} = {prob}", col, fontsize=7, bold=True, alpha=0.2)

arrow(ax, CX, class_y - 0.018, CX, Y["decision"] + 0.025)

# ── Decision ─────────────────────────────────────────────────────────
rounded_box(ax, CX, Y["decision"], 0.55, 0.04, "", C_DECIDE, alpha=0.08)
ax.text(CX, Y["decision"] + 0.01, "Bayes-Optimal Decision Rule",
        ha="center", va="center", fontsize=8, weight="bold", color=C_DECIDE)
ax.text(CX, Y["decision"] - 0.008,
        "a* = argmin_a  Σ_c  P(c|D) · L[a,c]",
        ha="center", va="center", fontsize=7, color="#14532d", fontfamily="monospace")

arrow(ax, CX, Y["decision"] - 0.025, CX, Y["output"] + 0.015)

# Output
for i, (label, desc, col) in enumerate([
    ("Green", "MCAR → simple imputation", "#16a34a"),
    ("Yellow", "MAR → multiple imputation", "#ca8a04"),
    ("Red", "MNAR → sensitivity analysis", "#dc2626"),
]):
    bx = 0.22 + i * 0.28
    rounded_box(ax, bx, Y["output"], 0.24, 0.025,
                f"{label}: {desc}", col, fontsize=6.5, bold=True, alpha=0.2)

# ── Save ─────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig("lacuna_architecture_diagram.pdf", bbox_inches="tight", dpi=300)
plt.savefig("lacuna_architecture_diagram.png", bbox_inches="tight", dpi=200)
print("Saved lacuna_architecture_diagram.pdf and .png")
