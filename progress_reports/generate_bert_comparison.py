#!/usr/bin/env python3
"""
Generate the BERT vs Lacuna comparison diagram as a PDF figure.

Produces: bert_lacuna_comparison.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colours ──────────────────────────────────────────────────────────
C_BERT    = "#3b82f6"  # blue
C_LACUNA  = "#7c3aed"  # purple
C_SHARED  = "#6366f1"  # indigo (shared elements)
C_ARROW   = "#475569"
C_BG      = "#f8fafc"


def rbox(ax, x, y, w, h, text, color, fontsize=7.5, alpha=0.12,
         lw=1.2, bold=False, mono=False):
    """Draw a rounded box with centred text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=(*matplotlib.colors.to_rgb(color), alpha),
        edgecolor=color, linewidth=lw,
    )
    ax.add_patch(box)
    fam = "monospace" if mono else "sans-serif"
    wt = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="black", weight=wt, fontfamily=fam)


def arrow_down(ax, x, y1, y2, label="", color=C_ARROW):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0))
    if label:
        ax.text(x + 0.01, (y1 + y2) / 2, label, fontsize=5.5,
                color="#64748b", ha="left", va="center", style="italic")


def section_label(ax, x, y, text, color, fontsize=8):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, weight="bold")


def dim_text(ax, x, y, text, color="#6366f1"):
    ax.text(x, y, text, ha="center", va="center", fontsize=5.5,
            color=color, fontfamily="monospace")


# ── Layout constants ─────────────────────────────────────────────────
LX = 0.25   # BERT column centre
RX = 0.75   # Lacuna column centre
BW = 0.38   # box width
SH = 0.028  # standard box height

# Vertical positions (shared between columns)
stages = {
    "title":  0.96,
    "input":  0.88,
    "embed":  0.76,
    "trans":  0.60,
    "pool":   0.44,
    "output": 0.26,
    "result": 0.14,
}

# ── Figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# Divider
ax.plot([0.50, 0.50], [0.04, 0.93], color="#cbd5e1", lw=1.0,
        linestyle="--", zorder=0)

# ── Titles ───────────────────────────────────────────────────────────
ax.text(LX, stages["title"], "BERT-Base", ha="center", va="center",
        fontsize=14, weight="bold", color=C_BERT)
ax.text(LX, stages["title"] - 0.02, "110M params  |  h=768, L=12, H=12",
        ha="center", va="center", fontsize=7, color="#64748b")

ax.text(RX, stages["title"], "Lacuna-Base", ha="center", va="center",
        fontsize=14, weight="bold", color=C_LACUNA)
ax.text(RX, stages["title"] - 0.02, "901K params  |  h=128, L=4, H=4",
        ha="center", va="center", fontsize=7, color="#64748b")

# ══════════════════════════════════════════════════════════════════════
# INPUT LAYER
# ══════════════════════════════════════════════════════════════════════
stage_y = stages["input"]

section_label(ax, LX, stage_y + 0.04, "Input", C_BERT)
section_label(ax, RX, stage_y + 0.04, "Input", C_LACUNA)

# BERT input
rbox(ax, LX, stage_y, BW, 0.055, "", C_BERT, alpha=0.06)
ax.text(LX, stage_y + 0.015, "Sentence with [MASK] tokens", ha="center",
        va="center", fontsize=7.5, weight="bold", color=C_BERT)
ax.text(LX, stage_y - 0.005, "[The] [cat] [MASK] [on] [the] [mat]",
        ha="center", va="center", fontsize=7, color="#1e40af", fontfamily="monospace")
dim_text(ax, LX, stage_y - 0.035, "[seq_len] discrete token IDs", C_BERT)

# Lacuna input
rbox(ax, RX, stage_y, BW, 0.055, "", C_LACUNA, alpha=0.06)
ax.text(RX, stage_y + 0.015, "Dataset row with missing cells", ha="center",
        va="center", fontsize=7.5, weight="bold", color=C_LACUNA)
ax.text(RX, stage_y - 0.005, "[2.3,1,0,0.0] [0,0,0,0.33] [-1.2,1,0,0.67]",
        ha="center", va="center", fontsize=6.5, color="#6d28d9", fontfamily="monospace")
dim_text(ax, RX, stage_y - 0.035, "[B, 128, 48, 4]  continuous 4D tokens", C_LACUNA)

arrow_down(ax, LX, stage_y - 0.045, stages["embed"] + 0.065)
arrow_down(ax, RX, stage_y - 0.045, stages["embed"] + 0.065)

# ══════════════════════════════════════════════════════════════════════
# EMBEDDING LAYER
# ══════════════════════════════════════════════════════════════════════
stage_y = stages["embed"]

section_label(ax, LX, stage_y + 0.06, "Token Embedding", C_BERT)
section_label(ax, RX, stage_y + 0.06, "Token Embedding", C_LACUNA)

# BERT embedding
rbox(ax, LX, stage_y + 0.03, BW, 0.04, "", C_BERT, alpha=0.06)
ax.text(LX, stage_y + 0.04, "Sum of three embeddings:", ha="center",
        va="center", fontsize=7, weight="bold", color=C_BERT)
ax.text(LX, stage_y + 0.02, "Token Embed (30522→768) + Position (512→768) + Segment (2→768)",
        ha="center", va="center", fontsize=5.5, color="#1e40af")
dim_text(ax, LX, stage_y - 0.005, "→ [seq_len, 768]", C_BERT)

# Lacuna embedding
rbox(ax, RX, stage_y + 0.03, BW, 0.04, "", C_LACUNA, alpha=0.06)
ax.text(RX, stage_y + 0.04, "Concat of four embeddings:", ha="center",
        va="center", fontsize=7, weight="bold", color=C_LACUNA)
ax.text(RX, stage_y + 0.023,
        "Value Lin(1→32) | Obs Emb(2,32) | Mask Emb(2,32) | Pos Emb(32,32)",
        ha="center", va="center", fontsize=5.5, color="#6d28d9")
ax.text(RX, stage_y + 0.008, "→ Concat → Linear(128→128) → LayerNorm",
        ha="center", va="center", fontsize=5.5, color="#6d28d9")
dim_text(ax, RX, stage_y - 0.005, "→ [B, 128, 48, 128]", C_LACUNA)

# Comparison note
rbox(ax, 0.50, stage_y - 0.035, 0.55, 0.025,
     "Both: learned embeddings + positional encoding. BERT sums; Lacuna concatenates + projects.",
     C_SHARED, fontsize=6, alpha=0.08)

arrow_down(ax, LX, stage_y - 0.055, stages["trans"] + 0.11)
arrow_down(ax, RX, stage_y - 0.055, stages["trans"] + 0.11)

# ══════════════════════════════════════════════════════════════════════
# TRANSFORMER ENCODER
# ══════════════════════════════════════════════════════════════════════
stage_y = stages["trans"]

section_label(ax, LX, stage_y + 0.10, "Transformer Encoder", C_BERT)
section_label(ax, RX, stage_y + 0.10, "Transformer Encoder", C_LACUNA)

# BERT transformer
rbox(ax, LX, stage_y + 0.05, BW, 0.08, "", C_BERT, alpha=0.06)
ax.text(LX, stage_y + 0.075, "×12 layers of:", ha="center",
        va="center", fontsize=7.5, weight="bold", color=C_BERT)
ax.text(LX, stage_y + 0.055, "Multi-Head Self-Attention", ha="center",
        va="center", fontsize=7, color=C_BERT)
ax.text(LX, stage_y + 0.04, "  H=12, dₖ=64, full sequence scope", ha="center",
        va="center", fontsize=6, color="#1e40af")
ax.text(LX, stage_y + 0.02, "Feed-Forward Network", ha="center",
        va="center", fontsize=7, color=C_BERT)
ax.text(LX, stage_y + 0.005, "  768 → 3072 → 768, GELU", ha="center",
        va="center", fontsize=6, color="#1e40af")

# QKV formula
rbox(ax, LX, stage_y - 0.035, BW, 0.03, "", C_BERT, alpha=0.04)
ax.text(LX, stage_y - 0.028, "Attn = softmax(QKᵀ / √dₖ) V", ha="center",
        va="center", fontsize=7, color=C_BERT, fontfamily="monospace", weight="bold")
ax.text(LX, stage_y - 0.043, "Q,K,V ∈ ℝ^{seq×768},  12 heads × 64 dims",
        ha="center", va="center", fontsize=5.5, color="#1e40af")
dim_text(ax, LX, stage_y - 0.06, "→ [seq_len, 768]", C_BERT)

# Lacuna transformer
rbox(ax, RX, stage_y + 0.05, BW, 0.08, "", C_LACUNA, alpha=0.06)
ax.text(RX, stage_y + 0.075, "×4 layers of:", ha="center",
        va="center", fontsize=7.5, weight="bold", color=C_LACUNA)
ax.text(RX, stage_y + 0.055, "Row-Wise Multi-Head Self-Attention", ha="center",
        va="center", fontsize=7, color=C_LACUNA)
ax.text(RX, stage_y + 0.04, "  H=4, dₖ=32, within-row scope only", ha="center",
        va="center", fontsize=6, color="#6d28d9")
ax.text(RX, stage_y + 0.02, "Feed-Forward Network", ha="center",
        va="center", fontsize=7, color=C_LACUNA)
ax.text(RX, stage_y + 0.005, "  128 → 512 → 128, GELU", ha="center",
        va="center", fontsize=6, color="#6d28d9")

# QKV formula
rbox(ax, RX, stage_y - 0.035, BW, 0.03, "", C_LACUNA, alpha=0.04)
ax.text(RX, stage_y - 0.028, "Attn = softmax(QKᵀ / √32) V", ha="center",
        va="center", fontsize=7, color=C_LACUNA, fontfamily="monospace", weight="bold")
ax.text(RX, stage_y - 0.043, "Q,K,V ∈ ℝ^{d×128},  4 heads × 32 dims",
        ha="center", va="center", fontsize=5.5, color="#6d28d9")
dim_text(ax, RX, stage_y - 0.06, "→ [B, 128, 48, 128]", C_LACUNA)

# Comparison note
rbox(ax, 0.50, stage_y - 0.09, 0.60, 0.03,
     "Same QKV formula.  BERT: words attend to all words.  Lacuna: features attend to features in same row.",
     C_SHARED, fontsize=6, alpha=0.08)

arrow_down(ax, LX, stage_y - 0.11, stages["pool"] + 0.085)
arrow_down(ax, RX, stage_y - 0.11, stages["pool"] + 0.085)

# ══════════════════════════════════════════════════════════════════════
# POOLING / REPRESENTATION
# ══════════════════════════════════════════════════════════════════════
stage_y = stages["pool"]

section_label(ax, LX, stage_y + 0.075, "Sequence Representation", C_BERT)
section_label(ax, RX, stage_y + 0.075, "Hierarchical Pooling", C_LACUNA)

# BERT [CLS]
rbox(ax, LX, stage_y + 0.03, BW, 0.05, "", C_BERT, alpha=0.06)
ax.text(LX, stage_y + 0.045, "[CLS] Token", ha="center",
        va="center", fontsize=8, weight="bold", color=C_BERT)
ax.text(LX, stage_y + 0.025, "Special token at position 0", ha="center",
        va="center", fontsize=6.5, color="#1e40af")
ax.text(LX, stage_y + 0.01, "Its final-layer representation = sentence embedding",
        ha="center", va="center", fontsize=6, color="#1e40af")
dim_text(ax, LX, stage_y - 0.01, "→ [768]", C_BERT)

# Lacuna hierarchical pooling
rbox(ax, RX, stage_y + 0.03, BW, 0.07, "", C_LACUNA, alpha=0.06)
ax.text(RX, stage_y + 0.055, "Two-Stage Attention Pooling", ha="center",
        va="center", fontsize=8, weight="bold", color=C_LACUNA)
ax.text(RX, stage_y + 0.035, "Stage 1: features → row  (AttnPool, learned weights)",
        ha="center", va="center", fontsize=6, color="#6d28d9")
dim_text(ax, RX, stage_y + 0.02, "[B, 128, 48, 128] → [B, 128, 128]", C_LACUNA)
ax.text(RX, stage_y + 0.005, "Stage 2: rows → dataset  (AttnPool → Linear → LN)",
        ha="center", va="center", fontsize=6, color="#6d28d9")
dim_text(ax, RX, stage_y - 0.008, "[B, 128, 128] → [B, 64]", C_LACUNA)

# Comparison note
rbox(ax, 0.50, stage_y - 0.045, 0.60, 0.03,
     "BERT uses a special [CLS] token. Lacuna learns attention weights over features then rows.",
     C_SHARED, fontsize=6, alpha=0.08)

arrow_down(ax, LX, stage_y - 0.065, stages["output"] + 0.085)
arrow_down(ax, RX, stage_y - 0.065, stages["output"] + 0.085)

# ══════════════════════════════════════════════════════════════════════
# OUTPUT HEAD
# ══════════════════════════════════════════════════════════════════════
stage_y = stages["output"]

section_label(ax, LX, stage_y + 0.075, "Output Head", C_BERT)
section_label(ax, RX, stage_y + 0.075, "Output: MoE + Decision", C_LACUNA)

# BERT output
rbox(ax, LX, stage_y + 0.03, BW, 0.06, "", C_BERT, alpha=0.06)
ax.text(LX, stage_y + 0.05, "MLM Head", ha="center",
        va="center", fontsize=8, weight="bold", color=C_BERT)
ax.text(LX, stage_y + 0.03, "Linear(768 → 30522) → Softmax over vocab",
        ha="center", va="center", fontsize=6, color="#1e40af")
ax.text(LX, stage_y + 0.015, 'Predict: "sat" (98%)', ha="center",
        va="center", fontsize=7, color=C_BERT, weight="bold")
dim_text(ax, LX, stage_y - 0.005, "→ P(word | context)", C_BERT)

# Lacuna output
rbox(ax, RX, stage_y + 0.03, BW, 0.06, "", C_LACUNA, alpha=0.06)
ax.text(RX, stage_y + 0.05, "MoE Gating Network", ha="center",
        va="center", fontsize=8, weight="bold", color=C_LACUNA)
ax.text(RX, stage_y + 0.033, "Concat [z(64) | ε(3) | f(16)] = 83 dims",
        ha="center", va="center", fontsize=6, color="#6d28d9")
ax.text(RX, stage_y + 0.018, "MLP(83→64→3) → softmax(·/T)",
        ha="center", va="center", fontsize=6, color="#6d28d9")
ax.text(RX, stage_y + 0.003, "→ Bayes-optimal decision (loss matrix)",
        ha="center", va="center", fontsize=6, color="#6d28d9")
dim_text(ax, RX, stage_y - 0.012, "→ P(MCAR|D), P(MAR|D), P(MNAR|D) → action",
         C_LACUNA)

arrow_down(ax, LX, stage_y - 0.04, stages["result"] + 0.03)
arrow_down(ax, RX, stage_y - 0.04, stages["result"] + 0.03)

# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════
stage_y = stages["result"]

rbox(ax, LX, stage_y, BW, 0.035, "", C_BERT, alpha=0.06)
ax.text(LX, stage_y + 0.005, 'Predicted word: "sat"', ha="center",
        va="center", fontsize=8, weight="bold", color=C_BERT)
ax.text(LX, stage_y - 0.01, "Trained on BookCorpus + Wikipedia",
        ha="center", va="center", fontsize=6, color="#1e40af")

rbox(ax, RX, stage_y, BW, 0.035, "", C_LACUNA, alpha=0.06)
ax.text(RX, stage_y + 0.005, "MAR: 72% | MCAR: 15% | MNAR: 13%",
        ha="center", va="center", fontsize=7, weight="bold", color=C_LACUNA,
        fontfamily="monospace")
ax.text(RX, stage_y - 0.01, "→ Yellow: use multiple imputation",
        ha="center", va="center", fontsize=6.5, color="#6d28d9")

# ══════════════════════════════════════════════════════════════════════
# BOTTOM COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════
table_y = 0.055
rbox(ax, 0.50, table_y, 0.92, 0.055, "", C_SHARED, alpha=0.04, lw=1.5)
ax.text(0.50, table_y + 0.02, "Shared: QKV attention formula  |  Pre-norm residuals  |  GELU activation  |  Dropout regularization  |  Learned positional embeddings",
        ha="center", va="center", fontsize=6.5, color=C_SHARED, weight="bold")
ax.text(0.50, table_y - 0.002,
        "Adapted: 4D tokens (not vocab)  |  Row-wise scope (not full seq)  |  Attention pooling (not [CLS])  |  Reconstruction heads  |  Missingness features  |  MoE gating",
        ha="center", va="center", fontsize=6, color="#7c3aed")
ax.text(0.50, table_y - 0.02,
        "Scale: 122× smaller (901K vs 110M)  |  4 vs 12 layers  |  128 vs 768 hidden  |  Tabular data vs natural language",
        ha="center", va="center", fontsize=6, color="#64748b")

# ── Save ─────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig("bert_lacuna_comparison.pdf", bbox_inches="tight", dpi=300)
plt.savefig("bert_lacuna_comparison.png", bbox_inches="tight", dpi=200)
print("Saved bert_lacuna_comparison.pdf and .png")
