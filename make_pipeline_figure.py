#!/usr/bin/env python3
"""Generate the RAG pipeline diagram for the README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import math

OUT = Path(__file__).parent / "results" / "plots" / "pipeline.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

C = {
    "user":   "#3a7cbd",
    "tfidf":  "#5b8db8",
    "llm":    "#7b5ea7",
    "judge":  "#c87d1e",
    "corpus": "#3a9e5c",
    "prompt": "#7b5ea7",
    "attack": "#c0392b",
    "safe":   "#27ae60",
    "block":  "#c0392b",
    "arrow":  "#555555",
    "bg":     "#fafafa",
    "def_bg": "#eeeaf7",
}

PAD = 0.1   # FancyBboxPatch pad — visual size = specified (w,h) + 2*PAD on each axis

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis("off")


def draw_box(cx, cy, w, h, label, color, fsz=13, tc="white"):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={PAD}",
        facecolor=color, edgecolor="white", linewidth=2, zorder=3,
    ))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fsz, color=tc, fontweight="bold", zorder=4,
            multialignment="center")


def edge(cx, cy, w, h, side):
    """Return the visual edge midpoint of a box (includes pad)."""
    if side == "right":  return (cx + w/2 + PAD, cy)
    if side == "left":   return (cx - w/2 - PAD, cy)
    if side == "top":    return (cx, cy + h/2 + PAD)
    if side == "bottom": return (cx, cy - h/2 - PAD)


def arr(p1, p2, color=None, lw=2.2, gap=0.10):
    """Arrow from box-edge p1 to box-edge p2.
    Tail pulled inward by `gap` so it starts cleanly outside the source box.
    Arrowhead tip lands exactly at p2 (the target box edge).
    """
    x1, y1 = p1
    x2, y2 = p2
    color = color or C["arrow"]
    d = math.hypot(x2 - x1, y2 - y1)
    if d < 1e-6:
        return
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1 + ux * gap, y1 + uy * gap),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15),
        zorder=2,
    )


# ── Box specs: (cx, cy, w, h) ─────────────────────────────────────────────
BOXES = {
    "UQ":  (1.4,  3.1, 1.9, 1.0),   # User Query
    "TF":  (4.2,  3.1, 2.0, 1.0),   # TF-IDF Retrieval
    "LLM": (7.3,  3.1, 2.1, 1.0),   # Main LLM
    "JDG": (10.6, 3.1, 2.1, 1.0),   # Semantic Judge
    "SAF": (13.0, 4.5, 1.6, 0.8),   # Safe
    "BLK": (13.0, 1.7, 1.6, 0.8),   # Blocked
    "CRP": (4.2,  1.0, 2.0, 0.8),   # Doc Corpus
    "SYP": (7.3,  1.0, 2.0, 0.8),   # System Prompt
    "ATK": (4.2,  5.2, 2.0, 0.8),   # Attacker
}


def e(name, side):
    return edge(*BOXES[name], side)


# ── Defense layer shaded region ───────────────────────────────────────────
sx = e("SYP", "left")[0]   - 0.25
ex = e("JDG", "right")[0]  + 0.25
sy = e("SYP", "bottom")[1] - 0.25
ey = e("JDG", "top")[1]    + 0.55
ax.add_patch(FancyBboxPatch(
    (sx, sy), ex - sx, ey - sy,
    boxstyle="round,pad=0.1",
    facecolor=C["def_bg"], edgecolor="#9b7ec8",
    linewidth=2, linestyle="--", zorder=1, alpha=0.9,
))
ax.text((sx + ex) / 2, ey + 0.21, "Defense Layer",
        ha="center", fontsize=16, color="#7b5ea7",
        fontstyle="italic", fontweight="bold", zorder=4)

# ── Draw all boxes ────────────────────────────────────────────────────────
draw_box(*BOXES["UQ"],  "User\nQuery",       C["user"])
draw_box(*BOXES["TF"],  "TF-IDF\nRetrieval", C["tfidf"])
draw_box(*BOXES["LLM"], "Main LLM",          C["llm"])
draw_box(*BOXES["JDG"], "Semantic\nJudge",   C["judge"])
draw_box(*BOXES["SAF"], "✓  Safe",    C["safe"],  fsz=12)
draw_box(*BOXES["BLK"], "✗  Blocked", C["block"], fsz=12)
draw_box(*BOXES["CRP"], "Doc Corpus",    C["corpus"], fsz=12)
draw_box(*BOXES["SYP"], "System Prompt", C["prompt"], fsz=12)
draw_box(*BOXES["ATK"], "⚠  Attacker",   C["attack"], fsz=12)

# ── Arrows ────────────────────────────────────────────────────────────────
# main pipeline (horizontal)
arr(e("UQ",  "right"), e("TF",  "left"))
arr(e("TF",  "right"), e("LLM", "left"))
arr(e("LLM", "right"), e("JDG", "left"))

# vertical feeds
arr(e("CRP", "top"),    e("TF",  "bottom"), C["corpus"])
arr(e("ATK", "bottom"), e("TF",  "top"),    C["attack"])
arr(e("SYP", "top"),    e("LLM", "bottom"), C["prompt"])

# judge outputs (diagonal to Safe / Blocked)
arr(e("JDG", "right"), e("SAF", "left"), C["safe"])
arr(e("JDG", "right"), e("BLK", "left"), C["block"])

fig.savefig(OUT, bbox_inches="tight", dpi=160, facecolor=C["bg"])
plt.close(fig)
print(f"Saved → {OUT}")
