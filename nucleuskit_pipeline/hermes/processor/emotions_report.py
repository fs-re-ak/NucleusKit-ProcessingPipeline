"""
Emotions Classifier Report Generator

Produces visual statistics from ``features/emotions/emotionClassifierInputs.csv``:

- One jitter plot per predicted emotion, showing the distribution of the
  L2-normalised per-channel RMS across all windows classified as that emotion.
  A thick horizontal bar marks the per-channel median.

- A pie chart summarising the overall distribution of predicted emotions.

Outputs are written under ``features/emotions/``:
  ``report_jitter_{emotion}.png``   (one file per emotion present in the data)
  ``report_emotion_distribution.png``

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Spring 2026
"""

from __future__ import annotations

from os import path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from nucleuskit_pipeline.logging_utils import printInfo, printError

# Ordered channel RMS feature columns in emotionClassifierInputs.csv.
# AVG_RMS is excluded from per-channel jitter plots (it is a derived scalar).
CHANNEL_RMS_COLUMNS: list[str] = [
    "HEAD_R_RMS",
    "HEAD_L_RMS",
    "CHEEK_R_RMS",
    "CHEEK_L_RMS",
    "EAR_R_RMS",
    "FOREHEAD_L_RMS",
    "BROW_RMS",
    "NOSE_RMS",
]

# Short display labels for the X-axis (same order as CHANNEL_RMS_COLUMNS).
CHANNEL_LABELS: list[str] = [
    "HEAD R",
    "HEAD L",
    "CHEEK R",
    "CHEEK L",
    "EAR R",
    "FOREHEAD L",
    "BROW",
    "NOSE",
]

_EMOTION_COLOURS: dict[str, str] = {
    "Neutral":   "#7f8c8d",
    "Happiness": "#f1c40f",
    "Anger":     "#e74c3c",
    "Surprise":  "#3498db",
    "Contempt":  "#8e44ad",
    "Disgust":   "#27ae60",
    "Fear":      "#e67e22",
    "Sadness":   "#2980b9",
}
_DEFAULT_COLOUR = "#95a5a6"

_RNG_SEED = 42


def _jitter(n: int, width: float = 0.18, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return horizontal jitter offsets centred on 0."""
    rng = rng or np.random.default_rng(_RNG_SEED)
    return rng.uniform(-width, width, size=n)


# ---------------------------------------------------------------------------
# Jitter plots
# ---------------------------------------------------------------------------

def _save_jitter_plot(
    subset: pd.DataFrame,
    emotion: str,
    out_path: str,
    rng: np.random.Generator,
) -> None:
    """Create and save a jitter plot for one emotion."""
    n = len(subset)
    colour = _EMOTION_COLOURS.get(emotion, _DEFAULT_COLOUR)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    for idx, col in enumerate(CHANNEL_RMS_COLUMNS):
        vals = subset[col].to_numpy(dtype=float)

        # Jitter scatter
        jx = idx + _jitter(n, width=0.18, rng=rng)
        ax.scatter(
            jx, vals,
            color=colour,
            alpha=0.45,
            s=14,
            linewidths=0,
            zorder=2,
        )

        # Median bar
        ax.plot(
            idx, np.median(vals),
            marker="_",
            markersize=28,
            markeredgewidth=2.5,
            color="#2c3e50",
            zorder=3,
        )

    ax.set_xticks(range(len(CHANNEL_LABELS)))
    ax.set_xticklabels(CHANNEL_LABELS, rotation=30, ha="right", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.set_ylabel("L2-normalised RMS", fontsize=11)
    ax.set_xlabel("Channel", fontsize=11)
    ax.set_title(
        f"{emotion}  (n = {n:,})",
        fontsize=13,
        fontweight="bold",
        color=colour,
        pad=10,
    )
    ax.grid(axis="y", color="#cccccc", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pie chart
# ---------------------------------------------------------------------------

def _save_pie_chart(counts: pd.Series, out_path: str) -> None:
    """Create and save the emotion distribution pie chart."""
    labels = counts.index.tolist()
    sizes = counts.values.tolist()
    colours = [_EMOTION_COLOURS.get(lbl, _DEFAULT_COLOUR) for lbl in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#fafafa")

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colours,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(9)
    for t in texts:
        t.set_fontsize(11)

    total = sum(sizes)
    ax.set_title(
        f"Emotion Distribution  (n = {total:,})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(recpath: str) -> None:
    """
    Generate the statistics report for ``emotionClassifierInputs.csv``.

    Args:
        recpath: Path to the recording directory.
    """
    csv_path = path.join(recpath, "features", "emotions", "emotionClassifierInputs.csv")
    out_dir  = path.join(recpath, "features", "emotions")

    if not path.isfile(csv_path):
        printError(f"[emotionsReportGenerator] File not found: {csv_path}")
        return

    printInfo(f"[emotionsReportGenerator] Generating report from {csv_path}")
    df = pd.read_csv(csv_path)

    # Keep only windows with a valid predicted label.
    df = df.dropna(subset=["PredictedLabel"])

    if df.empty:
        printError("[emotionsReportGenerator] No labelled windows — skipping report")
        return

    missing = [c for c in CHANNEL_RMS_COLUMNS if c not in df.columns]
    if missing:
        printError(f"[emotionsReportGenerator] Missing columns: {missing}")
        return

    os.makedirs(out_dir, exist_ok=True)

    emotions = sorted(df["PredictedLabel"].unique().tolist())
    rng = np.random.default_rng(_RNG_SEED)

    # -- Jitter plots (one per emotion) ------------------------------------
    for emotion in emotions:
        subset = df[df["PredictedLabel"] == emotion]
        safe_name = emotion.lower().replace(" ", "_")
        out_path  = path.join(out_dir, f"report_jitter_{safe_name}.png")
        _save_jitter_plot(subset, emotion, out_path, rng)
        printInfo(f"[emotionsReportGenerator] Saved {out_path}")

    # -- Pie chart ---------------------------------------------------------
    counts   = df["PredictedLabel"].value_counts()
    out_path = path.join(out_dir, "report_emotion_distribution.png")
    _save_pie_chart(counts, out_path)
    printInfo(f"[emotionsReportGenerator] Saved {out_path}")

    printInfo("[emotionsReportGenerator] Report generation complete")
