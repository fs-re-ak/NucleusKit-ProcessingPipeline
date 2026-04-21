"""Self-contained HTML report: original vs repaired RMS (matplotlib inline)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fig_to_base64(fig: matplotlib.figure.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def write_comparison_report(
    out_path: Path,
    original: pd.DataFrame,
    repaired: pd.DataFrame,
    target_channel_idx: int,
    channel_names: Sequence[str],
) -> None:
    """Write a single HTML file with stats and plots."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts = original.iloc[:, 0].values
    n_channels = len(channel_names)
    o_data = original.iloc[:, 1 : 1 + n_channels].values.astype(float)
    r_data = repaired.iloc[:, 1 : 1 + n_channels].values.astype(float)

    rows = []
    for i, name in enumerate(channel_names):
        diff = r_data[:, i] - o_data[:, i]
        mae = float(np.nanmean(np.abs(diff)))
        rmse = float(np.sqrt(np.nanmean(diff**2)))
        rows.append(
            f"<tr><td>{name}</td><td>{mae:.6f}</td><td>{rmse:.6f}</td></tr>"
        )

    # Target channel overlay
    fig1, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, o_data[:, target_channel_idx], label="Original", alpha=0.85)
    ax.plot(ts, r_data[:, target_channel_idx], label="Repaired", alpha=0.85)
    ax.set_xlabel("Time")
    ax.set_ylabel("RMS")
    ax.set_title(f"Channel {target_channel_idx} ({channel_names[target_channel_idx]})")
    ax.legend()
    b64_target = _fig_to_base64(fig1)

    # Delta on target
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    delta = r_data[:, target_channel_idx] - o_data[:, target_channel_idx]
    ax2.plot(ts, delta, color="C2")
    ax2.axhline(0.0, color="k", linewidth=0.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Repaired − original")
    ax2.set_title("Delta (target channel)")
    b64_delta = _fig_to_base64(fig2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Channel fix report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; }}
    table {{ border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.6rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    h2 {{ margin-top: 1.5rem; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Channel repair report</h1>
  <p>Target channel index: {target_channel_idx} ({channel_names[target_channel_idx]}).</p>
  <h2>Per-channel MAE / RMSE (repaired vs original)</h2>
  <table>
    <tr><th>Channel</th><th>MAE</th><th>RMSE</th></tr>
    {''.join(rows)}
  </table>
  <h2>Target channel: original vs repaired</h2>
  <p><img src="data:image/png;base64,{b64_target}" alt="Target overlay"/></p>
  <h2>Delta (repaired − original)</h2>
  <p><img src="data:image/png;base64,{b64_delta}" alt="Delta"/></p>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
