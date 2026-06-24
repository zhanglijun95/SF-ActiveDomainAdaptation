#!/usr/bin/env python3
"""Plot the RTSM anchor figure for the sparse-label SFDA paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["PETS", "LPU", "LPLD", "DDT"]
PURE_SFDA = np.array([46.520, 48.695, 47.043, 53.669])
RTSM = np.array([52.885, 53.420, 55.321, 58.695])
SOURCE_ONLY = 32.608
FULL_TARGET_ORACLE = 62.507


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ljzhang/code/SFDAPaper/6a2704d6e6161016a8915dda/figures"),
        help="Directory where the PDF/PNG figure will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.2,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
            "lines.linewidth": 1.9,
            "lines.markersize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Okabe-Ito-inspired colors remain distinguishable in grayscale print.
    colors = {
        "PETS": "#0072B2",
        "LPU": "#009E73",
        "LPLD": "#D55E00",
        "DDT": "#CC79A7",
    }
    markers = {"PETS": "o", "LPU": "s", "LPLD": "^", "DDT": "D"}
    label_offsets = {"PETS": -0.55, "LPU": 0.35, "LPLD": 0.18, "DDT": 0.12}

    fig, ax = plt.subplots(figsize=(2.65, 1.90))
    x = np.array([0, 1])
    reference_color = "#6E6E6E"

    for value, label in [
        (SOURCE_ONLY, "Source-Only"),
        (FULL_TARGET_ORACLE, "Full-Target Oracle"),
    ]:
        ax.axhline(
            value,
            color=reference_color,
            linewidth=1.0,
            linestyle=(0, (3.0, 2.0)),
            zorder=1,
        )
        ax.text(
            -0.055,
            value + 0.45,
            label,
            color=reference_color,
            fontsize=6.5,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4},
        )

    for method, pure, rtsm in zip(METHODS, PURE_SFDA, RTSM):
        y = np.array([pure, rtsm])
        ax.plot(
            x,
            y,
            color=colors[method],
            marker=markers[method],
            label=method,
            zorder=3,
        )
        ax.text(
            1.035,
            rtsm + label_offsets[method],
            f"+{rtsm - pure:.3f}",
            color=colors[method],
            fontsize=7.2,
            va="center",
            ha="left",
        )

    ax.set_xlim(-0.08, 1.36)
    ax.set_ylim(30.0, 64.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["Pure\nSFDA", "RTSM\n(+5% labels)"])
    ax.set_yticks([30, 40, 50, 60])
    ax.set_ylabel("AP50")
    ax.tick_params(axis="both", length=2.5, pad=2)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, 1.28),
        ncol=4,
        handlelength=1.2,
        columnspacing=0.7,
        handletextpad=0.35,
        frameon=False,
    )

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#444444")
        ax.spines[spine].set_linewidth(0.8)

    pdf_path = args.output_dir / "fig_rtsm_anchor_line.pdf"
    png_path = args.output_dir / "fig_rtsm_anchor_line.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
