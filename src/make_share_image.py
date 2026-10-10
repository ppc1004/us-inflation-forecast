"""Render the 1200x630 social preview card (data/social_preview.png).

This is the image shown when the site link is shared on LinkedIn, Slack,
LINE, etc. It is regenerated with every data update so the numbers are current.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_utils import load_yoy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CPI_PATH = DATA_DIR / "cpi_data.csv"
FORECAST_PATH = DATA_DIR / "yoy_forecast.csv"
OUTPUT_PATH = DATA_DIR / "social_preview.png"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#ebeae4"
PAGE = "#f7f7f5"
ACTUAL = "#2a78d6"
MODEL = "#eb6834"
FED_TARGET = 2.0


def make_share_image() -> None:
    yoy = load_yoy(CPI_PATH)
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"]).set_index("date")

    history = yoy.iloc[-36:]
    latest_date, latest = yoy.index[-1], yoy.iloc[-1]
    gap = latest - FED_TARGET
    next_row = forecast.iloc[0]

    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PAGE)

    # ---- text block ----
    fig.text(0.055, 0.86, "U.S. ECONOMIC FORECAST", fontsize=15, color=INK_2, weight="bold")
    fig.text(0.055, 0.60, f"{latest:.1f}%", fontsize=96, color=INK, weight="bold")
    fig.text(0.055, 0.52, f"Annual CPI inflation, {latest_date:%B %Y}", fontsize=19, color=INK_2)
    direction = "above" if gap >= 0 else "below"
    fig.text(0.055, 0.465, f"{abs(gap):.1f} points {direction} the Fed's 2% target", fontsize=19, color=INK_2)
    fig.text(
        0.055,
        0.33,
        f"Forecast for {next_row.name:%b %Y}: {next_row['mean']:.1f}%",
        fontsize=21,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.055,
        0.275,
        f"95% interval {next_row['mean_ci_lower']:.1f}% to {next_row['mean_ci_upper']:.1f}%",
        fontsize=17,
        color=INK_2,
    )
    fig.text(
        0.055,
        0.09,
        "ARIMA model  ·  rolling backtest  ·  updated with every CPI release",
        fontsize=14,
        color=MUTED,
    )

    # ---- chart ----
    ax = fig.add_axes([0.55, 0.17, 0.40, 0.66], facecolor=PAGE)
    fc = forecast.iloc[:12]
    bridge_x = [history.index[-1]] + list(fc.index)
    ax.fill_between(
        bridge_x,
        [latest] + list(fc["mean_ci_lower"]),
        [latest] + list(fc["mean_ci_upper"]),
        color=MODEL,
        alpha=0.13,
        linewidth=0,
    )
    ax.plot(history.index, history.values, color=ACTUAL, linewidth=3, solid_capstyle="round")
    ax.plot(bridge_x, [latest] + list(fc["mean"]), color=MODEL, linewidth=3, linestyle=(0, (4, 3)))
    ax.axhline(FED_TARGET, color=MUTED, linewidth=1.2, linestyle=(0, (3, 3)))
    ax.scatter([latest_date], [latest], s=90, color=ACTUAL, edgecolor="white", linewidth=2.5, zorder=5)

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=MUTED, labelsize=12, length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    fig.savefig(OUTPUT_PATH, facecolor=PAGE)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    make_share_image()
