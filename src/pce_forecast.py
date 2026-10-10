"""12-month forecasts of PCE inflation (headline and core) with a horizon backtest.

PCE is the Federal Reserve's preferred inflation measure; its 2% target is
defined in terms of headline PCE. The method mirrors the CPI forecast:

1. year-over-year inflation from the price index (data/pce_data.csv),
2. an ARIMA order chosen by AIC from a small candidate set,
3. a 12-month forecast with 95% intervals,
4. a backtest over the last 48 forecast origins, each re-fit only on data
   available at the time and scored 1-12 months ahead against a "no change"
   forecast, with 95%-interval coverage.

Outputs
    data/pce_forecast.csv          series, model, date, mean, lower, upper
    data/pce_horizon_backtest.csv  series, horizon, n, model_mae, ..., coverage_95
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from data_utils import load_column_yoy
from horizon_backtest import HORIZON, score_origin, summarise

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "pce_data.csv"
FORECAST_PATH = DATA_DIR / "pce_forecast.csv"
BACKTEST_PATH = DATA_DIR / "pce_horizon_backtest.csv"

SERIES = {"pce": "PCE", "core_pce": "Core PCE"}
CANDIDATES = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (1, 0, 2), (2, 0, 1), (2, 0, 2)]
N_ORIGINS = 48


def select_order(yoy: pd.Series) -> tuple[int, int, int]:
    """Lowest-AIC ARIMA order among CANDIDATES (falls back to (1, 0, 2))."""
    best, best_aic = (1, 0, 2), float("inf")
    for order in CANDIDATES:
        try:
            aic = ARIMA(yoy, order=order).fit().aic
        except Exception:
            continue
        if aic < best_aic:
            best, best_aic = order, aic
    return best


def forecast_frame(yoy: pd.Series, order: tuple[int, int, int]) -> pd.DataFrame:
    frame = ARIMA(yoy, order=order).fit().get_forecast(steps=HORIZON).summary_frame(alpha=0.05)
    return pd.DataFrame(
        {
            "date": frame.index,
            "mean": frame["mean"].to_numpy(),
            "lower": frame["mean_ci_lower"].to_numpy(),
            "upper": frame["mean_ci_upper"].to_numpy(),
        }
    )


def backtest(yoy: pd.Series, order: tuple[int, int, int]) -> pd.DataFrame:
    records = []
    for position in range(len(yoy) - N_ORIGINS, len(yoy)):
        fitted = ARIMA(yoy.iloc[:position], order=order).fit()
        frame = fitted.get_forecast(steps=HORIZON).summary_frame(alpha=0.05)
        records.extend(score_origin(yoy, position, frame))
    return summarise(records)


def run() -> None:
    warnings.filterwarnings("ignore")
    forecasts, backtests = [], []

    for column, label in SERIES.items():
        yoy = load_column_yoy(DATA_PATH, column)
        order = select_order(yoy)
        model = f"ARIMA{order}".replace(" ", "")

        fc = forecast_frame(yoy, order).assign(series=column, model=model)
        forecasts.append(fc)

        bt = backtest(yoy, order).assign(series=column)
        backtests.append(bt)
        print(f"{label}: {model}, latest {yoy.iloc[-1]:.2f}%, 12-month forecast {fc['mean'].iloc[-1]:.2f}%")

    cols = ["series", "model", "date", "mean", "lower", "upper"]
    pd.concat(forecasts)[cols].round(4).to_csv(FORECAST_PATH, index=False, date_format="%Y-%m-%d")
    bt_all = pd.concat(backtests)
    bt_all[["series"] + [c for c in bt_all.columns if c != "series"]].to_csv(BACKTEST_PATH, index=False)
    print(f"Saved {FORECAST_PATH.name} and {BACKTEST_PATH.name}")


if __name__ == "__main__":
    run()
