"""12-month forecasts of the federal funds rate and 2-, 10- and 30-year Treasury yields.

Interest rates behave close to a random walk: next month's yield is usually
very near this month's. Each series is therefore modelled in first differences
(ARIMA(p, 1, q)), with p and q chosen by AIC, on monthly averages since 1990.
The "no change" (random walk) forecast is the benchmark to beat.

For every series the script writes a 12-month forecast with 95% intervals and
a backtest: over the last 48 months the model is re-fit on data available at
the time and scored 1-12 months ahead against "no change".

Outputs
    data/rates_forecast.csv  series, model, date, mean, lower, upper
    data/rates_backtest.csv  series, horizon, n, model_mae, model_rmse, naive_mae, naive_rmse, coverage_95
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from data_utils import load_level
from horizon_backtest import HORIZON, score_origin, summarise

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "extra_series.csv"
FORECAST_PATH = DATA_DIR / "rates_forecast.csv"
BACKTEST_PATH = DATA_DIR / "rates_backtest.csv"

SERIES = {"fed_funds": "Fed funds", "gs2": "2-year", "gs10": "10-year", "gs30": "30-year"}
START = "1990-01-01"
N_ORIGINS = 48
CANDIDATES = [(p, 1, q) for p, q in itertools.product(range(3), range(3))]


def select_order(level: pd.Series) -> tuple[int, int, int]:
    best, best_aic = (0, 1, 1), float("inf")
    for order in CANDIDATES:
        try:
            aic = ARIMA(level, order=order).fit().aic
        except Exception:
            continue
        if aic < best_aic:
            best, best_aic = order, aic
    return best


def fit_forecast(level: pd.Series, order: tuple[int, int, int]) -> pd.DataFrame:
    frame = ARIMA(level, order=order).fit().get_forecast(steps=HORIZON).summary_frame(alpha=0.05)
    # Nominal policy rates and yields stay at or above zero in the U.S. data; clip the lower band.
    frame["mean_ci_lower"] = frame["mean_ci_lower"].clip(lower=0)
    return frame


def run() -> None:
    warnings.filterwarnings("ignore")
    forecasts, backtests = [], []

    for column, label in SERIES.items():
        level = load_level(DATA_PATH, column, START).dropna()
        if len(level) < N_ORIGINS + 60:
            print(f"{label}: not enough data, skipped")
            continue
        order = select_order(level)
        model = f"ARIMA{order}".replace(" ", "")

        frame = fit_forecast(level, order)
        forecasts.append(
            pd.DataFrame(
                {
                    "series": column,
                    "model": model,
                    "date": frame.index,
                    "mean": frame["mean"].to_numpy(),
                    "lower": frame["mean_ci_lower"].to_numpy(),
                    "upper": frame["mean_ci_upper"].to_numpy(),
                }
            )
        )

        records = []
        for position in range(len(level) - N_ORIGINS, len(level)):
            records.extend(score_origin(level, position, fit_forecast(level.iloc[:position], order)))
        backtests.append(summarise(records).assign(series=column))

        print(f"{label}: {model}, latest {level.iloc[-1]:.2f}%, in 12 months {frame['mean'].iloc[-1]:.2f}%")

    if not forecasts:
        raise SystemExit("No rate series available.")

    cols = ["series", "model", "date", "mean", "lower", "upper"]
    pd.concat(forecasts)[cols].round(4).to_csv(FORECAST_PATH, index=False, date_format="%Y-%m-%d")
    bt = pd.concat(backtests)
    bt[["series"] + [c for c in bt.columns if c != "series"]].to_csv(BACKTEST_PATH, index=False)
    print(f"Saved {FORECAST_PATH.name} and {BACKTEST_PATH.name}")


if __name__ == "__main__":
    run()
