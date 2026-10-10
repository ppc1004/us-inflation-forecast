"""Shared data preparation for every model script."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def _monthly(path: str | Path) -> pd.DataFrame:
    """Read a date-indexed CSV onto a complete monthly (MS) index."""
    return (
        pd.read_csv(path, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .asfreq("MS")  # one row per calendar month; a missing month becomes NaN
    )


def load_yoy(cpi_path: str | Path) -> pd.Series:
    """Return monthly year-over-year CPI inflation (%) as a Series with MS frequency.

    BLS did not publish October 2025 CPI (government shutdown). Missing months
    inside the series are filled by linear interpolation so that pct_change(12)
    always compares a month with the same calendar month one year earlier.
    """
    cpi = _monthly(cpi_path)["cpi"].interpolate(limit_area="inside")
    yoy = cpi.pct_change(12, fill_method=None) * 100
    return yoy.dropna().asfreq("MS")


def load_extra_yoy(extra_path: str | Path, columns: Iterable[str]) -> pd.DataFrame:
    """Year-over-year % change of selected columns in data/extra_series.csv."""
    data = _monthly(extra_path)
    present = [c for c in columns if c in data.columns]
    levels = data[present].interpolate(limit_area="inside")
    return levels.pct_change(12, fill_method=None) * 100


def load_column_yoy(path: str | Path, column: str) -> pd.Series:
    """Year-over-year % change of one index column (e.g. pce in data/pce_data.csv).

    Same treatment as load_yoy: a complete monthly calendar, interior gaps
    interpolated, so every rate compares the same calendar month a year apart.
    """
    level = _monthly(path)[column].interpolate(limit_area="inside")
    yoy = level.pct_change(12, fill_method=None) * 100
    return yoy.dropna().asfreq("MS")


def load_level(path: str | Path, column: str, start: str | None = None) -> pd.Series:
    """A level series (e.g. a Treasury yield) as its latest unbroken monthly stretch.

    Rates are modelled in levels. If the series has a publication gap (the
    30-year Treasury was discontinued from 2002 to 2006), only the observations
    after the last gap are returned, so the model never bridges a break.
    """
    series = _monthly(path)[column]
    if start is not None:
        series = series[series.index >= start]
    last_valid = series.last_valid_index()
    if last_valid is None:
        return series.dropna()
    series = series.loc[:last_valid]
    gaps = series.index[series.isna()]
    if len(gaps):
        series = series[series.index > gaps.max()]
    return series.asfreq("MS")
