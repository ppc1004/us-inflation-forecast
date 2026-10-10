"""Parsing of the BLS and BEA flat files, on small inline samples (no network)."""

import pandas as pd
import pytest

from data_utils import load_level
from download_bls import parse_bls, to_wide
from download_pce import parse_nipa, spending_weights

BLS_SAMPLE = """series_id        \tyear\tperiod\t       value\tfootnote_codes
CUUR0000SEFH     \t2024\tM12\t      300.0\t
CUUR0000SEFH     \t2024\tM13\t      280.0\t
CUUR0000SEFH     \t2025\tM01\t      310.5\t
CUSR0000SEFH     \t2025\tM01\t      999.0\t
CUUR0000SA0      \t2025\tM01\t      317.7\t
CUUR0000SA0      \t2025\tS01\t      316.0\t
"""

NIPA_SAMPLE = """%SeriesCode,Period,Value
DPCERC,2026M07,"20,000"
DPCERC,2026M08,"21,000"
DHLCRC,2026M08,"3,150"
DHLCRG,2026M08,"125.148"
DPCERG,2026M08,"131.579"
DPCERG,2026Q2,"130.000"
"""


def test_parse_bls_keeps_monthly_rows_of_prefix():
    long = parse_bls(BLS_SAMPLE, "CUUR0000")
    assert set(long["series_id"]) == {"CUUR0000SEFH", "CUUR0000SA0"}
    assert len(long) == 3  # M13 (annual) and S01 (semiannual) dropped, SA series dropped
    assert long["value"].max() == pytest.approx(317.7)


def test_to_wide_renames_and_filters_start():
    long = parse_bls(BLS_SAMPLE, "CUUR0000")
    wide = to_wide(long, {"CUUR0000SA0": "SA0", "CUUR0000SEFH": "SEFH"})
    assert list(wide.columns) == ["SA0", "SEFH"]
    assert wide.loc["2025-01-01", "SEFH"] == pytest.approx(310.5)


def test_parse_nipa_and_weights():
    wide = parse_nipa(NIPA_SAMPLE, {"DPCERC", "DHLCRC", "DHLCRG", "DPCERG"})
    assert wide.loc["2026-08-01", "DPCERC"] == pytest.approx(21000)
    assert wide.loc["2026-08-01", "DHLCRG"] == pytest.approx(125.148)
    assert len(wide) == 2  # quarterly row ignored

    tree = pd.DataFrame({"code": ["DPCE", "DHLC"], "name": ["PCE", "Health"], "parent": [None, "DPCE"]})
    weights = spending_weights(tree, wide[["DPCERC", "DHLCRC"]]).set_index("code")
    assert weights.loc["DPCE", "weight"] == pytest.approx(100)
    assert weights.loc["DHLC", "weight"] == pytest.approx(15)


def test_load_level_uses_last_unbroken_stretch(tmp_path):
    path = tmp_path / "rates.csv"
    dates = pd.date_range("2001-01-01", periods=8, freq="MS")
    values = [5.0, 5.1, None, None, 4.8, 4.7, 4.9, None]
    pd.DataFrame({"date": dates, "gs30": values}).to_csv(path, index=False)
    level = load_level(path, "gs30")
    assert list(level.index) == list(dates[4:7])
    assert level.iloc[-1] == pytest.approx(4.9)
