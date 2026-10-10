"""Download detailed CPI components and average prices from the BLS flat files.

FRED carries only a handful of CPI components, so the detailed categories
(about 160 of them, from "Food and beverages" down to "Eggs" or "College
tuition and fees") come straight from the Bureau of Labor Statistics:

    https://download.bls.gov/pub/time.series/cu/   CPI-U index levels
    https://download.bls.gov/pub/time.series/ap/   average retail prices

Which categories to keep, their names, their place in the hierarchy and their
weights (relative importance) are listed in data/cpi_weights.csv.

Outputs
    data/cpi_components.csv  monthly index level per category, not seasonally
                             adjusted (year-over-year changes need no seasonal
                             adjustment), one column per BLS item code, since 1979
    data/avg_prices.csv      monthly U.S. city average price per item, since 1979

BLS rejects anonymous scripts, so requests carry a descriptive User-Agent.
Set the BLS_USER_AGENT environment variable (for example to "name email") to
override it. If a download fails, the previous file is kept unchanged.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WEIGHTS_PATH = DATA_DIR / "cpi_weights.csv"
COMPONENTS_PATH = DATA_DIR / "cpi_components.csv"
PRICES_PATH = DATA_DIR / "avg_prices.csv"

BASE = "https://download.bls.gov/pub/time.series/"
START = "1979-01-01"  # one year before 1980 so year-over-year rates start in 1980

# CPI-U, U.S. city average files that together hold every category in cpi_weights.csv
CU_FILES = [
    "cu.data.1.AllItems",
    "cu.data.11.USFoodBeverage",
    "cu.data.12.USHousing",
    "cu.data.13.USApparel",
    "cu.data.14.USTransportation",
    "cu.data.15.USMedical",
    "cu.data.16.USRecreation",
    "cu.data.17.USEducationAndCommunication",
    "cu.data.18.USOtherGoodsAndServices",
    "cu.data.20.USCommoditiesServicesSpecial",
]
AP_FILES = ["ap.data.0.Current"]  # every current average-price series, full history

# BLS average-price item code -> column name. Labels and units live in index.html.
AVG_PRICES = {
    "708111": "eggs",
    "709112": "milk",
    "702111": "bread",
    "717311": "coffee",
    "703112": "ground_beef",
    "703613": "steak",
    "704111": "bacon",
    "706111": "chicken",
    "710212": "cheese",
    "710411": "ice_cream",
    "711211": "bananas",
    "711311": "oranges",
    "712112": "potatoes",
    "712311": "tomatoes",
    "715211": "sugar",
    "701111": "flour",
    "701312": "rice",
    "718311": "potato_chips",
    "713111": "orange_juice",
    "720111": "beer",
    "74714": "gasoline",
    "74717": "diesel",
    "72610": "electricity",
    "72620": "natural_gas",
}

USER_AGENT = (
    os.environ.get("BLS_USER_AGENT")
    or "us-inflation-forecast research dashboard (https://github.com/ppc1004/us-inflation-forecast)"
)


def fetch_text(name: str, retries: int = 3) -> str:
    url = BASE + name.split(".")[0] + "/" + name
    headers = {"User-Agent": USER_AGENT, "Accept": "text/plain"}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=120)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            if attempt == retries - 1:
                raise
            print(f"  {name}: {error}, retrying")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def parse_bls(text: str, prefix: str) -> pd.DataFrame:
    """Parse a BLS time-series flat file into long form (series_id, date, value).

    Only monthly observations (periods M01..M12) of series that start with
    `prefix` are kept; annual averages (M13) and semiannual periods are dropped.
    """
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["series_id"] = df["series_id"].str.strip()
    df["period"] = df["period"].str.strip()
    df = df[df["series_id"].str.startswith(prefix) & df["period"].str.match(r"^M(0[1-9]|1[0-2])$")]
    df = df.assign(
        date=pd.to_datetime(df["year"].str.strip() + "-" + df["period"].str[1:] + "-01"),
        value=pd.to_numeric(df["value"].str.strip(), errors="coerce"),
    )
    return df[["series_id", "date", "value"]].dropna()


def to_wide(long: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Pivot long BLS data to one column per requested series, monthly since START."""
    long = long[long["series_id"].isin(columns)]
    wide = long.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
    wide = wide.rename(columns=columns)
    ordered = [c for c in columns.values() if c in wide.columns]
    wide = wide[ordered].sort_index()
    wide = wide[wide.index >= START]
    # one row per calendar month, so a month BLS skipped (October 2025) stays visible as a gap
    if len(wide):
        wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="MS"))
    wide.index.name = "date"
    return wide


def download_components() -> None:
    codes = pd.read_csv(WEIGHTS_PATH)["code"].tolist()
    wanted = {f"CUUR0000{code}": code for code in codes}

    frames = []
    for name in CU_FILES:
        frames.append(parse_bls(fetch_text(name), "CUUR0000"))
        print(f"  {name}: ok")
        time.sleep(1)

    wide = to_wide(pd.concat(frames, ignore_index=True), wanted)
    missing = sorted(set(codes) - set(wide.columns))
    if missing:
        print(f"  no BLS data for: {', '.join(missing)}")
    if "SA0" not in wide.columns or len(wide.columns) < len(codes) * 0.8:
        raise RuntimeError("CPI component download looks incomplete; keeping the previous file.")

    wide.round(3).to_csv(COMPONENTS_PATH, date_format="%Y-%m-%d")
    print(f"Saved {len(wide)} months x {len(wide.columns)} categories to {COMPONENTS_PATH}")


def download_prices() -> None:
    wanted = {f"APU0000{item}": col for item, col in AVG_PRICES.items()}
    frames = []
    for name in AP_FILES:
        frames.append(parse_bls(fetch_text(name), "APU0000"))
        print(f"  {name}: ok")
        time.sleep(1)

    wide = to_wide(pd.concat(frames, ignore_index=True), wanted)
    if len(wide.columns) < len(AVG_PRICES) * 0.7:
        raise RuntimeError("Average-price download looks incomplete; keeping the previous file.")

    wide.round(3).to_csv(PRICES_PATH, date_format="%Y-%m-%d")
    print(f"Saved {len(wide)} months x {len(wide.columns)} items to {PRICES_PATH}")


def main() -> None:
    failures = 0
    for step in (download_components, download_prices):
        try:
            step()
        except Exception as error:  # keep the previous file; one failure must not stop the other
            failures += 1
            print(f"{step.__name__} failed: {error}")
    if failures == 2:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
