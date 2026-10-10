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
override it. If the flat files are blocked, the BLS JSON API is used instead
(see api_update). If every download fails, the previous file is kept unchanged.
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


def fetch_text(name: str, retries: int = 2) -> str:
    url = BASE + name.split(".")[0] + "/" + name
    headers = {"User-Agent": USER_AGENT, "Accept": "text/plain"}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=120)
            response.raise_for_status()
            return response.text
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code == 403:
                raise  # access denied (e.g. cloud IP ranges): retrying will not help
            if attempt == retries - 1:
                raise
            print(f"  {name}: {error}, retrying")
            time.sleep(5)
        except requests.RequestException as error:
            if attempt == retries - 1:
                raise
            print(f"  {name}: {error}, retrying")
            time.sleep(5)
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


# ---------------------------------------------------------------------------
# BLS Public Data API fallback
#
# download.bls.gov refuses some cloud networks (GitHub Actions runners get HTTP
# 403). The JSON API at api.bls.gov is then used instead. Without a key it
# allows 25 queries a day of 25 series and 10 years each; with a free key
# (repository secret BLS_API_KEY, https://data.bls.gov/registrationEngine/)
# 500 queries of 50 series and 20 years. Without a key a full history takes two
# daily runs: the most recent decades are fetched first and older ones are
# added on the next run, merging with what is already saved.
# ---------------------------------------------------------------------------

API_URL = "https://api.bls.gov/publicAPI/{}/timeseries/data/"  # v2 with a key, v1 without
API_KEY = os.environ.get("BLS_API_KEY") or None
FIRST_YEAR = int(START[:4])


class ApiBudget:
    def __init__(self) -> None:
        self.left = 450 if API_KEY else 24
        self.per_query = 50 if API_KEY else 25
        self.years = 20 if API_KEY else 10


def api_query(series_ids: list[str], start: int, end: int) -> pd.DataFrame:
    payload: dict = {"seriesid": series_ids, "startyear": str(start), "endyear": str(end)}
    if API_KEY:
        payload["registrationkey"] = API_KEY
    response = requests.post(
        API_URL.format("v2" if API_KEY else "v1"),
        json=payload,
        headers={"User-Agent": USER_AGENT},
        timeout=90,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API: {body.get('status')} {body.get('message')}")
    rows = []
    for series in body.get("Results", {}).get("series", []):
        for obs in series.get("data", []):
            period = obs.get("period", "")
            if len(period) == 3 and period.startswith("M") and period != "M13":
                rows.append((series["seriesID"], f"{obs['year']}-{period[1:]}-01", obs.get("value")))
    long = pd.DataFrame(rows, columns=["series_id", "date", "value"])
    long["date"] = pd.to_datetime(long["date"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna()


def existing_long(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    """The saved wide file back in long form, keyed by BLS series id."""
    if not path.exists():
        return pd.DataFrame(columns=["series_id", "date", "value"])
    wide = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    reverse = {col: sid for sid, col in columns.items()}
    long = wide.rename(columns=reverse).reset_index().melt(id_vars="date", var_name="series_id")
    return long.dropna()[["series_id", "date", "value"]]


def api_windows(have_from: int | None, this_year: int, span: int) -> list[tuple[int, int]]:
    """Year windows to request, newest first.

    With a saved file: refresh the last two years, then backfill anything older
    than what is saved. Without one: the whole history in `span`-year windows.
    """
    windows = []
    if have_from is not None:
        windows.append((max(this_year - 1, FIRST_YEAR), this_year))
        end = have_from - 1
    else:
        end = this_year
    while end >= FIRST_YEAR:
        start = max(FIRST_YEAR, end - span + 1)
        windows.append((start, end))
        end = start - 1
    return windows


def api_update(path: Path, columns: dict[str, str], budget: ApiBudget) -> pd.DataFrame:
    """Merge API data into the saved file: refresh recent years, then backfill history."""
    old = existing_long(path, columns)
    this_year = pd.Timestamp.today().year
    have_from = int(old["date"].min().year) if len(old) else None

    ids = list(columns)
    chunks = [ids[k : k + budget.per_query] for k in range(0, len(ids), budget.per_query)]
    frames = [old]
    for start, end in api_windows(have_from, this_year, budget.years):
        if budget.left < len(chunks):
            print("  BLS API daily budget used up; older history will be added on the next run.")
            break
        try:
            window = [api_query(chunk, start, end) for chunk in chunks]
        except Exception as error:
            print(f"  API {start}-{end} failed: {error}")
            break
        budget.left -= len(chunks)
        frames.extend(window)
        print(f"  API {start}-{end}: ok ({budget.left} queries left today)")
        time.sleep(0.5)
    if len(frames) == 1:
        raise RuntimeError("BLS API returned nothing new.")
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(["series_id", "date"], keep="last")
    return to_wide(merged, columns)


def flat_or_api(files: list[str], prefix: str, columns: dict[str, str], path: Path, budget: ApiBudget):
    try:
        frames = []
        for name in files:
            frames.append(parse_bls(fetch_text(name), prefix))
            print(f"  {name}: ok")
            time.sleep(1)
        return to_wide(pd.concat(frames, ignore_index=True), columns)
    except Exception as error:
        print(f"  flat files unavailable ({error}); using the BLS API")
        return api_update(path, columns, budget)


def download_components(budget: ApiBudget) -> None:
    codes = pd.read_csv(WEIGHTS_PATH)["code"].tolist()
    wanted = {f"CUUR0000{code}": code for code in codes}
    wide = flat_or_api(CU_FILES, "CUUR0000", wanted, COMPONENTS_PATH, budget)

    missing = sorted(set(codes) - set(wide.columns))
    if missing:
        print(f"  no BLS data for: {', '.join(missing)}")
    if "SA0" not in wide.columns or len(wide.columns) < len(codes) * 0.8:
        raise RuntimeError("CPI component download looks incomplete; keeping the previous file.")

    wide.round(3).to_csv(COMPONENTS_PATH, date_format="%Y-%m-%d")
    print(f"Saved {wide.index.min():%Y-%m}..{wide.index.max():%Y-%m} x {len(wide.columns)} categories")


def download_prices(budget: ApiBudget) -> None:
    wanted = {f"APU0000{item}": col for item, col in AVG_PRICES.items()}
    wide = flat_or_api(AP_FILES, "APU0000", wanted, PRICES_PATH, budget)
    if len(wide.columns) < len(AVG_PRICES) * 0.7:
        raise RuntimeError("Average-price download looks incomplete; keeping the previous file.")

    wide.round(3).to_csv(PRICES_PATH, date_format="%Y-%m-%d")
    print(f"Saved {wide.index.min():%Y-%m}..{wide.index.max():%Y-%m} x {len(wide.columns)} items")


def main() -> None:
    budget = ApiBudget()
    failures = 0
    for step in (download_components, download_prices):
        try:
            step(budget)
        except Exception as error:  # keep the previous file; one failure must not stop the other
            failures += 1
            print(f"{step.__name__} failed: {error}")
    if failures == 2:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
