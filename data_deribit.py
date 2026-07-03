"""
Stage 1 (Deribit) - Global, easy-to-pull option data.
======================================================

Deribit exposes a fully public REST API: no API key, no cookie handshake, no
scraping tricks. BTC and ETH option chains are deep and liquid, and Deribit
publishes its own mark implied vol per instrument. Crypto also has a genuine
rough-volatility literature, so this is a defensible data source for the thesis
rather than just a convenience.

Endpoints used (all public GET, https://www.deribit.com):
  /api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option
      -> one row per option instrument: mark price, mark IV, bid/ask, volume,
         underlying index price.
  /api/v2/public/get_instruments?currency=BTC&kind=option&expired=false
      -> instrument metadata: strike, expiry timestamp, option type.

Instrument names look like:  BTC-27DEC24-60000-C
                             ^cur ^expiry   ^strike ^type(C/P)

This module produces the SAME tidy schema as the rest of the pipeline:
    ['symbol','expiry','T','type','strike','spot','ltp','bid','ask','iv_mkt']
so surface.py / calibrate.py / pinn.py consume it with zero changes.

RUN ON YOUR OWN NETWORK (the Anthropic sandbox blocks all outbound hosts):
    python data_deribit.py --currency BTC --out data/deribit_chain.csv
"""

import argparse
import datetime as dt
import json
import time
import numpy as np
import pandas as pd

BASE = "https://www.deribit.com/api/v2/public/"


def _get(endpoint, params, timeout=15, retries=3):
    """Minimal dependency-light GET returning parsed JSON 'result'."""
    import urllib.request
    import urllib.parse
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)["result"]
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"Deribit GET failed for {endpoint}: {last}")


def _parse_instrument_name(name):
    """
    'BTC-27DEC24-60000-C' -> (expiry_date, strike, 'C').
    Deribit expiries are 08:00 UTC on the listed day.
    """
    parts = name.split("-")
    cur, exp_s, strike_s, cp = parts[0], parts[1], parts[2], parts[3]
    expiry = dt.datetime.strptime(exp_s, "%d%b%y").replace(
        hour=8, tzinfo=dt.timezone.utc)
    return expiry, float(strike_s), ("C" if cp.upper().startswith("C") else "P")


def fetch_deribit_chain(currency="BTC"):
    """
    Pull the full live option chain for BTC or ETH and return the common-schema
    DataFrame. Uses only the book summary (which already contains mark IV,
    bid/ask, and the underlying index price), and derives strike/expiry/type
    from the instrument name.
    """
    summary = _get("get_book_summary_by_currency",
                   {"currency": currency, "kind": "option"})
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for it in summary:
        name = it.get("instrument_name", "")
        if not name or name.count("-") != 3:
            continue
        try:
            expiry, strike, cp = _parse_instrument_name(name)
        except Exception:
            continue
        T = (expiry - now).total_seconds() / (365.0 * 24 * 3600)
        if T <= 0:
            continue
        spot = it.get("underlying_price") or it.get("index_price")
        if spot is None:
            continue
        # Deribit option prices are quoted in COINS (fraction of underlying);
        # convert mark/bid/ask to absolute currency terms (x spot) for BS work.
        mark = it.get("mark_price")
        bid = it.get("bid_price")
        ask = it.get("ask_price")
        iv = it.get("mark_iv")  # already in percent
        rows.append(dict(
            symbol=f"{currency}",
            expiry=expiry.date(),
            T=T, type=cp, strike=float(strike), spot=float(spot),
            ltp=(mark*spot if mark is not None else np.nan),
            bid=(bid*spot if bid else np.nan),
            ask=(ask*spot if ask else np.nan),
            iv_mkt=(iv/100.0 if iv is not None else np.nan),
            volume=it.get("volume", 0.0),
        ))
    df = pd.DataFrame(rows)
    # keep only rows with a usable price or a usable mark IV
    df = df[(df["ltp"] > 0) | df["iv_mkt"].notna()].reset_index(drop=True)
    return df


def load_chain(currency="BTC", allow_network=True, csv_path=None):
    """
    Loader mirroring data_nse.load_chain's interface.
    - If csv_path is given, read a previously-saved chain (works offline).
    - Else try the live Deribit pull.
    Returns (df, source_str).
    """
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["expiry"])
        return df, f"csv:{csv_path}"
    if allow_network:
        df = fetch_deribit_chain(currency)
        return df, "deribit_live"
    raise RuntimeError("No network and no csv_path given.")


# --- tiny embedded sample of Deribit's real response shape, for offline
#     testing of the PARSER only (not real-time prices) ---
_SAMPLE_SUMMARY = [
    {"instrument_name": "BTC-27DEC24-60000-C", "underlying_price": 61250.0,
     "index_price": 61230.0, "mark_price": 0.085, "bid_price": 0.083,
     "ask_price": 0.087, "mark_iv": 55.2, "volume": 12.3},
    {"instrument_name": "BTC-27DEC24-60000-P", "underlying_price": 61250.0,
     "index_price": 61230.0, "mark_price": 0.070, "bid_price": 0.068,
     "ask_price": 0.072, "mark_iv": 56.1, "volume": 8.0},
    {"instrument_name": "BTC-27DEC24-65000-C", "underlying_price": 61250.0,
     "index_price": 61230.0, "mark_price": 0.052, "bid_price": 0.050,
     "ask_price": 0.054, "mark_iv": 58.9, "volume": 20.1},
    {"instrument_name": "BTC-INDEX-thing", "mark_price": 1.0},   # malformed -> skipped
]


def _test_parser_offline():
    """Verify name parsing + schema mapping without any network."""
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for it in _SAMPLE_SUMMARY:
        name = it.get("instrument_name", "")
        if name.count("-") != 3:
            continue
        expiry, strike, cp = _parse_instrument_name(name)
        T = (expiry - now).total_seconds() / (365.0*24*3600)
        spot = it["underlying_price"]
        rows.append(dict(symbol="BTC", expiry=expiry.date(), T=T, type=cp,
                         strike=strike, spot=spot,
                         ltp=it["mark_price"]*spot,
                         bid=it["bid_price"]*spot, ask=it["ask_price"]*spot,
                         iv_mkt=it["mark_iv"]/100.0))
    df = pd.DataFrame(rows)
    assert list(df["type"]) == ["C", "P", "C"], df["type"].tolist()
    assert (df["strike"] == [60000, 60000, 65000]).all()
    assert abs(df["ltp"].iloc[0] - 0.085*61250) < 1e-6
    assert (df["iv_mkt"] > 0).all()
    print("[offline parser test] PASS")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC", choices=["BTC", "ETH"])
    ap.add_argument("--out", default="data/deribit_chain.csv")
    ap.add_argument("--offline-test", action="store_true",
                    help="test the parser on an embedded sample (no network)")
    args = ap.parse_args()

    if args.offline_test:
        _test_parser_offline()
    else:
        df = fetch_deribit_chain(args.currency)
        print(f"Fetched {len(df)} live {args.currency} option rows.")
        print(df.head(8).to_string(index=False))
        import os
        os.makedirs("data", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"saved -> {args.out}")
