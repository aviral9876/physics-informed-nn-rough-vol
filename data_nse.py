"""
Stage 1 - Data acquisition.
===========================

Two sources, one interface:

  fetch_nse_option_chain(symbol)   -> live NSE index option chain (NIFTY/BANKNIFTY)
  synthetic_option_chain(...)      -> realistic simulated chain (works everywhere)

WHY THIS DESIGN
---------------
NSE has no clean public options API. The undocumented endpoint
`/api/option-chain-indices?symbol=NIFTY` returns JSON, but ONLY if you first
hit the homepage to obtain session cookies and send browser-like headers;
otherwise it returns HTTP 401/403. Datacenter / cloud IPs are frequently
blocked outright. The fetcher below performs the cookie handshake correctly,
so it works from a normal machine/notebook. In sandboxed/blocked environments
we fall back to `synthetic_option_chain`, which produces a chain with a
realistic volatility smile/skew so every downstream stage still runs.

Both return a tidy pandas DataFrame with a common schema:
    ['symbol','expiry','T','type','strike','spot','ltp','bid','ask','iv_nse']
where T is year-fraction to expiry and type in {'C','P'}.
"""

import json
import time
import datetime as dt
import numpy as np
import pandas as pd

NSE_HOME = "https://www.nseindia.com"
NSE_OC = "https://www.nseindia.com/api/option-chain-indices?symbol={sym}"

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive",
}


def fetch_nse_option_chain(symbol="NIFTY", pause=1.0, timeout=15):
    """
    Live NSE index option chain. Requires the `requests` package and a
    non-blocked IP. Raises RuntimeError with a clear message if NSE refuses,
    so the caller can fall back to synthetic data.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("`requests` not installed: pip install requests") from e

    s = requests.Session()
    s.headers.update(_HEADERS)
    # 1) prime cookies by visiting the homepage and the option-chain page
    try:
        s.get(NSE_HOME, timeout=timeout)
        time.sleep(pause)
        s.get(NSE_HOME + "/option-chain", timeout=timeout)
        time.sleep(pause)
        # 2) hit the JSON API
        r = s.get(NSE_OC.format(sym=symbol), timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"NSE request failed: {e}") from e

    if r.status_code != 200:
        raise RuntimeError(f"NSE returned HTTP {r.status_code} "
                           f"(often IP-blocked; use synthetic fallback).")
    payload = r.json()
    return _parse_nse_json(payload, symbol)


def _parse_nse_json(payload, symbol):
    """Flatten NSE's nested option-chain JSON into the common schema."""
    records = payload["records"]
    spot = float(records["underlyingValue"])
    today = dt.date.today()
    rows = []
    for item in records["data"]:
        strike = float(item["strikePrice"])
        expiry_str = item["expiryDate"]           # e.g. '30-Jan-2025'
        expiry = dt.datetime.strptime(expiry_str, "%d-%b-%Y").date()
        T = max((expiry - today).days, 0) / 365.0
        for typ, key in [("C", "CE"), ("P", "PE")]:
            if key in item:
                leg = item[key]
                rows.append(dict(
                    symbol=symbol, expiry=expiry, T=T, type=typ,
                    strike=strike, spot=spot,
                    ltp=float(leg.get("lastPrice", np.nan)),
                    bid=float(leg.get("bidprice", np.nan)),
                    ask=float(leg.get("askPrice", np.nan)),
                    iv_nse=float(leg.get("impliedVolatility", np.nan)) / 100.0,
                ))
    df = pd.DataFrame(rows)
    return df[df["T"] > 0].reset_index(drop=True)


def synthetic_option_chain(spot=22000.0, r=0.065, symbol="NIFTY_SYNTH",
                           maturities_days=(7, 30, 60, 90, 180),
                           n_strikes=15, seed=0):
    """
    Realistic synthetic index option chain with a rough-vol-like smile.

    We generate an implied-vol surface with (i) a downward skew (equity
    leverage effect), (ii) a smile curvature, and (iii) a short-maturity skew
    that steepens as T -> 0 (the hallmark of rough volatility), then invert to
    Black-Scholes prices and add small bid/ask noise. This lets the whole
    calibration + PINN pipeline run without any network access, while looking
    like a genuine NIFTY chain.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    today = dt.date.today()

    def bs_price(S, K, T, sig, r, typ):
        d1 = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        if typ == "C":
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    # smile parametrisation in log-moneyness k = log(K/F)
    def iv_surface(k, T):
        atm = 0.13 + 0.02*np.exp(-3*T)          # ATM term structure
        # skew steepens as T->0 like T^{H-1/2} with H~0.1  => ~ T^{-0.4}
        skew = -0.30 * (T + 0.02) ** (-0.4) * 0.06
        curv = 0.5 + 0.3*np.exp(-2*T)
        iv = atm + skew*k + curv*k**2
        return np.clip(iv, 0.03, 1.0)

    rows = []
    for d in maturities_days:
        T = d / 365.0
        F = spot * np.exp(r*T)
        # strikes on a log-moneyness grid, rounded to NIFTY's 50-pt steps
        kk = np.linspace(-0.18, 0.18, n_strikes)
        strikes = np.round(F*np.exp(kk) / 50.0) * 50.0
        strikes = np.unique(strikes)
        for K in strikes:
            k = np.log(K / F)
            sig = float(iv_surface(k, T))
            for typ in ("C", "P"):
                px = bs_price(spot, K, T, sig, r, typ)
                spread = max(0.5, 0.01*px)       # realistic-ish spread
                bid = max(px - spread + rng.normal(0, 0.05*spread), 0.05)
                ask = px + spread + rng.normal(0, 0.05*spread)
                rows.append(dict(
                    symbol=symbol, expiry=today + dt.timedelta(days=d),
                    T=T, type=typ, strike=float(K), spot=spot,
                    ltp=float(px), bid=float(bid), ask=float(ask),
                    iv_nse=sig))
    return pd.DataFrame(rows).reset_index(drop=True)


def load_chain(symbol="NIFTY", allow_network=True, **kw):
    """
    Convenience loader: try live NSE, fall back to synthetic on any failure.
    Returns (df, source_str).
    """
    if allow_network:
        try:
            df = fetch_nse_option_chain(symbol)
            return df, "nse_live"
        except Exception as e:
            print(f"[data] NSE fetch failed ({e}). Using synthetic fallback.")
    df = synthetic_option_chain(**kw)
    return df, "synthetic"


if __name__ == "__main__":
    df, src = load_chain("NIFTY", allow_network=True)
    print("source:", src, "| rows:", len(df))
    print(df.head(8).to_string(index=False))
    df.to_csv("data/option_chain.csv", index=False)
    print("saved -> data/option_chain.csv")
