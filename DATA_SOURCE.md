# Pulling real option data (Deribit)

The Anthropic sandbox blocks all outbound hosts, so live data must be pulled on
YOUR network. Deribit is chosen because it is the easiest global source:
fully public REST API, no key, no cookies, deep BTC/ETH option chains, exchange
mark IV included, and a real rough-volatility literature to cite.

## One command

```bash
cd rv_pinn_pipeline
python data_deribit.py --currency BTC --out data/deribit_chain.csv
```

That writes a few hundred rows to `data/deribit_chain.csv` in the pipeline's
common schema. Then run everything:

```bash
python run_all.py --csv data/deribit_chain.csv --pinn-iters 6000
```

Or pull live and run in one go:

```bash
python run_all.py --online --currency BTC
```

## If you'd rather not run code

Open this URL in any browser and save the JSON, or paste it back to me:

```
https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option
```

Each element has: `instrument_name` (e.g. `BTC-27DEC24-60000-C`),
`underlying_price`, `mark_price`, `bid_price`, `ask_price`, `mark_iv`,
`volume`. The parser in `data_deribit.py` turns that into the pipeline schema.

## Verify the parser without network

```bash
python data_deribit.py --offline-test      # parses an embedded real-format sample
```

## Notes on Deribit quirks (already handled in code)

- Option prices are quoted in COINS (fraction of the underlying). The fetcher
  multiplies mark/bid/ask by the underlying price to get absolute USD, which is
  what Black-Scholes / calibration expect.
- Expiries are 08:00 UTC on the listed date; T is computed from now in UTC.
- `mark_iv` is in percent; the fetcher divides by 100.
- Use r=0 for crypto (no clean risk-free/carry); the forward is taken from the
  quotes via put-call structure. Pass a funding-implied rate if you have one.
- ETH works identically: `--currency ETH`.

## Other easy sources (if you want equities instead)

- **yfinance** (`pip install yfinance`): `Ticker("SPY").option_chain(expiry)`
  gives calls/puts with IV. Free, no key, but IVs are noisier and less liquid
  than Deribit's; adapt `data_deribit.py`'s output columns.
- **OptionMetrics** (via a university subscription): gold standard for SPX,
  clean IVs; drop a CSV with the common columns and skip the fetcher entirely.
