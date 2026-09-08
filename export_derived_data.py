r"""
Export the two robustness datasets in DERIVED form for the public repository.

The manuscript's data section promises the weekly implied-vol surfaces and the
daily realised variance, not the raw trade tape or the raw five-minute bars.
This writes exactly those:

    data/hist_weekly_surfaces.csv   weekly (Wednesday) BTC surfaces, 2023-01 to
                                    2024-12, in the calibration schema, from
                                    hist_surface.py with its filters applied
    data/btc_perp_daily_rv.csv      daily realised variance and volatility of
                                    the Deribit BTC perpetual from 5-min bars,
                                    the series diagnose_physical_hurst.py uses

Both are small (a few MB) and are what the regime and physical-H scripts need
to be reproduced without C:\Data.
"""
import os
import numpy as np, pandas as pd

from hist_surface import load_panel, surface_for_date, trading_dates
from diagnose_physical_hurst import realised_vol

os.makedirs("data", exist_ok=True)

panel = load_panel()
rows = []
for d in trading_dates(panel, "2023-01-01", "2024-12-31"):
    s = surface_for_date(panel, d)
    if s is None:
        continue
    s = s.copy(); s.insert(0, "date", d.date().isoformat())
    rows.append(s)
surf = pd.concat(rows, ignore_index=True)
surf.to_csv("data/hist_weekly_surfaces.csv", index=False, float_format="%.6g")
print("weekly surfaces: %d weeks, %d points -> data/hist_weekly_surfaces.csv"
      % (surf["date"].nunique(), len(surf)))

sig = realised_vol()
rv = pd.DataFrame({"date": pd.to_datetime(sig.index).date if hasattr(sig, "index") else np.arange(len(sig)),
                   "realised_vol": np.asarray(sig, float)})
rv["realised_var"] = rv["realised_vol"] ** 2
rv.to_csv("data/btc_perp_daily_rv.csv", index=False, float_format="%.6g")
print("daily realised vol: %d days -> data/btc_perp_daily_rv.csv" % len(rv))
