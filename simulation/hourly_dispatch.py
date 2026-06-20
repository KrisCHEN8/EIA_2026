"""
Hourly Production Dispatch
==========================
Re-dispatches the daily production mix to hourly resolution.

The daily dispatch data gives us daily MWh per source. This module
distributes each daily total across 24 hours following the hourly
network demand profile shape, while respecting the priority order.

Priority order: Waste heat → Electricity → Biomass → Fossil fuel
(same as daily_new_rule.py but at hourly resolution)
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DH production mix"

PRIORITY_SOURCES = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]


def load_daily_dispatch(year: int = 2024) -> pd.DataFrame:
    """Load the daily priority dispatch data."""
    path = DATA_DIR / f"daily_priority_dispatch_{year}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def distribute_daily_to_hourly(
    daily_dispatch: pd.DataFrame,
    hourly_demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribute daily production mix to hourly resolution following
    the demand profile shape.

    Parameters
    ----------
    daily_dispatch : pd.DataFrame
        Daily dispatch with columns: date, Waste heat, Electricity, Biomass, Fossil fuel, daily_total_mwh.
    hourly_demand : pd.DataFrame
        Hourly demand with columns: date, network_load_mw (or similar).

    Returns
    -------
    pd.DataFrame with hourly production mix.
    """
    hourly = hourly_demand.copy()
    hourly["date_only"] = hourly["date"].dt.date

    # Merge daily totals
    daily = daily_dispatch.copy()
    daily["date_only"] = daily["date"].dt.date

    # For each day, distribute each source proportionally to hourly demand shape
    results = []

    for date_val, day_group in hourly.groupby("date_only"):
        daily_row = daily[daily["date_only"] == date_val]
        if daily_row.empty:
            continue
        daily_row = daily_row.iloc[0]

        n_hours = len(day_group)
        demand = day_group["network_load_mw"].values.copy()
        demand_sum = demand.sum()

        if demand_sum <= 0:
            weights = np.ones(n_hours) / n_hours
        else:
            weights = demand / demand_sum

        row_data = pd.DataFrame({"date": day_group["date"].values})
        row_data["hourly_demand_mw"] = demand.ravel()

        # Distribute each source by the hourly weight
        remaining_demand_mwh = demand * 1.0  # MW × 1h = MWh per hour

        for src in PRIORITY_SOURCES:
            daily_src_mwh = daily_row[src]
            # Distribute proportionally
            hourly_src = weights * daily_src_mwh
            # But cap at remaining demand per hour
            hourly_src = np.minimum(hourly_src, remaining_demand_mwh)
            row_data[src] = hourly_src
            remaining_demand_mwh = remaining_demand_mwh - hourly_src

        # Any leftover goes to fossil
        row_data["Fossil fuel"] = row_data["Fossil fuel"] + np.maximum(remaining_demand_mwh, 0)

        row_data["hourly_total_mwh"] = sum(row_data[src] for src in PRIORITY_SOURCES)

        results.append(row_data)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def redispatch_for_modified_load(
    daily_dispatch: pd.DataFrame,
    original_hourly_demand_mw: np.ndarray,
    modified_hourly_demand_mw: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Re-dispatch production for a modified (DSM) load profile.

    The key insight: when DSM reduces peak demand, the marginal source
    (fossil fuel, being last in priority) is what gets reduced. When DSM
    increases off-peak demand, base-load sources (waste heat) that were
    already running can serve more — or if waste heat capacity is maxed,
    electricity/biomass serve the extra.

    Parameters
    ----------
    daily_dispatch : pd.DataFrame
        Original daily dispatch.
    original_hourly_demand_mw : np.ndarray
        Original hourly demand (MW).
    modified_hourly_demand_mw : np.ndarray
        Modified hourly demand after DSM (MW).
    dates : pd.DatetimeIndex
        Timestamps for each hour.

    Returns
    -------
    pd.DataFrame with re-dispatched hourly production.
    """
    daily = daily_dispatch.copy()
    daily["date_only"] = daily["date"].dt.date

    date_only = dates.date
    results = []

    for date_val in np.unique(date_only):
        mask = date_only == date_val
        orig = original_hourly_demand_mw[mask]
        mod = modified_hourly_demand_mw[mask]
        hrs = dates[mask]

        daily_row = daily[daily["date_only"] == date_val]
        if daily_row.empty:
            continue
        daily_row = daily_row.iloc[0]

        n = len(orig)
        demand_change_mwh = mod - orig  # positive = more demand, negative = less

        # Original hourly dispatch (proportional)
        orig_sum = orig.sum()
        if orig_sum <= 0:
            weights = np.ones(n) / n
        else:
            weights = orig / orig_sum

        row_data = pd.DataFrame({"date": hrs})
        row_data["hourly_demand_original_mw"] = orig
        row_data["hourly_demand_modified_mw"] = mod

        # Start with original proportional dispatch
        for src in PRIORITY_SOURCES:
            row_data[src] = weights * daily_row[src]

        # Adjust for demand changes:
        # Reduction → cut fossil first, then biomass, then electricity
        # Increase → add from the cheapest available source
        for i in range(n):
            delta = demand_change_mwh[i]

            if delta < 0:
                # Demand decreased: cut from marginal (fossil → biomass → electricity)
                remaining_cut = abs(delta)
                for src in reversed(PRIORITY_SOURCES):
                    if src == "Waste heat":
                        continue  # Waste heat runs anyway (must-run)
                    cut = min(remaining_cut, row_data.loc[row_data.index[i], src])
                    row_data.loc[row_data.index[i], src] -= cut
                    remaining_cut -= cut
                    if remaining_cut <= 1e-9:
                        break

            elif delta > 0:
                # Demand increased: use cheapest available source with headroom
                remaining_add = delta
                for src in PRIORITY_SOURCES:
                    # Use waste heat if there's daily headroom
                    daily_remaining = max(0, daily_row[src] - row_data[src].sum())
                    add = min(remaining_add, daily_remaining / max(1, n - i))
                    row_data.loc[row_data.index[i], src] += add
                    remaining_add -= add
                    if remaining_add <= 1e-9:
                        break
                # Any leftover goes to fossil (the backup)
                if remaining_add > 1e-9:
                    row_data.loc[row_data.index[i], "Fossil fuel"] += remaining_add

        row_data["hourly_total_mwh"] = sum(row_data[src] for src in PRIORITY_SOURCES)

        results.append(row_data)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    from network_scaling import load_and_scale_building_load

    daily = load_daily_dispatch(2024)
    hourly = load_and_scale_building_load(2024)
    hourly = hourly.rename(columns={"network_load_mw": "network_load_mw"})

    result = distribute_daily_to_hourly(daily, hourly)
    print(f"Hourly dispatch: {len(result)} rows")
    print(f"Annual totals (MWh):")
    for src in PRIORITY_SOURCES:
        print(f"  {src}: {result[src].sum():,.0f}")
    print(f"  Total: {result['hourly_total_mwh'].sum():,.0f}")
    print(f"  (Daily dispatch total: {daily['daily_total_mwh'].sum():,.0f})")
