import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DH production mix"

BUILDING_AREA_M2 = 1350.0


def _read_building_load(year: int) -> pd.DataFrame:
    load = pd.read_csv(DATA_DIR / "building_load.csv")
    # CSV has a duplicated 'date' column: keep the first date column and the load
    load = pd.DataFrame({"date": pd.to_datetime(load.iloc[:, 0]), "load": load.iloc[:, -1]})
    load = load[load["date"].dt.year == year].sort_values("date").reset_index(drop=True)
    load["building_load_kw"] = load["load"].clip(lower=0.0)
    return load[["date", "building_load_kw"]]


def _read_daily_dispatch(year: int) -> pd.DataFrame:
    d = pd.read_csv(DATA_DIR / f"daily_priority_dispatch_{year}.csv", parse_dates=["date"])
    return d[["date", "daily_total_mwh"]]


def compute_scale_factor(year: int = 2024) -> dict:
    """Annual (energy-based) scale factor: kept for reporting and as a fallback."""
    network_annual_mwh = _read_daily_dispatch(year)["daily_total_mwh"].sum()
    building_annual_kwh = _read_building_load(year)["building_load_kw"].sum()
    building_annual_mwh = building_annual_kwh / 1000.0
    specific_demand = building_annual_kwh / BUILDING_AREA_M2
    scale_factor = network_annual_mwh / building_annual_mwh
    return {
        "network_annual_mwh": network_annual_mwh,
        "building_annual_mwh": building_annual_mwh,
        "building_area_m2": BUILDING_AREA_M2,
        "specific_demand_kwh_m2": specific_demand,
        "scale_factor": scale_factor,
        "effective_area_m2": network_annual_mwh * 1e3 / specific_demand,
        "year": year,
    }


def load_and_scale_building_load(year: int = 2024, anchor_daily: bool = True) -> pd.DataFrame:
    """
    Hourly building demand and the matching network load.

    Returns a DataFrame with columns:
        date
        building_load_kw  : raw building profile (data)
        scale_factor      : per-hour factor building -> network (constant within a day)
        network_load_mw   : building_load_kw * scale_factor / 1000
        building_demand_kw: network_load_mw * 1000 / scale_factor (= building_load_kw;
                            explicit so the round trip is visible)
    """
    load = _read_building_load(year)
    sf_annual = compute_scale_factor(year)["scale_factor"]

    if anchor_daily:
        daily = _read_daily_dispatch(year)
        daily_mwh = pd.Series(daily["daily_total_mwh"].values, index=daily["date"].dt.date)
        day = load["date"].dt.date
        building_daily_mwh = load.groupby(day)["building_load_kw"].transform("sum") / 1000.0
        network_daily_mwh = day.map(daily_mwh)
        sf = network_daily_mwh / building_daily_mwh
        # Days missing in the dispatch data, or with zero building load: annual factor
        sf = sf.where(np.isfinite(sf) & (sf > 0), sf_annual)
        load["scale_factor"] = sf.astype(float).values
    else:
        load["scale_factor"] = sf_annual

    load["network_load_mw"] = load["building_load_kw"] * load["scale_factor"] / 1000.0
    load["building_demand_kw"] = load["network_load_mw"] * 1000.0 / load["scale_factor"]
    return load[["date", "building_load_kw", "scale_factor", "network_load_mw", "building_demand_kw"]]


if __name__ == "__main__":
    info = compute_scale_factor(2024)
    print("=== Network Scaling Parameters (annual) ===")
    for k, v in info.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")

    df = load_and_scale_building_load(2024)
    print(f"\nDaily-anchored network load ({len(df)} hours):")
    print(f"  Peak: {df['network_load_mw'].max():.1f} MW")
    print(f"  Mean: {df['network_load_mw'].mean():.1f} MW")
    print(f"  Energy: {df['network_load_mw'].sum():,.0f} MWh")
    print(f"  Daily scale factor range: {df['scale_factor'].min():,.0f} - {df['scale_factor'].max():,.0f}")
