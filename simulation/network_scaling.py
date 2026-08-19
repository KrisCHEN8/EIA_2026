"""
Network Scaling Module
======================
Scales a single-building hourly load profile to the full DH network demand.

Method:
    1. Compute total annual network demand from daily production dispatch data.
    2. Compute total annual building demand from hourly building load.
    3. Scale factor = network_demand / building_demand.
    4. Apply diversity factor (peaks don't coincide across thousands of buildings).

The diversity factor reduces the instantaneous peak but preserves total energy.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DH production mix"


def compute_scale_factor(year: int = 2024) -> dict:
    """
    Compute the energy-based scaling factor from one building to the full network.

    Returns a dict with all intermediate values for transparency.
    """
    # ── Network annual demand (from daily dispatch) ─────────────────────────
    dispatch = pd.read_csv(DATA_DIR / f"daily_priority_dispatch_{year}.csv")
    network_annual_mwh = dispatch["daily_total_mwh"].sum()

    # ── Building annual demand ──────────────────────────────────────────────
    load = pd.read_csv(DATA_DIR / "building_load.csv")
    # Handle the duplicate 'date' column in the CSV
    if load.shape[1] == 3:
        load.columns = ["idx", "date", "load"]
    load["date"] = pd.to_datetime(load["date"])
    load_year = load[load["date"].dt.year == year]
    building_annual_kwh = load_year["load"].sum()   # kW × 1 h = kWh
    building_annual_mwh = building_annual_kwh / 1000.0

    building_area_m2 = 1350.0
    specific_demand = building_annual_kwh / building_area_m2  # kWh/m²/year

    # ── Scale factor ───────────────────────────────────────────────────────
    scale_factor = network_annual_mwh / building_annual_mwh

    # Effective floor area served by the network
    effective_area_m2 = network_annual_mwh * 1e3 / specific_demand

    return {
        "network_annual_mwh": network_annual_mwh,
        "building_annual_mwh": building_annual_mwh,
        "building_area_m2": building_area_m2,
        "specific_demand_kwh_m2": specific_demand,
        "scale_factor": scale_factor,
        "effective_area_m2": effective_area_m2,
        "year": year,
    }


def load_and_scale_building_load(
    year: int = 2024,
    diversity_factor: float = 0.85,
) -> pd.DataFrame:
    """
    Load the hourly building load for the given year and scale to network level.

    Parameters
    ----------
    year : int
        The year to filter.
    diversity_factor : float
        Multiplier applied to the scaled peak to account for load diversity
        across many buildings (0.6–1.0, lower = more peak reduction).
        Applied as a rolling-average smoothing: the peak is reduced but annual
        energy is preserved by redistributing clipped energy to adjacent hours.

    Returns
    -------
    pd.DataFrame with columns:
        date, building_load_kw, network_load_kw, network_load_mw
    """
    info = compute_scale_factor(year)
    sf = info["scale_factor"]

    # ── Load building data ──────────────────────────────────────────────────
    load = pd.read_csv(DATA_DIR / "building_load.csv")
    if load.shape[1] == 3:
        load.columns = ["idx", "date", "load"]
    load["date"] = pd.to_datetime(load["date"])
    load = load[load["date"].dt.year == year].copy()
    load = load.sort_values("date").reset_index(drop=True)

    # ── Scale ───────────────────────────────────────────────────────────────
    load["building_load_kw"] = load["load"].clip(lower=0)
    load["network_load_kw"] = load["building_load_kw"] * sf

    # ── Apply diversity: smooth out peaks via weighted rolling mean ─────────
    if diversity_factor < 1.0:
        # Window width chosen so that the smoothed peak ≈ diversity_factor × raw peak
        window = max(3, int(2 * (1.0 / diversity_factor)))
        smoothed = load["network_load_kw"].rolling(window, center=True, min_periods=1).mean()
        # Preserve total energy: rescale smoothed series to match original sum
        energy_ratio = load["network_load_kw"].sum() / smoothed.sum()
        load["network_load_kw"] = smoothed * energy_ratio

    load["network_load_mw"] = load["network_load_kw"] / 1000.0

    return load[["date", "building_load_kw", "network_load_kw", "network_load_mw"]].copy()


if __name__ == "__main__":
    info = compute_scale_factor(2024)
    print("=== Network Scaling Parameters ===")
    for k, v in info.items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")

    df = load_and_scale_building_load(2024, diversity_factor=0.85)
    print(f"\nNetwork load profile ({len(df)} hours):")
    print(f"  Peak: {df['network_load_mw'].max():.1f} MW")
    print(f"  Mean: {df['network_load_mw'].mean():.1f} MW")
    print(f"  Annual energy: {df['network_load_mw'].sum():.1f} MWh")
    print(df.head(10))
