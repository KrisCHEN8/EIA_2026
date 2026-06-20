"""
Main Simulation Runner
======================
Runs three scenarios for the full year 2024:
    1. Baseline — no TES, no DSM
    2. TES only — TES at substations, no active DSM control
    3. TES + DSM — TES with peak shaving + load shifting

For each scenario, computes:
    - Hourly network demand profile
    - Return temperatures via substation model
    - Hourly production dispatch
    - CHP efficiency metrics

Outputs results to CSV files and summary statistics.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "simulation"))

from simulation.network_scaling import load_and_scale_building_load, compute_scale_factor
from simulation.substation_model import Substation, load_supply_temperature
from simulation.chp_model import CHPModel, FlueGasCondensationModel
from simulation.dsm_strategies import PeakShavingStrategy, LoadShiftingStrategy, CombinedDSMStrategy
from simulation.hourly_dispatch import load_daily_dispatch, distribute_daily_to_hourly, redispatch_for_modified_load

YEAR = 2024
OUTPUT_DIR = BASE_DIR / "simulation" / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

PRIORITY_SOURCES = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]


def run_scenario(
    scenario_name: str,
    hourly_demand: pd.DataFrame,
    supply_temps: pd.DataFrame,
    weather: pd.DataFrame,
    daily_dispatch: pd.DataFrame,
    dsm_strategy=None,
) -> pd.DataFrame:
    """
    Run a single scenario.

    Parameters
    ----------
    scenario_name : str
    hourly_demand : pd.DataFrame with 'date', 'network_load_mw'
    supply_temps : pd.DataFrame with 'date', 'T_supply'
    weather : pd.DataFrame with 'date', 'temperature'
    daily_dispatch : pd.DataFrame with daily production mix
    dsm_strategy : DSM strategy object (None for baseline)

    Returns
    -------
    pd.DataFrame with hourly results.
    """
    print(f"\n{'='*60}")
    print(f"  Running: {scenario_name}")
    print(f"{'='*60}")

    # ── Merge data ──────────────────────────────────────────────────────────
    df = hourly_demand[["date", "network_load_mw"]].copy()
    df = df.merge(supply_temps[["date", "T_supply"]], on="date", how="left")
    df = df.merge(weather[["date", "temperature"]], on="date", how="left")
    df = df.dropna().reset_index(drop=True)

    original_load_mw = df["network_load_mw"].values.copy()

    # ── Apply DSM if applicable ─────────────────────────────────────────────
    if dsm_strategy is not None:
        hours_of_day = df["date"].dt.hour.values

        if hasattr(dsm_strategy, "peak_shaving"):
            # Combined strategy
            dsm_result = dsm_strategy.apply(original_load_mw, hours_of_day)
        elif hasattr(dsm_strategy, "cap_percentile"):
            # Peak shaving only
            dsm_result = dsm_strategy.apply(original_load_mw)
        else:
            # Load shifting only
            dsm_result = dsm_strategy.apply(original_load_mw, hours_of_day)

        df["modified_load_mw"] = dsm_result["modified_load_mw"]
        df["tes_charge_mw"] = dsm_result["tes_charge_mw"]
    else:
        df["modified_load_mw"] = original_load_mw
        df["tes_charge_mw"] = 0.0

    # ── Compute return temperatures ─────────────────────────────────────────
    # Design capacity = 90th percentile of original demand (MW → kW)
    design_cap_kw = np.percentile(original_load_mw, 90) * 1000.0
    sub = Substation(design_capacity_kw=design_cap_kw)
    T_returns = []
    T_sec_returns = []
    m_dots = []

    for _, row in df.iterrows():
        result = sub.compute_return_temperature(
            T_supply_primary=row["T_supply"],
            Q_demand_kw=row["modified_load_mw"] * 1000.0,  # MW → kW
            T_outdoor=row["temperature"],
        )
        T_returns.append(result["T_return_primary"])
        T_sec_returns.append(result["T_secondary_return"])
        m_dots.append(result["m_dot_primary"])

    df["T_return_primary"] = T_returns
    df["T_secondary_return"] = T_sec_returns
    df["m_dot_primary"] = m_dots

    # ── Production dispatch ─────────────────────────────────────────────────
    if dsm_strategy is not None:
        dispatch = redispatch_for_modified_load(
            daily_dispatch,
            original_load_mw,
            df["modified_load_mw"].values,
            pd.DatetimeIndex(df["date"]),
        )
    else:
        dispatch_input = pd.DataFrame({
            "date": df["date"],
            "network_load_mw": df["modified_load_mw"],
        })
        dispatch = distribute_daily_to_hourly(daily_dispatch, dispatch_input)

    # Merge dispatch results
    if not dispatch.empty:
        for src in PRIORITY_SOURCES:
            if src in dispatch.columns:
                df[src] = dispatch[src].values[:len(df)]
            else:
                df[src] = 0.0
    else:
        for src in PRIORITY_SOURCES:
            df[src] = 0.0

    # ── CHP efficiency ──────────────────────────────────────────────────────
    chp = CHPModel()
    fgc = FlueGasCondensationModel()

    eta_els = []
    extra_elecs = []
    fgc_recoveries = []

    for _, row in df.iterrows():
        eff = chp.compute_efficiency(row["T_return_primary"])
        eta_els.append(eff["eta_el"])

        # FGC recovery
        fgc_kw = fgc.compute_recovery(
            row["T_return_primary"],
            row.get("Waste heat", 0) * 1000.0,  # MWh → kW for this hour
        )
        fgc_recoveries.append(fgc_kw / 1000.0)  # back to MW

    df["eta_el_chp"] = eta_els
    df["fgc_recovery_mw"] = fgc_recoveries

    df["scenario"] = scenario_name

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"  Hours simulated: {len(df)}")
    print(f"  Demand — original peak: {original_load_mw.max():.1f} MW, "
          f"modified peak: {df['modified_load_mw'].max():.1f} MW")
    print(f"  Mean T_return: {df['T_return_primary'].mean():.1f} °C")
    print(f"  Mean η_el (CHP): {df['eta_el_chp'].mean():.4f}")
    for src in PRIORITY_SOURCES:
        print(f"  {src}: {df[src].sum():,.0f} MWh")
    print(f"  FGC recovery: {df['fgc_recovery_mw'].sum():,.0f} MWh")

    return df


def main():
    print("Loading data...")

    # ── Load data ───────────────────────────────────────────────────────────
    hourly_demand = load_and_scale_building_load(YEAR, diversity_factor=0.85)
    supply_temps = load_supply_temperature(YEAR)
    daily_dispatch = load_daily_dispatch(YEAR)

    # Weather data
    weather = pd.read_csv(BASE_DIR / "DH production mix" / "weather.csv")
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather[weather["date"].dt.year == YEAR].copy()

    scale_info = compute_scale_factor(YEAR)
    print(f"\nNetwork scale factor: {scale_info['scale_factor']:.1f}")
    print(f"Network annual demand: {scale_info['network_annual_mwh']:,.0f} MWh")
    print(f"Peak network load: {hourly_demand['network_load_mw'].max():.1f} MW")

    # ── Scenario 1: Baseline (no TES, no DSM) ──────────────────────────────
    baseline = run_scenario(
        "Baseline (no TES/DSM)",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
        dsm_strategy=None,
    )

    # ── Scenario 2: TES only ───────────────────────────────────────────────
    # TES at substations acts as a thermal buffer, modeled as mild peak shaving
    tes_only = PeakShavingStrategy(
        cap_percentile=90,  # Only shave the top 10% peaks
        tes_efficiency=0.92,
    )
    tes_result = run_scenario(
        "TES only",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
        dsm_strategy=tes_only,
    )

    # ── Scenario 3: TES + DSM ──────────────────────────────────────────────
    dsm_combined = CombinedDSMStrategy(
        peak_shaving=PeakShavingStrategy(
            cap_percentile=80,
            tes_efficiency=0.92,
        ),
        load_shifting=LoadShiftingStrategy(
            shift_fraction=0.15,
            tes_efficiency=0.92,
        ),
    )
    dsm_result = run_scenario(
        "TES + DSM",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
        dsm_strategy=dsm_combined,
    )

    # ── Save results ────────────────────────────────────────────────────────
    baseline.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    tes_result.to_csv(OUTPUT_DIR / "tes_results.csv", index=False)
    dsm_result.to_csv(OUTPUT_DIR / "dsm_results.csv", index=False)

    # ── Combined summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — Scenario Comparison")
    print("=" * 70)

    scenarios = {
        "Baseline": baseline,
        "TES only": tes_result,
        "TES + DSM": dsm_result,
    }

    summary_rows = []
    for name, df in scenarios.items():
        row = {
            "Scenario": name,
            "Peak demand (MW)": df["modified_load_mw"].max(),
            "Mean demand (MW)": df["modified_load_mw"].mean(),
            "Mean T_return (°C)": df["T_return_primary"].mean(),
            "Mean η_el CHP": df["eta_el_chp"].mean(),
            "Waste heat (GWh)": df["Waste heat"].sum() / 1000,
            "Electricity (GWh)": df["Electricity"].sum() / 1000,
            "Biomass (GWh)": df["Biomass"].sum() / 1000,
            "Fossil fuel (GWh)": df["Fossil fuel"].sum() / 1000,
            "FGC recovery (GWh)": df["fgc_recovery_mw"].sum() / 1000,
            "Load factor": df["modified_load_mw"].mean() / df["modified_load_mw"].max(),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # Print formatted
    print(summary.to_string(index=False, float_format="{:.4f}".format))

    # ── Compute benefits relative to baseline ───────────────────────────────
    print("\n" + "=" * 70)
    print("  BENEFITS vs BASELINE")
    print("=" * 70)

    chp = CHPModel()
    base_row = summary_rows[0]

    for name, row in zip(["TES only", "TES + DSM"], summary_rows[1:]):
        print(f"\n  --- {name} ---")
        print(f"  Peak reduction: {base_row['Peak demand (MW)'] - row['Peak demand (MW)']:.1f} MW "
              f"({(1 - row['Peak demand (MW)']/base_row['Peak demand (MW)'])*100:.1f}%)")
        print(f"  ΔT_return: {base_row['Mean T_return (°C)'] - row['Mean T_return (°C)']:.2f} °C")
        print(f"  Δη_el: {(row['Mean η_el CHP'] - base_row['Mean η_el CHP'])*100:.3f} pp")
        print(f"  Fossil reduction: {base_row['Fossil fuel (GWh)'] - row['Fossil fuel (GWh)']:.2f} GWh "
              f"({(1 - row['Fossil fuel (GWh)']/max(base_row['Fossil fuel (GWh)'], 0.001))*100:.1f}%)")
        print(f"  Load factor improvement: {row['Load factor'] - base_row['Load factor']:.4f}")

        benefits = chp.compute_benefits(
            Q_heat_mwh=row["Waste heat (GWh)"] * 1000,
            T_return_baseline=base_row["Mean T_return (°C)"],
            T_return_scenario=row["Mean T_return (°C)"],
            waste_mwh=row["Waste heat (GWh)"] * 1000,
            fossil_mwh=row["Fossil fuel (GWh)"] * 1000,
            biomass_mwh=row["Biomass (GWh)"] * 1000,
        )
        print(f"  Extra CHP electricity: {benefits['extra_electricity_mwh']:,.0f} MWh")
        print(f"  Extra revenue: {benefits['extra_revenue_nok']:,.0f} NOK")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
