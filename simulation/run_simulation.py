"""
Main Simulation Runner
======================
Runs three scenarios for the full year 2024:
    1. Baseline — no TES, no DSM
    2. Water Tank + DSM — physical 1000L Water Tank storage with off-peak charging and peak discharging
    3. PCM Storage + DSM — physical 1000L-equivalent PCM Storage Tank with matching controls

For each scenario, computes:
    - Hourly network demand profile
    - Return temperatures via substation model coupled with storage states
    - Hourly production dispatch
    - CHP efficiency metrics
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
from simulation.hourly_dispatch import load_daily_dispatch, distribute_daily_to_hourly, redispatch_for_modified_load

YEAR = 2024
OUTPUT_DIR = BASE_DIR / "simulation" / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

PRIORITY_SOURCES = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]
CP_WATER = 4.184  # kJ/(kg·K)


def run_scenario(
    scenario_name: str,
    hourly_demand: pd.DataFrame,
    supply_temps: pd.DataFrame,
    weather: pd.DataFrame,
    daily_dispatch: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run a single scenario with physical hour-by-hour storage integration.

    Parameters
    ----------
    scenario_name : str
    hourly_demand : pd.DataFrame with 'date', 'network_load_mw'
    supply_temps : pd.DataFrame with 'date', 'T_supply'
    weather : pd.DataFrame with 'date', 'temperature'
    daily_dispatch : pd.DataFrame with daily production mix

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

    # Filter for January to March (heating season: months 1 to 3)
    df = df[(df["date"] >= f"{YEAR}-01-01 00:00:00") & (df["date"] <= f"{YEAR}-03-31 23:00:00")].reset_index(drop=True)

    original_load_mw = df["network_load_mw"].values.copy()

    # ── Compute scale factor ────────────────────────────────────────────────
    scale_info = compute_scale_factor(YEAR)
    scale_factor = scale_info["scale_factor"]

    # Design capacity = 90th percentile of original building demand (kW)
    design_cap_kw = np.percentile(original_load_mw, 90) * 1000.0 / scale_factor
    substation = Substation(design_capacity_kw=design_cap_kw * scale_factor)

    # ── Initialize Storage Tank if applicable ───────────────────────────────
    tank = None
    if scenario_name == "Water Tank + DSM":
        from TES_model import WaterTank
        # 1000L Water Tank
        tank = WaterTank(water_mass_kg=1000.0, initial_temperature=50.0)
    elif scenario_name == "PCM Storage + DSM":
        from TES_model import PCMStorageTank
        # 1000L equivalent volume: 417 kg water + 483 kg PCM (approx 1000 liters total)
        tank = PCMStorageTank(water_mass_kg=417.0, pcm_mass_kg=483.0, initial_temperature=50.0)

    # Sizing Peak Shaving Cap (90% of annual peak demand)
    network_peak_mw = original_load_mw.max()
    Q_cap_building_kw = (network_peak_mw * 0.90 * 1000.0) / scale_factor

    # Simulation lists
    modified_load_mw = []
    tes_charge_mw = []
    T_returns = []
    T_sec_returns = []
    m_dots_primary = []
    T_water_temps = []
    T_pcm_temps = []

    # Charging and sub-stepping settings
    m_dot_charge_building_nominal = 0.05  # kg/s nominal charging flow
    n_steps = 60                          # 60 steps per hour for numerical stability
    dt_sub = 60.0                         # 60-second sub-step
    dt_seconds = 3600.0

    # ── Hourly physical simulation loop ──────────────────────────────────────
    for idx, row in df.iterrows():
        T_supply = row["T_supply"]
        T_outdoor = row["temperature"]
        hour = row["date"].hour

        # Building-level heat demand (kW)
        Q_demand_building = (row["network_load_mw"] * 1000.0) / scale_factor

        # Secondary side radiator temperatures estimation (from substation model)
        T_sec_supply = min(
            T_supply - substation.approach_temp,
            substation.T_sec_supply_design,
        )

        T_out_design = -20.0
        T_out_balance = 17.0
        load_fraction = np.clip(
            (T_out_balance - T_outdoor) / (T_out_balance - T_out_design),
            0.0, 1.0,
        )
        T_sec_return = (
            substation.T_sec_return_design * load_fraction
            + T_sec_supply * (1 - load_fraction)
        )

        # Secondary radiator water mass flow rate (kg/s)
        delta_T_sec = T_sec_supply - T_sec_return
        if delta_T_sec <= 1.0:  # Avoid division by zero or very small values
            m_dot_sec = 0.0
        else:
            m_dot_sec = Q_demand_building / (CP_WATER * delta_T_sec)
        
        # Cap at maximum design flow rate to prevent numerical blowup
        m_dot_sec = min(m_dot_sec, 0.5)

        Q_charge_kw = 0.0
        Q_discharge_kw = 0.0
        T_out_tank = T_supply

        # DSM Schedule:
        # Off-peak charging: 23:00 - 06:00
        # Peak discharging: 07:00 - 09:00, 17:00 - 20:00
        is_charging_hour = hour in [23, 0, 1, 2, 3, 4, 5, 6]
        is_discharging_hour = hour in [7, 8, 9, 17, 18, 19, 20]

        if tank is not None:
            T_water = tank.T_water

            # Fossil avoidance rule: only charge when network demand is under 150 MW
            # (which avoids base-load fossil dispatch)
            is_low_load_hour = row["network_load_mw"] < 150.0

            if is_charging_hour and is_low_load_hour and (T_water < T_supply - 1.0):
                # Peak prevention: cap the charging flow rate so building load + charging <= Q_cap_building
                max_charging_power_kw = max(0.0, Q_cap_building_kw - Q_demand_building)
                m_dot_charge_max = max_charging_power_kw / (CP_WATER * max(0.1, T_supply - T_water))
                m_dot_charge = min(m_dot_charge_building_nominal, m_dot_charge_max)

                if m_dot_charge > 0.001:
                    # Sub-step the tank model for stability
                    T_out_sum = 0.0
                    for _ in range(n_steps):
                        T_out_sub = tank.step(m_dot=m_dot_charge, T_in=T_supply, dt=dt_sub)
                        T_out_sum += T_out_sub
                    T_out_tank = T_out_sum / n_steps

                    # Thermal power absorbed by the storage (kW)
                    Q_charge_kw = m_dot_charge * CP_WATER * (T_supply - T_out_tank)
                    Q_charge_kw = max(Q_charge_kw, 0.0)
                else:
                    m_dot_charge = 0.0
                    T_out_sum = 0.0
                    for _ in range(n_steps):
                        T_out_sub = tank.step(m_dot=0.0, T_in=0.0, dt=dt_sub)
                        T_out_sum += T_out_sub
                    T_out_tank = T_out_sum / n_steps

                # Building load is met normally by primary network
                Q_primary_sub_kw = Q_demand_building

                # Run substation for building load
                sub_res = substation.compute_return_temperature(T_supply, Q_primary_sub_kw, T_outdoor)
                T_return_sub = sub_res["T_return_primary"]
                m_dot_sub = sub_res["m_dot_primary"]

                # Overall return temperature is the flow-weighted average of substation return and tank return
                m_dot_total = m_dot_sub + m_dot_charge
                if m_dot_total > 0:
                    T_return_overall = (m_dot_sub * T_return_sub + m_dot_charge * T_out_tank) / m_dot_total
                else:
                    T_return_overall = T_supply

            elif is_discharging_hour and (T_water > T_sec_return + 2.0):
                # Discharge: route secondary radiator return water through tank
                # Sub-step the tank model for stability
                T_out_sum = 0.0
                for _ in range(n_steps):
                    T_out_sub = tank.step(m_dot=m_dot_sec, T_in=T_sec_return, dt=dt_sub)
                    T_out_sum += T_out_sub
                T_out_tank = T_out_sum / n_steps

                # Heat rate delivered from tank to building heating circuit (kW)
                Q_discharge_kw = m_dot_sec * CP_WATER * (T_out_tank - T_sec_return)
                Q_discharge_kw = np.clip(Q_discharge_kw, 0.0, Q_demand_building)

                # Remaining demand to be supplied by the primary network
                Q_primary_sub_kw = Q_demand_building - Q_discharge_kw

                # Run substation for the reduced primary load
                sub_res = substation.compute_return_temperature(T_supply, Q_primary_sub_kw, T_outdoor)
                T_return_overall = sub_res["T_return_primary"]
                m_dot_total = sub_res["m_dot_primary"]

            else:
                # Standby: no flow through the tank, only ambient losses
                T_out_sum = 0.0
                for _ in range(n_steps):
                    T_out_sub = tank.step(m_dot=0.0, T_in=0.0, dt=dt_sub)
                    T_out_sum += T_out_sub
                T_out_tank = T_out_sum / n_steps
                
                Q_primary_sub_kw = Q_demand_building

                sub_res = substation.compute_return_temperature(T_supply, Q_primary_sub_kw, T_outdoor)
                T_return_overall = sub_res["T_return_primary"]
                m_dot_total = sub_res["m_dot_primary"]

            # Save tank state temperatures
            T_water_temps.append(tank.T_water)
            if hasattr(tank, "pcm"):
                T_pcm_temps.append(tank.pcm.T)
            else:
                T_pcm_temps.append(np.nan)
        else:
            # Baseline: no storage tank
            Q_primary_sub_kw = Q_demand_building
            sub_res = substation.compute_return_temperature(T_supply, Q_primary_sub_kw, T_outdoor)
            T_return_overall = sub_res["T_return_primary"]
            m_dot_total = sub_res["m_dot_primary"]

            T_water_temps.append(np.nan)
            T_pcm_temps.append(np.nan)

        # Scale parameters back to full network level
        net_charge_building_kw = Q_charge_kw - Q_discharge_kw
        net_charge_network_mw = (net_charge_building_kw * scale_factor) / 1000.0
        modified_demand_network_mw = original_load_mw[idx] + net_charge_network_mw

        modified_load_mw.append(max(modified_demand_network_mw, 0.0))
        tes_charge_mw.append(net_charge_network_mw)
        T_returns.append(T_return_overall)
        T_sec_returns.append(T_sec_return)
        m_dots_primary.append(m_dot_total * scale_factor)

    df["modified_load_mw"] = modified_load_mw
    df["tes_charge_mw"] = tes_charge_mw
    df["T_return_primary"] = T_returns
    df["T_secondary_return"] = T_sec_returns
    df["m_dot_primary"] = m_dots_primary
    df["T_water"] = T_water_temps
    df["T_pcm"] = T_pcm_temps

    # ── Production dispatch ─────────────────────────────────────────────────
    if tank is not None:
        # Re-dispatch for modified load profile (DSM/shifting)
        dispatch = redispatch_for_modified_load(
            daily_dispatch,
            original_load_mw,
            df["modified_load_mw"].values,
            pd.DatetimeIndex(df["date"]),
        )
    else:
        # Standard proportional dispatch matching baseline demand shape
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

    # ── CHP efficiency and Flue Gas Condensation ────────────────────────────
    chp = CHPModel()
    fgc = FlueGasCondensationModel()

    eta_els = []
    fgc_recoveries = []

    for _, row in df.iterrows():
        eff = chp.compute_efficiency(row["T_return_primary"])
        eta_els.append(eff["eta_el"])

        # FGC recovery based on waste heat hourly output (MWh -> kW)
        fgc_kw = fgc.compute_recovery(
            row["T_return_primary"],
            row.get("Waste heat", 0) * 1000.0,
        )
        fgc_recoveries.append(fgc_kw / 1000.0)  # Convert back to MW

    df["eta_el_chp"] = eta_els
    df["fgc_recovery_mw"] = fgc_recoveries
    df["scenario"] = scenario_name

    # ── Summary KPIs printout ───────────────────────────────────────────────
    print(f"  Hours simulated: {len(df)}")
    print(f"  Demand — original peak: {original_load_mw.max():.1f} MW, "
          f"modified peak: {df['modified_load_mw'].max():.1f} MW")
    # Flow-weighted return temperature average
    flow_sum = df["m_dot_primary"].sum()
    if flow_sum > 0:
        weighted_T_return = (df["T_return_primary"] * df["m_dot_primary"]).sum() / flow_sum
    else:
        weighted_T_return = df["T_return_primary"].mean()
    print(f"  Mean T_return (flow-weighted): {weighted_T_return:.2f} °C (unweighted mean: {df['T_return_primary'].mean():.2f} °C)")
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
    )

    # ── Scenario 2: Water Tank + DSM ───────────────────────────────────────
    water_tank_res = run_scenario(
        "Water Tank + DSM",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
    )

    # ── Scenario 3: PCM Storage + DSM ──────────────────────────────────────
    pcm_res = run_scenario(
        "PCM Storage + DSM",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
    )

    # ── Save results ────────────────────────────────────────────────────────
    baseline.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    water_tank_res.to_csv(OUTPUT_DIR / "tes_results.csv", index=False)
    pcm_res.to_csv(OUTPUT_DIR / "dsm_results.csv", index=False)

    # ── Combined summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — Scenario Comparison")
    print("=" * 70)

    scenarios = {
        "Baseline": baseline,
        "Water Tank + DSM": water_tank_res,
        "PCM Storage + DSM": pcm_res,
    }

    summary_rows = []
    for name, df in scenarios.items():
        flow_sum = df["m_dot_primary"].sum()
        mean_t_return = (df["T_return_primary"] * df["m_dot_primary"]).sum() / flow_sum if flow_sum > 0 else df["T_return_primary"].mean()
        row = {
            "Scenario": name,
            "Peak demand (MW)": df["modified_load_mw"].max(),
            "Mean demand (MW)": df["modified_load_mw"].mean(),
            "Mean T_return (°C)": mean_t_return,
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

    for name, row in zip(["Water Tank + DSM", "PCM Storage + DSM"], summary_rows[1:]):
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
