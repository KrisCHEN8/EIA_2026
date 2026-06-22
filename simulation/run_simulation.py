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
from simulation.TES_model import WaterTank, PCMStorageTank

YEAR = 2024
OUTPUT_DIR = BASE_DIR / "simulation" / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

PRIORITY_SOURCES = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]
CP_WATER = 4.184    # kJ/(kg·K)
ETA_HEX = 0.95      # HEX efficiency
T_APPROACH = 3.0    # Minimum primary-to-secondary return approach temperature (degC)
T_SEC_SUPPLY = 55.0   # Target secondary supply temperature (degC)
M_DOT_SEC = 0.25   # Secondary mass flow rate (kg/s)
T_MIN_TES = 40.0    # Minimum allowable TES temperature (degC); forces charging if breached


def compute_secondary_return_temperature(Q_demand_kw, T_sec_supply, m_dot_sec):
    if m_dot_sec > 0:
        T_sec_return = T_sec_supply - Q_demand_kw / (m_dot_sec * CP_WATER)
    else:
        T_sec_return = T_sec_supply
    return np.clip(T_sec_return, 25.0, T_sec_supply)


def find_required_T_in(tank, m_dot, T_target, dt_sub, n_steps):
    """
    Solve for the tank inlet temperature (T_in) that will result in the
    final tank water temperature reaching T_target after n_steps.
    """
    if m_dot <= 0:
        return T_target

    T_water_init = tank.T_water
    T_pcm_init = getattr(tank.pcm, "T", None) if hasattr(tank, "pcm") else None

    def run_sim(T_in_candidate):
        tank.T_water = T_water_init
        if hasattr(tank, "pcm") and T_pcm_init is not None:
            tank.pcm.T = T_pcm_init
        for _ in range(n_steps):
            tank.step(m_dot=m_dot, T_in=T_in_candidate, dt=dt_sub)
        return tank.T_water

    low = 25.0
    high = 150.0

    y_low = run_sim(low)
    y_high = run_sim(high)

    if y_low >= T_target:
        T_in_opt = low
    elif y_high <= T_target:
        T_in_opt = high
    else:
        for _ in range(12):
            mid = (low + high) / 2.0
            y_mid = run_sim(mid)
            if y_mid < T_target:
                low = mid
            else:
                high = mid
        T_in_opt = (low + high) / 2.0

    tank.T_water = T_water_init
    if hasattr(tank, "pcm") and T_pcm_init is not None:
        tank.pcm.T = T_pcm_init

    return T_in_opt


def run_scenario(
    scenario_name: str,
    hourly_demand: pd.DataFrame,
    supply_temps: pd.Series,
    weather: pd.DataFrame,
    daily_dispatch: pd.DataFrame,
):
    """
    Run a single scenario with physical hour-by-hour storage integration.

    scenario_name: str
    hourly_demand: pd.DataFrame with 'date', 'network_load_mw'
    supply_temps: pd.Series of hourly 'T_supply'
    weather: pd.DataFrame with 'date', 'temperature'
    daily_dispatch: pd.DataFrame with daily DH production mix
    """
    print(f"\n{'='*60}")
    print(f"  Running: {scenario_name}")
    print(f"{'='*60}")

    df = hourly_demand[["date", "network_load_mw"]].copy()

    supply_temps_df = pd.DataFrame({
        "date": hourly_demand["date"].sort_values().reset_index(drop=True),
        "T_supply": supply_temps
    })

    df = df.merge(supply_temps_df, on="date", how="left")
    df = df.merge(weather, on="date", how="left")
    df = df.dropna().reset_index(drop=True)

    # Filter for heatings eason (January to March)
    df = df[(df["date"] >= f"{YEAR}-01-01 00:00:00") & (df["date"] <= f"{YEAR}-03-31 23:00:00")].reset_index(drop=True)

    original_load_mw = df["network_load_mw"].values.copy()

    # Compute scale factor
    scale_info = compute_scale_factor(YEAR)
    scale_factor = scale_info["scale_factor"]

    # Sizing and initializing the TES based on the scenario name
    if scenario_name == "Water Tank + DSM":
        tank = WaterTank(water_mass_kg=1000.0, initial_temperature=50.0)
    elif scenario_name == "PCM Storage + DSM":
        tank = PCMStorageTank(water_mass_kg=500.0, pcm_mass_kg=578.0, initial_temperature=50.0)
    else:
        tank = None

    substation = Substation(
        secondary_supply_temp=T_SEC_SUPPLY,
        max_primary_flow_kgs=0.5,
        T_approach=T_APPROACH,
    )

    # Simulation lists
    modified_load_mw = []
    tes_charge_mw = []
    T_returns = []
    T_sec_returns = []
    T_sec_HEX_outs = []
    m_dots_primary = []
    T_water_temps = []
    T_pcm_temps = []
    Q_primary_kws = []

    # Charging and sub-stepping settings
    n_steps = 60
    dt_sub = 60.0
    max_primary_flow_discharging = 0.01

    # Hourly physical simulation loop
    for idx, row in df.iterrows():
        T_supply = row["T_supply"]
        T_outdoor = row["temperature"]
        hour = row["date"].hour
        network_load_mw = row["network_load_mw"]

        # Building-level heat demand
        Q_demand_building = (network_load_mw * 1000.0) / scale_factor

        m_dot_sec = M_DOT_SEC if Q_demand_building > 0 else 0.0

        if tank is None:
            # Baseline
            T_sec_supply_target = T_SEC_SUPPLY
            T_sec_return = compute_secondary_return_temperature(Q_demand_building, T_sec_supply_target, m_dot_sec)

            # Solve for primary flow rate to hit T_sec_supply_target.
            # delta_T uses T_sec_return + T_APPROACH as the effective cold-side floor,
            # enforcing the same minimum approach temperature as compute_hex.
            dT_sec = max(0.0, T_sec_supply_target - T_sec_return)
            delta_T = T_supply - (T_sec_return + T_APPROACH)
            if delta_T > 0 and ETA_HEX > 0:
                m_dot_needed = (m_dot_sec * dT_sec) / (ETA_HEX * delta_T)
            else:
                m_dot_needed = 0.0

            m_dot_primary = min(m_dot_needed, substation.max_primary_flow)

            # Compute HEX
            r = substation.compute_hex(
                T_supply_primary=T_supply,
                m_dot_primary=m_dot_primary,
                T_sec_return=T_sec_return,
                T_sec_supply_desired=T_sec_supply_target,
                out_temp=T_outdoor,
                eta_HEX=ETA_HEX,
                m_dot_sec=m_dot_sec,
            )

            T_return_primary = r["T_return_primary"]
            T_sec_supply = r["T_sec_supply_HEX"]
            Q_primary_kw = r["Q_primary_kw"]

            # Recalculate secondary return based on actual supply temperature
            Q_delivered_kw = ETA_HEX * Q_primary_kw
            T_sec_return = compute_secondary_return_temperature(Q_delivered_kw, T_sec_supply, m_dot_sec)

            T_water_val = np.nan
            T_pcm_val = np.nan
            T_sec_HEX_out = T_sec_supply

        else:
            # TES scenario
            T_water_current = tank.T_water
            T_sec_return_est = compute_secondary_return_temperature(Q_demand_building, T_SEC_SUPPLY, m_dot_sec=M_DOT_SEC)

            is_low_load_hour = network_load_mw < 150.0
            is_charging_hour = hour in [23, 0, 1, 2, 3, 4, 5, 6]
            is_discharging_hour = hour in [7, 8, 9, 17, 18, 19, 20]
            is_below_min_temp = T_water_current < T_MIN_TES  # Hard minimum constraint

            mode = "standby"

            if is_below_min_temp:
                # Emergency charging: TES temperature has fallen below the minimum
                # allowable threshold (T_MIN_TES = 40 °C). Force charging regardless
                # of the time-of-day schedule or network load level.
                m_dot_sec = M_DOT_SEC
                mode = "charging"
            elif is_charging_hour and is_low_load_hour and (T_water_current < T_SEC_SUPPLY - 1.0):
                # Scheduled off-peak charging
                m_dot_sec = M_DOT_SEC
                mode = "charging"
            elif is_discharging_hour and (T_water_current > T_sec_return_est + 2.0):
                mode = "discharging"
            else:
                mode = "standby"

            # Series circuit calculations
            T_sec_supply = T_water_current
            T_sec_return = compute_secondary_return_temperature(Q_demand_building, T_sec_supply, m_dot_sec=m_dot_sec)
            T_sec_inlet = T_sec_return

            # Target temperature selection based on mode
            if mode == "charging":
                T_target_HEX = find_required_T_in(tank, m_dot_sec, T_SEC_SUPPLY, dt_sub, n_steps)
                T_target_HEX = min(T_target_HEX, 75.0)  # maximum 75 degC
            else:
                T_target_HEX = T_SEC_SUPPLY

            # Calculate primary flow limit
            if mode == "discharging":
                limit = max_primary_flow_discharging
            else:
                limit = substation.max_primary_flow

            # Solve for primary flow rate to hit T_target_HEX
            dT_sec = max(0.0, T_target_HEX - T_sec_inlet)
            delta_T = T_supply - (T_sec_inlet + T_APPROACH)
            if delta_T > 0 and ETA_HEX > 0:
                m_dot_needed = (m_dot_sec * dT_sec) / (ETA_HEX * delta_T)
            else:
                m_dot_needed = 0.0

            m_dot_primary = min(m_dot_needed, limit)

            # Compute HEX
            r = substation.compute_hex(
                T_supply_primary=T_supply,
                m_dot_primary=m_dot_primary,
                T_sec_return=T_sec_inlet,
                T_sec_supply_desired=T_target_HEX,
                out_temp=T_outdoor,
                eta_HEX=ETA_HEX,
                m_dot_sec=m_dot_sec,
            )

            T_return_primary = r["T_return_primary"]
            T_sec_supply_HEX = r["T_sec_supply_HEX"]
            Q_primary_kw = r["Q_primary_kw"]

            # Step the tank using HEX outlet temperature
            T_out_sum = 0.0
            for _ in range(n_steps):
                T_out_sub = tank.step(m_dot=m_dot_sec, T_in=T_sec_supply_HEX, dt=dt_sub)
                T_out_sum += T_out_sub
            T_out_tank = T_out_sum / n_steps

            # Update final values
            T_sec_supply = T_out_tank
            T_sec_return = compute_secondary_return_temperature(Q_demand_building, T_sec_supply, m_dot_sec=m_dot_sec)

            T_water_val = tank.T_water
            T_pcm_val = tank.pcm.T if hasattr(tank, "pcm") else np.nan
            T_sec_HEX_out = T_sec_supply_HEX

        # Scale parameters back to full network level
        modified_demand_network_mw = (Q_primary_kw * scale_factor) / 1000.0
        net_charge_network_mw = modified_demand_network_mw - (Q_demand_building * scale_factor / 1000.0)

        modified_load_mw.append(max(modified_demand_network_mw, 0.0))
        tes_charge_mw.append(net_charge_network_mw)
        T_returns.append(T_return_primary)
        T_sec_returns.append(T_sec_return)
        T_sec_HEX_outs.append(T_sec_HEX_out)
        m_dots_primary.append(m_dot_primary * scale_factor)
        T_water_temps.append(T_water_val)
        T_pcm_temps.append(T_pcm_val)
        Q_primary_kws.append(Q_primary_kw)

    df["modified_load_mw"] = modified_load_mw
    df["tes_charge_mw"] = tes_charge_mw
    df["T_return_primary"] = T_returns
    df["T_secondary_return"] = T_sec_returns
    df["T_HEX_out"] = T_sec_HEX_outs
    df["m_dot_primary"] = m_dots_primary
    df["T_water"] = T_water_temps
    df["T_pcm"] = T_pcm_temps
    df["Q_primary_kw"] = Q_primary_kws

    # Production dispatch
    if tank is not None:
        # Redispatch for modified load profile
        dispatch = redispatch_for_modified_load(
            daily_dispatch,
            original_load_mw,
            df["modified_load_mw"].values,
            pd.DatetimeIndex(df["date"]),
        )
    else:
        # Proportional dispatch
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

    # CHP efficiency and Flue Gas Condensation
    chp = CHPModel()
    fgc = FlueGasCondensationModel()

    eta_els = []
    fgc_recoveries = []

    for _, row in df.iterrows():
        eff = chp.compute_efficiency(row["T_return_primary"])
        eta_els.append(eff["eta_el"])

        # FGC recovery based on waste heat hourly output
        fgc_kw = fgc.compute_recovery(
            row["T_return_primary"],
            row.get("Waste heat", 0) * 1000.0,
        )
        fgc_recoveries.append(fgc_kw / 1000.0)  # Convert back to MW

    df["eta_el_chp"] = eta_els
    df["fgc_recovery_mw"] = fgc_recoveries
    df["scenario"] = scenario_name

    # print KPIs
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

    # Load data
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

    # Baseline Scenario
    baseline = run_scenario(
        "Baseline (no TES/DSM)",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
    )

    # Water Tank scenario
    water_tank_res = run_scenario(
        "Water Tank + DSM",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
    )

    # PCM Storage scenario
    pcm_res = run_scenario(
        "PCM Storage + DSM",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
    )

    # Save results
    baseline.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    water_tank_res.to_csv(OUTPUT_DIR / "water_tank_results.csv", index=False)
    pcm_res.to_csv(OUTPUT_DIR / "pcm_storage_results.csv", index=False)

    # Combined summary
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

    print(summary.to_string(index=False, float_format="{:.4f}".format))

    # Benefits relative to baseline
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