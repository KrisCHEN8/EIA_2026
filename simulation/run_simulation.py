import sys
import numpy as np
import pandas as pd
from pathlib import Path

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
ETA_HEX = 0.90      # HEX efficiency
T_APPROACH = 3.0    # Minimum primary-to-secondary return approach temperature (degC)
T_SEC_SUPPLY = 60.0   # Target secondary supply temperature (degC)

# Variable Flow Constraints
M_DOT_SEC_MAX = 0.30    # Upper bound secondary flow rate (kg/s)
M_DOT_SEC_MIN = 0.02    # Lower bound secondary flow rate (kg/s)
DELTA_T_DESIGN = 20.0    # Design temperature difference (degC)

M_DOT_PRI_MAX = 0.1    # Upper bound primary flow rate (kg/s)
M_DOT_PRI_MIN = 0.05    # Lower bound primary flow rate (kg/s)

T_MIN_TES = 45.0    # Minimum TES temperature (degC)
T_MAX_TES = 65.0    # Maximum TES temperature (degC)

NUM_LAYERS = 3     # Number of layers in the stratified TES model


def get_secondary_supply_temp(t_outdoor):
    """
    Outdoor weather compensation curve
    GT31 (outdoor): -20, -10,   0,   5,  20
    GT11 (supply):   60,  56,  52,  40,  30
    """
    if t_outdoor <= -10.0:
        return 60.0
    elif t_outdoor <= 0.0:
        return 56.0
    elif t_outdoor <= 5.0:
        return 52.0
    else:
        return 40.0


def compute_realistic_secondary_side(Q_demand_kw, T_sec_supply, m_dot_sec, T_indoor=21.0):
    """
    Computes secondary return temperature
    """
    T_sec_supply_design = 60.0
    T_sec_return_design = 40.0
    
    # Calculate design LMTD
    dT_ln_design = (T_sec_supply_design - T_sec_return_design) / np.log(
        (T_sec_supply_design - T_indoor) / (T_sec_return_design - T_indoor)
    )

    Q_radiator_design = 50.0    # kW
    n_exponent = 1.3            # 1.3 for radiators

    if m_dot_sec <= 0 or T_sec_supply <= T_indoor:
        return T_sec_supply, 0.0

    low = T_indoor + 0.1
    high = T_sec_supply
    
    n_iter = 15
    for _ in range(n_iter):
        mid_T_return = (low + high) / 2.0
        
        # Heat delivered according to water energy balance
        Q_water = m_dot_sec * CP_WATER * (T_sec_supply - mid_T_return)

        # The heating supplied under this return temperature
        dT_ln = (T_sec_supply - mid_T_return) / np.log(
            (T_sec_supply - T_indoor) / (mid_T_return - T_indoor)
        )
        Q_max_radiator = Q_radiator_design * ((dT_ln / dT_ln_design) ** n_exponent)

        if Q_water > Q_max_radiator:
            low = mid_T_return
        else:
            high = mid_T_return

    T_sec_return_opt = (low + high) / 2.0
    Q_delivered_kw = m_dot_sec * CP_WATER * (T_sec_supply - T_sec_return_opt)

    if Q_delivered_kw > Q_demand_kw:
        Q_delivered_kw = Q_demand_kw
        T_sec_return_opt = T_sec_supply - Q_delivered_kw / (m_dot_sec * CP_WATER)

    return float(T_sec_return_opt), float(Q_delivered_kw)


def find_required_T_in(tank, m_dot, T_target, dt_sub, n_steps):
    """
    Solve for the tank inlet temperature (T_in) that will result in the
    final tank water temperature reaching T_target after n_steps.
    """
    if m_dot <= 0:
        return T_target

    T_water_init = np.copy(tank.T_water) if isinstance(tank.T_water, np.ndarray) else tank.T_water
    T_pcm_init = None
    if hasattr(tank, "pcm"):
        T_pcm_init = np.copy(tank.pcm.T) if isinstance(tank.pcm.T, np.ndarray) else tank.pcm.T

    def run_sim(T_in_candidate):
        if isinstance(tank.T_water, np.ndarray):
            tank.T_water = T_water_init.copy()
        else:
            tank.T_water = T_water_init
            
        if hasattr(tank, "pcm") and T_pcm_init is not None:
            if isinstance(tank.pcm.T, np.ndarray):
                tank.pcm.T = T_pcm_init.copy()
            else:
                tank.pcm.T = T_pcm_init
                
        for _ in range(n_steps):
            tank.step(m_dot=m_dot, T_in=T_in_candidate, dt=dt_sub)
        return tank.mean_temperature

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

    if isinstance(tank.T_water, np.ndarray):
        tank.T_water = T_water_init.copy()
    else:
        tank.T_water = T_water_init
        
    if hasattr(tank, "pcm") and T_pcm_init is not None:
        if isinstance(tank.pcm.T, np.ndarray):
            tank.pcm.T = T_pcm_init.copy()
        else:
            tank.pcm.T = T_pcm_init

    return T_in_opt


def run_scenario(
    scenario_name: str,
    hourly_demand: pd.DataFrame,
    supply_temps: pd.Series,
    weather: pd.DataFrame,
    daily_dispatch: pd.DataFrame,
):
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

    df = df[(df["date"] >= f"{YEAR}-01-01 00:00:00") & (df["date"] <= f"{YEAR}-03-31 23:00:00")].reset_index(drop=True)

    original_load_mw = df["network_load_mw"].values.copy()

    scale_info = compute_scale_factor(YEAR)
    scale_factor = scale_info["scale_factor"]

    if scenario_name == "Water Tank + DSM":
        tank = WaterTank(water_mass_kg=1000.0, initial_temperature=50.0, num_layers=NUM_LAYERS)
    elif scenario_name == "PCM Storage + DSM":
        tank = PCMStorageTank(water_mass_kg=500.0, pcm_mass_kg=578.0, initial_temperature=50.0, num_layers=NUM_LAYERS)
    else:
        tank = None

    substation = Substation(
        secondary_supply_temp=T_SEC_SUPPLY,
        max_primary_flow_kgs=M_DOT_PRI_MAX,
        T_approach=T_APPROACH,
    )

    modified_load_mw = []
    tes_charge_mw = []
    T_returns = []
    T_sec_returns = []
    T_sec_HEX_outs = []
    m_dots_primary = []
    T_water_temps = []
    T_pcm_temps = []
    Q_primary_kws = []
    T_sec_supply_targets = []
    m_dots_secondary = []
    T_water_layer_history = []
    T_pcm_layer_history = []

    n_steps = 60
    dt_sub = 60.0
    charging_state = True

    for idx, row in df.iterrows():
        T_supply = row["T_supply"]
        T_outdoor = row["temperature"]
        T_sec_supply_target = get_secondary_supply_temp(T_outdoor)
        T_sec_supply_targets.append(T_sec_supply_target)
        hour = row["date"].hour
        network_load_mw = row["network_load_mw"]

        # Building-level heat demand
        Q_demand_building = (network_load_mw * 1000.0) / scale_factor

        # determine the dynamic secondary flow rate
        if Q_demand_building > 0:
            m_dot_sec = Q_demand_building / (CP_WATER * DELTA_T_DESIGN)
            m_dot_sec = np.clip(m_dot_sec, M_DOT_SEC_MIN, M_DOT_SEC_MAX)
        else:
            m_dot_sec = 0.0

        if tank is None:
            # Baseline
            T_sec_return, _ = compute_realistic_secondary_side(Q_demand_building, T_sec_supply_target, m_dot_sec)

            dT_sec = max(0.0, T_sec_supply_target - T_sec_return)
            delta_T = T_supply - (T_sec_return + T_APPROACH)
            if delta_T > 0 and ETA_HEX > 0:
                m_dot_needed = (m_dot_sec * dT_sec) / (ETA_HEX * delta_T)
            else:
                m_dot_needed = 0.0

            m_dot_primary = min(m_dot_needed, substation.max_primary_flow)

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

            # Recalculate physical loop parameters using realistic physics response
            Q_delivered_kw = ETA_HEX * Q_primary_kw
            T_sec_return, _ = compute_realistic_secondary_side(Q_delivered_kw, T_sec_supply, m_dot_sec)

            T_water_val = np.nan
            T_pcm_val = np.nan
            T_sec_HEX_out = T_sec_supply

        else:
            # TES scenario
            T_water_top = tank.T_water[-1] if isinstance(tank.T_water, np.ndarray) else tank.T_water
            T_water_mean = tank.mean_temperature

            # Hysteretic control based on mean TES temperature
            if T_water_mean >= T_MAX_TES:
                charging_state = False
            elif T_water_mean <= T_MIN_TES:
                charging_state = True

            T_sec_supply = T_water_top
            T_sec_return, _ = compute_realistic_secondary_side(Q_demand_building, T_sec_supply, m_dot_sec)
            T_sec_inlet = T_sec_return

            if charging_state:
                m_dot_primary = M_DOT_PRI_MAX
                T_target_HEX = 75.0
            else:
                m_dot_primary = M_DOT_PRI_MIN
                T_target_HEX = T_sec_supply_target

            # Clip primary flow to substation's maximum capacity
            m_dot_primary = min(m_dot_primary, substation.max_primary_flow)

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

            T_out_sum = 0.0
            for _ in range(n_steps):
                T_out_sub = tank.step(m_dot=m_dot_sec, T_in=T_sec_supply_HEX, dt=dt_sub)
                T_out_sum += T_out_sub
            T_out_tank = T_out_sum / n_steps

            T_sec_supply = T_out_tank
            T_sec_return, _ = compute_realistic_secondary_side(Q_demand_building, T_sec_supply, m_dot_sec)

            T_water_val = tank.mean_temperature
            T_pcm_val = np.mean(tank.pcm.T) if hasattr(tank, "pcm") else np.nan
            T_sec_HEX_out = T_sec_supply_HEX

        # Record layer temperatures
        if tank is None:
            T_water_layers = np.full(NUM_LAYERS, np.nan)
            T_pcm_layers = np.full(NUM_LAYERS, np.nan)
        else:
            if isinstance(tank.T_water, np.ndarray):
                T_water_layers = np.copy(tank.T_water)
            else:
                T_water_layers = np.full(NUM_LAYERS, tank.T_water)
            
            if hasattr(tank, "pcm"):
                if isinstance(tank.pcm.T, np.ndarray):
                    T_pcm_layers = np.copy(tank.pcm.T)
                else:
                    T_pcm_layers = np.full(NUM_LAYERS, tank.pcm.T)
            else:
                T_pcm_layers = np.full(NUM_LAYERS, np.nan)

        T_water_layer_history.append(T_water_layers)
        T_pcm_layer_history.append(T_pcm_layers)

        modified_demand_network_mw = (Q_primary_kw * scale_factor) / 1000.0
        net_charge_network_mw = modified_demand_network_mw - (Q_demand_building * scale_factor / 1000.0)

        modified_load_mw.append(max(modified_demand_network_mw, 0.0))
        tes_charge_mw.append(net_charge_network_mw)
        T_returns.append(T_return_primary)
        T_sec_returns.append(T_sec_return)
        T_sec_HEX_outs.append(T_sec_HEX_out)
        m_dots_primary.append(m_dot_primary)
        T_water_temps.append(T_water_val)
        T_pcm_temps.append(T_pcm_val)
        Q_primary_kws.append(Q_primary_kw)
        m_dots_secondary.append(m_dot_sec)

    df["modified_load_mw"] = modified_load_mw
    df["tes_charge_mw"] = tes_charge_mw
    df["T_return_primary"] = T_returns
    df["T_secondary_return"] = T_sec_returns
    df["T_HEX_out"] = T_sec_HEX_outs
    df["m_dot_primary"] = m_dots_primary
    df["T_water"] = T_water_temps
    df["T_pcm"] = T_pcm_temps
    df["Q_primary_kw"] = Q_primary_kws
    df["T_sec_supply_target"] = T_sec_supply_targets
    df["m_dot_secondary"] = m_dots_secondary

    # Convert histories to numpy arrays
    T_water_layer_history = np.array(T_water_layer_history)
    T_pcm_layer_history = np.array(T_pcm_layer_history)

    # Save layers and mean temperature to dataframe
    df["T_water_mean"] = T_water_temps
    df["T_pcm_mean"] = T_pcm_temps
    for i in range(NUM_LAYERS):
        df[f"T_water_layer_{i}"] = T_water_layer_history[:, i]
        df[f"T_pcm_layer_{i}"] = T_pcm_layer_history[:, i]

    if tank is not None:
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

    if not dispatch.empty:
        for src in PRIORITY_SOURCES:
            if src in dispatch.columns:
                df[src] = dispatch[src].values[:len(df)]
            else:
                df[src] = 0.0
    else:
        for src in PRIORITY_SOURCES:
            df[src] = 0.0

    chp = CHPModel()
    fgc = FlueGasCondensationModel()

    eta_els = []
    fgc_recoveries = []

    for _, row in df.iterrows():
        eff = chp.compute_efficiency(row["T_return_primary"])
        eta_els.append(eff["eta_el"])

        fgc_kw = fgc.compute_recovery(
            row["T_return_primary"],
            row.get("Waste heat", 0) * 1000.0,
        )
        fgc_recoveries.append(fgc_kw / 1000.0)

    df["eta_el_chp"] = eta_els
    df["fgc_recovery_mw"] = fgc_recoveries
    df["scenario"] = scenario_name

    print(f"  Hours simulated: {len(df)}")
    print(f"  Demand — original peak: {original_load_mw.max():.1f} MW, "
          f"modified peak: {df['modified_load_mw'].max():.1f} MW")

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

    hourly_demand = load_and_scale_building_load(YEAR, diversity_factor=0.85)
    supply_temps = load_supply_temperature(YEAR)
    daily_dispatch = load_daily_dispatch(YEAR)

    weather = pd.read_csv(BASE_DIR / "DH production mix" / "weather.csv")
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather[weather["date"].dt.year == YEAR].copy()

    scale_info = compute_scale_factor(YEAR)
    print(f"\nNetwork scale factor: {scale_info['scale_factor']:.1f}")
    print(f"Network annual demand: {scale_info['network_annual_mwh']:,.0f} MWh")
    print(f"Peak network load: {hourly_demand['network_load_mw'].max():.1f} MW")

    baseline = run_scenario(
        "Baseline (no TES/DSM)",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
    )

    water_tank_res = run_scenario(
        "Water Tank + DSM",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
    )

    pcm_res = run_scenario(
        "PCM Storage + DSM",
        hourly_demand,
        supply_temps,
        weather,
        daily_dispatch,
    )

    baseline.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    water_tank_res.to_csv(OUTPUT_DIR / "water_tank_results.csv", index=False)
    pcm_res.to_csv(OUTPUT_DIR / "pcm_storage_results.csv", index=False)

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