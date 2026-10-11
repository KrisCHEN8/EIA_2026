import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from simulation.network_scaling import load_and_scale_building_load
from simulation.substation_model import Substation
from simulation.TES_model import WaterTank, PCMStorageTank, simulate_hour, CP_W, RHO_W
from simulation.chp_model import CHPModel, FlueGasCondensationModel
from simulation.hourly_dispatch import load_daily_dispatch, distribute_daily_to_hourly, redispatch_for_modified_load
from simulation.dsm_strategies import make_schedule

DATA_DIR = BASE_DIR / "DH production mix"
OUTPUT_DIR = BASE_DIR / "simulation" / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
YEAR = 2024
START = f"{YEAR}-01-01 00:00:00"
END = f"{YEAR}-04-30 23:00:00"

PRIORITY_SOURCES = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]

# Substation sizing (shared by all scenarios)
RAD_OVERSIZE = 1.1          # radiator design load = 1.1 x peak building demand
HEX_OVERSIZE = 1.5          # HEX design load = 1.5 x peak (headroom for charging)
LOSS_FRACTION = 0.0         # network data already include losses -> baseline matches data

# Tank
T_LOW = 40.0                # empty tank / reference temperature (degC)
T_CHARGE = 65.0             # charging setpoint at tank top (degC)
T_BOTTOM_CUTOFF = 45.0      # stop charging when tank bottom reaches this (degC)
NUM_LAYERS = 10
TANK_UA_LOSS = 2.0          # W/K
PCM_VOLUME_FRACTION = 0.5   # PCM tank: same outer volume, half filled with PCM containers
UA_PCM = 2000.0             # W/K
UA_PCM_MIN_FRACTION = 0.3

# Control
N_CHARGE_HOURS = 8          # per day
N_DISCHARGE_HOURS = 4       # per day
N_SUB = 6                   # substation/tank coupling steps per hour
CHARGE_MAX_RANK = 1         # production-aware: charge when marginal source is waste heat (0) or electricity (1)
DISCHARGE_MIN_RANK = 2      # production-aware: discharge when marginal source is biomass (2) or fossil (3)

# Scenario matrix
STORAGE_TYPES = ["water", "pcm"]
STRATEGIES = ["peak_shaving", "production_aware"]
TANK_SIZES_H = [2, 6]       # tank size in hours of mean building demand (water capacity T_LOW..T_CHARGE)
PENETRATIONS = [0.10, 0.25, 0.50]


# ---------------------------------------------------------------------------
#  Inputs
# ---------------------------------------------------------------------------
def heating_curve_secondary(t_outdoor):
    """Outdoor compensation curve for the radiator circuit (secondary supply setpoint)."""
    if t_outdoor <= -10.0:
        return 60.0
    elif t_outdoor <= 0.0:
        return 56.0
    elif t_outdoor <= 5.0:
        return 52.0
    return 40.0


def load_inputs():
    demand = load_and_scale_building_load(YEAR, anchor_daily=True)

    sup = pd.read_csv(DATA_DIR / "DH_supply_temp.csv")
    sup = pd.DataFrame({"date": pd.to_datetime(sup.iloc[:, 0]), "T_supply": sup["T_supply"]})

    weather = pd.read_csv(DATA_DIR / "weather.csv", parse_dates=["date"])

    df = demand.merge(sup, on="date", how="left").merge(weather, on="date", how="left")
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Warning: dropped {n_before - len(df)} hours with missing inputs")
    df["T_s_set"] = df["temperature"].apply(heating_curve_secondary)
    return df


# ---------------------------------------------------------------------------
#  Building level
# ---------------------------------------------------------------------------
def make_substation(peak_kw):
    return Substation(Q_design_kw=RAD_OVERSIZE * peak_kw,
                      Q_hex_design_kw=HEX_OVERSIZE * peak_kw,
                      loss_fraction=LOSS_FRACTION)


def make_tank(storage, size_h, mean_kw):
    """Size by water-tank capacity between T_LOW and T_CHARGE; PCM tank gets the SAME volume."""
    capacity_kwh = size_h * mean_kw
    volume_l = capacity_kwh * 3.6e6 / (RHO_W * CP_W * (T_CHARGE - T_LOW)) * 1000.0
    common = dict(initial_temperature=T_LOW, num_layers=NUM_LAYERS,
                  UA_loss=TANK_UA_LOSS, T_ref=T_LOW)
    if storage == "water":
        tank = WaterTank.from_volume(volume_l, **common)
    elif storage == "pcm":
        tank = PCMStorageTank.from_volume(volume_l, pcm_volume_fraction=PCM_VOLUME_FRACTION,
                                          UA_pcm=UA_PCM, UA_min_fraction=UA_PCM_MIN_FRACTION, **common)
    else:
        raise ValueError(storage)
    return tank, volume_l


def simulate_building(inputs, substation, tank=None, modes=None, Q_charge_kw=0.0, Q_discharge_kw=0.0):
    """Hour-by-hour building simulation. Returns an hourly DataFrame (building level, kW)."""
    rows = []
    for i, row in enumerate(inputs.itertuples(index=False)):
        mode = "direct" if modes is None else modes[i]
        Q_target = Q_charge_kw if mode == "charge" else Q_discharge_kw if mode == "discharge" else 0.0
        r = simulate_hour(substation, tank, mode, row.T_supply, row.T_s_set, row.building_demand_kw,
                          Q_tank_target_kw=Q_target, n_sub=N_SUB,
                          T_charge_set=T_CHARGE, T_bottom_cutoff=T_BOTTOM_CUTOFF)
        layers = r.pop("T_layers", None)
        r["mode_scheduled"] = mode
        if layers is not None:
            for k, t in enumerate(layers):
                r[f"T_layer_{k}"] = t
        rows.append(r)
    out = pd.DataFrame(rows)
    out.insert(0, "date", inputs["date"].values)
    out["Q_demand_kw"] = inputs["building_demand_kw"].values
    out["scale_factor"] = inputs["scale_factor"].values
    return out


def tank_energy_report(tank):
    if tank is None:
        return dict(tank_dE_kwh=0.0, tank_loss_kwh=0.0, tank_residual_kwh=0.0)
    return dict(
        tank_dE_kwh=(tank.stored_energy_J() - tank.E0) / 3.6e6,
        tank_loss_kwh=tank.E_loss / 3.6e6,
        tank_residual_kwh=tank.energy_residual() / 3.6e6,
    )


# ---------------------------------------------------------------------------
#  Network level
# ---------------------------------------------------------------------------
def aggregate_network(base, tes, p):
    """Mix (1-p) baseline buildings and p TES buildings, scaled to the network."""
    sf = base["scale_factor"].values
    net = pd.DataFrame({"date": base["date"].values})
    if tes is None or p == 0:
        q = base["Q_primary_kw"].values
        m_b, T_b = base["m_p_mean"].values, base["T_p_out"].values
        m_t, T_t = np.zeros_like(m_b), np.zeros_like(m_b)
        q_tank = np.zeros_like(q)
    else:
        q = (1 - p) * base["Q_primary_kw"].values + p * tes["Q_primary_kw"].values
        m_b, T_b = (1 - p) * base["m_p_mean"].values, base["T_p_out"].values
        m_t, T_t = p * tes["m_p_mean"].values, tes["T_p_out"].values
        q_tank = p * tes["Q_tank_kw"].values
    T_b = np.nan_to_num(T_b)
    T_t = np.nan_to_num(T_t)
    m = m_b + m_t
    net["network_load_mw"] = q * sf / 1000.0
    net["tes_charge_mw"] = q_tank * sf / 1000.0
    net["m_flow_network_kgs"] = m * sf
    with np.errstate(invalid="ignore", divide="ignore"):
        net["T_return_network"] = np.where(m > 0, (m_b * T_b + m_t * T_t) / m, np.nan)
    return net


def add_dispatch_and_chp(net, baseline_load_mw, daily_dispatch, chp, fgc, is_baseline):
    dates = pd.DatetimeIndex(net["date"])
    if is_baseline:
        disp = distribute_daily_to_hourly(daily_dispatch, net[["date", "network_load_mw"]])
    else:
        disp = redispatch_for_modified_load(daily_dispatch, baseline_load_mw,
                                            net["network_load_mw"].values, dates)
    disp = disp.set_index("date").reindex(dates)
    for src in PRIORITY_SOURCES:
        net[src] = disp[src].fillna(0.0).values

    T_ret = net["T_return_network"].ffill().bfill().values
    net["eta_el_chp"] = [chp.compute_efficiency(t)["eta_el"] for t in T_ret]
    net["eta_th_chp"] = [chp.compute_efficiency(t)["eta_th"] for t in T_ret]
    net["fgc_recovery_mw"] = [fgc.compute_recovery(t, w * 1000.0) / 1000.0
                              for t, w in zip(T_ret, net["Waste heat"].values)]
    return net


def flow_weighted(T, m):
    T, m = np.asarray(T, float), np.asarray(m, float)
    ok = np.isfinite(T) & (m > 0)
    return float((T[ok] * m[ok]).sum() / m[ok].sum()) if ok.any() else np.nan


def summarize(tag, cfg, net, base_net, bld, base_bld, p, tank_rep, p90_threshold):
    """One summary row. Building-level quantities are scaled to network level where marked."""
    sf = bld["scale_factor"].values
    load = net["network_load_mw"]
    row = dict(tag=tag, **cfg, penetration=p)
    row.update({
        "Peak demand (MW)": load.max(),
        "Mean demand (MW)": load.mean(),
        "Heat supplied (GWh)": load.sum() / 1000.0,
        "Load factor": load.mean() / load.max(),
        "Hours above baseline P90": int((load > p90_threshold).sum()),
        "T_return network (degC)": flow_weighted(net["T_return_network"], net["m_flow_network_kgs"]),
        "Mean eta_el CHP": net["eta_el_chp"].mean(),
        "FGC recovery (GWh)": net["fgc_recovery_mw"].sum() / 1000.0,
    })
    for src in PRIORITY_SOURCES:
        row[f"{src} (GWh)"] = net[src].sum() / 1000.0

    # Extra CHP electricity vs baseline, hour by hour: fuel_h * (eta_el - eta_el_base)
    fuel = base_net["Waste heat"].values / base_net["eta_th_chp"].values
    row["Extra CHP electricity (MWh)"] = float(np.sum(fuel * (net["eta_el_chp"].values - base_net["eta_el_chp"].values)))

    # Per-mode return temperature of TES buildings (building level)
    for mode in ["charge", "discharge", "direct"]:
        mask = (bld["mode_scheduled"] == mode).values
        row[f"T_p_out TES bldg, {mode} hours (degC)"] = flow_weighted(bld["T_p_out"][mask], bld["m_p_mean"][mask]) if mask.any() else np.nan

    # Storage operation, scaled to the network share p
    charged = bld["Q_tank_kw"].clip(lower=0) * sf
    discharged = (-bld["Q_tank_kw"]).clip(lower=0) * sf
    row["Energy charged (MWh)"] = p * charged.sum() / 1000.0
    row["Energy discharged (MWh)"] = p * discharged.sum() / 1000.0
    row["Tank dE end-start (MWh)"] = p * tank_rep["tank_dE_kwh"] * float(np.mean(sf)) / 1000.0
    row["Tank losses (MWh)"] = p * tank_rep["tank_loss_kwh"] * float(np.mean(sf)) / 1000.0
    row["Tank residual (kWh, 1 bldg)"] = tank_rep["tank_residual_kwh"]
    row["Coupling error (kWh, 1 bldg)"] = float((bld["Q_tank_kw"] - bld["Q_tank_substation_kw"]).sum())
    row["Unmet demand (MWh)"] = float(((1 - p) * base_bld["unmet_kwh"] + p * bld["unmet_kwh"]).mul(sf).sum() / 1000.0)
    row["Saturated hours (TES bldg)"] = float(bld["frac_saturated"].sum())
    # Heat supplied corrected for energy left in the tanks at the end
    row["Heat supplied, dE-corrected (GWh)"] = row["Heat supplied (GWh)"] - row["Tank dE end-start (MWh)"] / 1000.0
    return row


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Loading inputs...")
    inputs = load_inputs()
    daily_dispatch = load_daily_dispatch(YEAR)
    peak_kw = inputs["building_demand_kw"].max()
    mean_kw = inputs["building_demand_kw"].mean()
    print(f"  {len(inputs)} hours, {inputs['date'].min()} - {inputs['date'].max()}")
    print(f"  Building demand: peak {peak_kw:.1f} kW, mean {mean_kw:.1f} kW")

    substation = make_substation(peak_kw)
    chp, fgc = CHPModel(), FlueGasCondensationModel()

    # ---- Baseline -------------------------------------------------------
    print("\nBaseline building...")
    base_bld = simulate_building(inputs, substation)
    base_bld.to_csv(OUTPUT_DIR / "building_baseline.csv", index=False)
    base_net = add_dispatch_and_chp(aggregate_network(base_bld, None, 0.0), None,
                                    daily_dispatch, chp, fgc, is_baseline=True)
    base_net.to_csv(OUTPUT_DIR / "network_baseline.csv", index=False)
    baseline_load = base_net["network_load_mw"].values
    p90 = np.percentile(baseline_load, 90)

    summary = [summarize("baseline", dict(storage="none", strategy="none", size_h=0, volume_l=0.0),
                         base_net, base_net, base_bld, base_bld, 0.0, tank_energy_report(None), p90)]

    # ---- TES scenarios --------------------------------------------------
    for storage in STORAGE_TYPES:
        for strategy in STRATEGIES:
            modes, _ = make_schedule(strategy, inputs["date"], baseline_load, daily_dispatch,
                                     N_CHARGE_HOURS, N_DISCHARGE_HOURS,
                                     CHARGE_MAX_RANK, DISCHARGE_MIN_RANK)
            n_mode = pd.Series(modes).value_counts().to_dict()
            print(f"\nSchedule {strategy}: {n_mode.get('charge', 0)} charge h, "
                  f"{n_mode.get('discharge', 0)} discharge h")
            for size_h in TANK_SIZES_H:
                tag = f"{storage}_{strategy}_{size_h}h"
                tank, volume_l = make_tank(storage, size_h, mean_kw)
                cap_kwh = tank.capacity_kwh(T_LOW, T_CHARGE)
                print(f"\n{tag}: {volume_l:,.0f} L, capacity {cap_kwh:.1f} kWh")
                bld = simulate_building(inputs, substation, tank, modes,
                                        Q_charge_kw=cap_kwh / N_CHARGE_HOURS,
                                        Q_discharge_kw=cap_kwh / N_DISCHARGE_HOURS)
                bld["storage"], bld["strategy"], bld["size_h"] = storage, strategy, size_h
                bld.to_csv(OUTPUT_DIR / f"building_{tag}.csv", index=False)
                rep = tank_energy_report(tank)
                print(f"  tank dE {rep['tank_dE_kwh']:.2f} kWh, losses {rep['tank_loss_kwh']:.1f} kWh, "
                      f"residual {rep['tank_residual_kwh']:.2e} kWh")

                cfg = dict(storage=storage, strategy=strategy, size_h=size_h, volume_l=volume_l)
                for p in PENETRATIONS:
                    net = aggregate_network(base_bld, bld, p)
                    net = add_dispatch_and_chp(net, baseline_load, daily_dispatch, chp, fgc, is_baseline=False)
                    net.to_csv(OUTPUT_DIR / f"network_{tag}_p{int(round(p * 100))}.csv", index=False)
                    summary.append(summarize(tag, cfg, net, base_net, bld, base_bld, p, rep, p90))

    summary = pd.DataFrame(summary)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    base = summary.iloc[0]
    cols = ["tag", "penetration", "Peak demand (MW)", "T_return network (degC)",
            "Fossil fuel (GWh)", "Extra CHP electricity (MWh)", "Unmet demand (MWh)"]
    print("\n" + "=" * 90)
    print(summary[cols].to_string(index=False, float_format="{:.3f}".format))
    print("=" * 90)
    print(f"Baseline peak {base['Peak demand (MW)']:.1f} MW, T_return {base['T_return network (degC)']:.2f} degC")
    print(f"Results saved to {OUTPUT_DIR}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
