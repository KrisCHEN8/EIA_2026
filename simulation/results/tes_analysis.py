import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wt  = pd.read_csv("/home/yangzhe/myprojects/EIA_2026/simulation/results/water_tank_results.csv")
pcm = pd.read_csv("/home/yangzhe/myprojects/EIA_2026/simulation/results/pcm_storage_results.csv")
bl  = pd.read_csv("/home/yangzhe/myprojects/EIA_2026/simulation/results/baseline_results.csv")

for df in [wt, pcm, bl]:
    df["date"] = pd.to_datetime(df["date"])

# ---- TES sizing parameters (from run_simulation.py) ----
WT_MASS_KG   = 1000.0
PCM_WATER_KG = 500.0
PCM_PCM_KG   = 578.0
PCM_LATENT   = 150_000  # J/kg (55–70 °C window)
CP_WATER     = 4184.0   # J/kg·K
SCALE        = 8263.4   # building → network

# Nominal storage capacities
E_wt_kwh_5K  = WT_MASS_KG * CP_WATER * 5.0  / 3_600_000
E_wt_kwh_15K = WT_MASS_KG * CP_WATER * 15.0 / 3_600_000
E_pcm_water  = PCM_WATER_KG * CP_WATER * 5.0 / 3_600_000
E_pcm_latent = PCM_PCM_KG * PCM_LATENT      / 3_600_000
E_pcm_total  = E_pcm_water + E_pcm_latent

print("=== TES Nominal Storage Capacity (single-building) ===")
print(f"  Water Tank (1000 kg, ΔT=5 °C):   {E_wt_kwh_5K:.3f} kWh")
print(f"  Water Tank (1000 kg, ΔT=15 °C):  {E_wt_kwh_15K:.3f} kWh")
print(f"  PCM Tank – water part (ΔT=5 °C): {E_pcm_water:.3f} kWh")
print(f"  PCM Tank – latent:                {E_pcm_latent:.3f} kWh")
print(f"  PCM Tank – total:                 {E_pcm_total:.3f} kWh")

print(f"\n=== Scaled to Full Network (×{SCALE:.0f} buildings) ===")
print(f"  Water Tank (ΔT=5K):   {E_wt_kwh_5K*SCALE/1000:.2f} MWh")
print(f"  Water Tank (ΔT=15K):  {E_wt_kwh_15K*SCALE/1000:.2f} MWh")
print(f"  PCM Tank (total):      {E_pcm_total*SCALE/1000:.2f} MWh")

# ---- Demand shape ----
peak_mw  = bl["network_load_mw"].max()
mean_mw  = bl["network_load_mw"].mean()
cap_90   = peak_mw * 0.90
excess_mwh = ((bl["network_load_mw"] - cap_90).clip(lower=0)).sum()
print(f"\n=== Demand Profile (Jan–Mar 2024) ===")
print(f"  Original peak:              {peak_mw:.1f} MW")
print(f"  Mean demand:                {mean_mw:.1f} MW")
print(f"  90%-peak cap:               {cap_90:.1f} MW")
print(f"  Hours above cap:            {(bl['network_load_mw']>cap_90).sum()}")
print(f"  Cumulative excess above cap:{excess_mwh:.1f} MWh  ← energy TES must absorb/supply")

# ---- Peak shaving check ----
print(f"\n=== Peak Shaving Performance ===")
for name, df in [("Water Tank", wt), ("PCM Storage", pcm)]:
    peak_mod  = df["modified_load_mw"].max()
    peak_red  = peak_mw - peak_mod
    h_above   = (df["modified_load_mw"] > cap_90).sum()
    print(f"  {name}: modified peak = {peak_mod:.1f} MW  "
          f"(reduction {peak_red:.1f} MW / {peak_red/peak_mw*100:.1f}%)")
    print(f"           hours still above 90%-cap: {h_above}")

# ---- Charge/discharge diagnostics ----
for label, df in [("Water Tank", wt), ("PCM Storage", pcm)]:
    df["tes_charge_mw"] = pd.to_numeric(df["tes_charge_mw"], errors="coerce")
    c  = (df["tes_charge_mw"] > 0.1).sum()
    d  = (df["tes_charge_mw"] < -0.1).sum()
    s  = len(df) - c - d
    ec = df.loc[df["tes_charge_mw"] > 0, "tes_charge_mw"].sum()
    ed = df.loc[df["tes_charge_mw"] < 0, "tes_charge_mw"].sum()
    print(f"\n=== {label} – Charge/Discharge ===")
    print(f"  Charging hours:     {c}")
    print(f"  Discharging hours:  {d}")
    print(f"  Standby hours:      {s}")
    print(f"  Energy charged:    +{ec:.1f} MWh")
    print(f"  Energy discharged:  {ed:.1f} MWh")
    print(f"  Round-trip efficiency: {abs(ed)/ec*100:.1f}%  (discharged / charged)")
    print(f"  T_water: {df['T_water'].min():.1f} – {df['T_water'].max():.1f} °C  "
          f"(mean {df['T_water'].mean():.1f} °C)")
    if "T_pcm" in df.columns and df["T_pcm"].notna().any():
        print(f"  T_pcm:   {df['T_pcm'].dropna().min():.1f} – {df['T_pcm'].dropna().max():.1f} °C")

# ---- Temperature adequacy ----
print(f"\n=== Temperature Adequacy (can tank deliver ≥ 50 °C?) ===")
for name, df in [("Water Tank", wt), ("PCM Storage", pcm)]:
    lo50 = (df["T_water"] < 50.0).sum()
    lo45 = (df["T_water"] < 45.0).sum()
    lo40 = (df["T_water"] < 40.0).sum()
    print(f"  {name}:  T_water < 50°C: {lo50}h  |  <45°C: {lo45}h  |  <40°C: {lo40}h")

# ---- Demand shift per hour ----
print(f"\n=== Per-hour Demand Shift vs Baseline ===")
for name, df in [("Water Tank", wt), ("PCM Storage", pcm)]:
    shift = bl["network_load_mw"].values - df["modified_load_mw"].values
    print(f"  {name}: max_shift_up={shift.max():.2f} MW  max_shift_down={shift.min():.2f} MW  "
          f"mean|shift|={np.abs(shift).mean():.3f} MW")

# ---- Key sizing ratio: storage capacity vs shiftable load ----
print(f"\n=== Sizing Assessment ===")
# How many hours of mean demand can each TES supply?
mean_building_kw = (mean_mw * 1000) / SCALE   # kW at building level
print(f"  Mean building-level demand: {mean_building_kw:.3f} kW")
print(f"  Water Tank autonomy (ΔT=5K):   {E_wt_kwh_5K/mean_building_kw*60:.0f} min")
print(f"  Water Tank autonomy (ΔT=15K):  {E_wt_kwh_15K/mean_building_kw*60:.0f} min")
print(f"  PCM Tank autonomy (total):     {E_pcm_total/mean_building_kw*60:.0f} min")

# Network-level: MWh storage vs excess load above 90%-cap
print(f"  Network WT capacity (ΔT=15K): {E_wt_kwh_15K*SCALE/1000:.2f} MWh  "
      f"vs cumulative excess: {excess_mwh:.1f} MWh")
print(f"  Ratio (capacity / excess):    {E_wt_kwh_15K*SCALE/1000/max(excess_mwh,0.001)*100:.1f}%")
print(f"  Network PCM capacity (total): {E_pcm_total*SCALE/1000:.2f} MWh  "
      f"vs cumulative excess: {excess_mwh:.1f} MWh")
print(f"  Ratio (capacity / excess):    {E_pcm_total*SCALE/1000/max(excess_mwh,0.001)*100:.1f}%")

# ---- Plotting Secondary Supply & Return Temperatures ----
print(f"\n=== Generating Temperature Plots ===")

# Create subplots comparing the three scenarios
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=True)

# Select a week in January to show hourly variation clearly (e.g. Jan 15 to Jan 22)
start_date = "2024-01-15"
end_date = "2024-01-22"

bl_week = bl[(bl["date"] >= start_date) & (bl["date"] < end_date)]
wt_week = wt[(wt["date"] >= start_date) & (wt["date"] < end_date)]
pcm_week = pcm[(pcm["date"] >= start_date) & (pcm["date"] < end_date)]

# Plotting function for clean code
def plot_scenario(ax, df_week, title, show_tank_water=False, show_pcm=False):
    if "T_HEX_out" in df_week.columns:
        ax.plot(df_week["date"], df_week["T_HEX_out"], color="#e74c3c", label="HEX Outlet (Secondary)", linewidth=1.5)
    else:
        ax.plot(df_week["date"], df_week["T_secondary_supply"], color="#e74c3c", label="Secondary Supply", linewidth=1.5)
    ax.plot(df_week["date"], df_week["T_secondary_return"], color="#3498db", label="Secondary Return", linewidth=1.5)
    if show_tank_water and "T_water" in df_week.columns:
        ax.plot(df_week["date"], df_week["T_water"], color="#2ecc71", label="Tank Water Temp", linestyle="--", linewidth=1.5)
    if show_pcm and "T_pcm" in df_week.columns:
        ax.plot(df_week["date"], df_week["T_pcm"], color="#9b59b6", label="PCM Temp", linestyle=":", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

plot_scenario(axes[0], bl_week, "Baseline (no TES/DSM)")
plot_scenario(axes[1], wt_week, "Water Tank + DSM", show_tank_water=True)
plot_scenario(axes[2], pcm_week, "PCM Storage + DSM", show_tank_water=True, show_pcm=True)

axes[2].set_xlabel("Date")
plt.suptitle(f"HEX Outlet & Return Temperature Comparison ({start_date} to {end_date})", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

output_path = "/home/yangzhe/myprojects/EIA_2026/simulation/results/secondary_temps_weekly.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"  Saved weekly plot to: {output_path}")
plt.close()

# Also save a seasonal view using daily averages
bl_daily = bl.groupby(bl["date"].dt.date).mean(numeric_only=True).reset_index()
wt_daily = wt.groupby(wt["date"].dt.date).mean(numeric_only=True).reset_index()
pcm_daily = pcm.groupby(pcm["date"].dt.date).mean(numeric_only=True).reset_index()

bl_daily["date"] = pd.to_datetime(bl_daily["date"])
wt_daily["date"] = pd.to_datetime(wt_daily["date"])
pcm_daily["date"] = pd.to_datetime(pcm_daily["date"])

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=True)

plot_scenario(axes[0], bl_daily, "Baseline (no TES/DSM) - Daily Average")
plot_scenario(axes[1], wt_daily, "Water Tank + DSM - Daily Average", show_tank_water=True)
plot_scenario(axes[2], pcm_daily, "PCM Storage + DSM - Daily Average", show_tank_water=True, show_pcm=True)

axes[2].set_xlabel("Date")
plt.suptitle("HEX Outlet & Return Temperature Comparison (Daily Average Jan-Mar)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

output_path_seasonal = "/home/yangzhe/myprojects/EIA_2026/simulation/results/secondary_temps_seasonal.png"
plt.savefig(output_path_seasonal, dpi=150, bbox_inches="tight")
print(f"  Saved seasonal plot to: {output_path_seasonal}")
plt.close()
