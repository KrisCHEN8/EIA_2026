"""
Scenario Comparison & Visualization
====================================
Generates publication-quality figures comparing the three scenarios:
    1. Baseline (no TES/DSM)
    2. TES only
    3. TES + DSM

Figures produced:
    1. Load duration curves (all 3 scenarios overlaid)
    2. Weekly load profile comparison (winter peak week)
    3. Return temperature distributions (histogram)
    4. Monthly production mix (stacked bar, per scenario)
    5. CHP efficiency time series
    6. Summary KPI bar charts
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "simulation" / "results"
FIGS_DIR = BASE_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

PRIORITY_SOURCES = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]
SOURCE_COLORS = {
    "Waste heat": "#2196F3",
    "Electricity": "#FF9800",
    "Biomass": "#4CAF50",
    "Fossil fuel": "#F44336",
}
SCENARIO_COLORS = {
    "Baseline (no TES/DSM)": "#E57373",
    "TES only": "#64B5F6",
    "TES + DSM": "#81C784",
}
SCENARIO_SHORT = {
    "Baseline (no TES/DSM)": "Baseline",
    "TES only": "TES",
    "TES + DSM": "TES+DSM",
}


def load_results():
    """Load all scenario results."""
    scenarios = {}
    for fname, label in [
        ("baseline_results.csv", "Baseline (no TES/DSM)"),
        ("tes_results.csv", "TES only"),
        ("dsm_results.csv", "TES + DSM"),
    ]:
        path = RESULTS_DIR / fname
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            scenarios[label] = df
        else:
            print(f"Warning: {path} not found, skipping {label}")
    return scenarios


def plot_load_duration_curves(scenarios: dict):
    """Load duration curve: sorted demand from highest to lowest."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for name, df in scenarios.items():
        sorted_load = np.sort(df["modified_load_mw"].values)[::-1]
        hours = np.arange(1, len(sorted_load) + 1)
        ax.plot(hours, sorted_load,
                color=SCENARIO_COLORS.get(name, "white"),
                linewidth=2, label=SCENARIO_SHORT.get(name, name),
                alpha=0.9)

    ax.set_xlabel("Hours (sorted)", color="white", fontsize=13)
    ax.set_ylabel("Network Demand (MW)", color="white", fontsize=13)
    ax.set_title("Load Duration Curves — Scenario Comparison", color="white", fontsize=14, pad=12)
    ax.legend(fontsize=12, labelcolor="white", facecolor="#1a1a2e", framealpha=0.5)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")
    ax.grid(True, color="#333355", linewidth=0.6, alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "load_duration_curves.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ✓ Load duration curves")


def plot_weekly_profile(scenarios: dict):
    """Weekly load profile during winter peak week."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for name, df in scenarios.items():
        # Find the peak week (week with highest total demand)
        df_copy = df.copy()
        df_copy["week"] = df_copy["date"].dt.isocalendar().week.astype(int)
        weekly_sum = df_copy.groupby("week")["modified_load_mw"].sum()
        peak_week = weekly_sum.idxmax()
        week_data = df_copy[df_copy["week"] == peak_week].copy()

        ax.plot(range(len(week_data)), week_data["modified_load_mw"].values,
                color=SCENARIO_COLORS.get(name, "white"),
                linewidth=2, label=SCENARIO_SHORT.get(name, name),
                alpha=0.9)

    ax.set_xlabel("Hour of week", color="white", fontsize=13)
    ax.set_ylabel("Network Demand (MW)", color="white", fontsize=13)
    ax.set_title(f"Peak Week Load Profile — Scenario Comparison", color="white", fontsize=14, pad=12)
    ax.legend(fontsize=12, labelcolor="white", facecolor="#1a1a2e", framealpha=0.5)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")
    ax.grid(True, color="#333355", linewidth=0.6, alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "weekly_peak_profile.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ✓ Weekly peak profile")


def plot_return_temperature_distribution(scenarios: dict):
    """Histogram of return temperatures."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bins = np.linspace(30, 65, 50)

    for name, df in scenarios.items():
        ax.hist(df["T_return_primary"].values, bins=bins,
                color=SCENARIO_COLORS.get(name, "white"),
                alpha=0.5, label=SCENARIO_SHORT.get(name, name),
                edgecolor="white", linewidth=0.3)

    # Add vertical lines for means
    for name, df in scenarios.items():
        mean_t = df["T_return_primary"].mean()
        ax.axvline(mean_t, color=SCENARIO_COLORS.get(name, "white"),
                   linestyle="--", linewidth=2, alpha=0.8)
        ax.text(mean_t + 0.3, ax.get_ylim()[1] * 0.9,
                f"{SCENARIO_SHORT.get(name, name)}: {mean_t:.1f}°C",
                color=SCENARIO_COLORS.get(name, "white"), fontsize=10,
                rotation=90, va="top")

    ax.set_xlabel("Return Temperature (°C)", color="white", fontsize=13)
    ax.set_ylabel("Frequency (hours)", color="white", fontsize=13)
    ax.set_title("Primary Return Temperature Distribution", color="white", fontsize=14, pad=12)
    ax.legend(fontsize=12, labelcolor="white", facecolor="#1a1a2e", framealpha=0.5)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")
    ax.grid(True, axis="y", color="#333355", linewidth=0.6, alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "return_temp_distribution.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ✓ Return temperature distribution")


def plot_monthly_production_mix(scenarios: dict):
    """Monthly production mix stacked bars for each scenario."""
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(14, 5 * n_scenarios), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")

    if n_scenarios == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, scenarios.items()):
        ax.set_facecolor("#16213e")
        df_copy = df.copy()
        df_copy["month"] = df_copy["date"].dt.month
        monthly = df_copy.groupby("month")[PRIORITY_SOURCES].sum() / 1000  # MWh → GWh

        bottom = np.zeros(12)
        x = np.arange(1, 13)

        for src in PRIORITY_SOURCES:
            vals = [monthly.loc[m, src] if m in monthly.index else 0 for m in x]
            ax.bar(x, vals, bottom=bottom, color=SOURCE_COLORS[src],
                   label=src, alpha=0.85, edgecolor="white", linewidth=0.3)
            bottom += vals

        ax.set_ylabel("Energy (GWh)", color="white", fontsize=12)
        ax.set_title(f"{SCENARIO_SHORT.get(name, name)}", color="white", fontsize=13, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(month_labels, color="white")
        ax.tick_params(colors="white")
        ax.legend(fontsize=10, labelcolor="white", facecolor="#1a1a2e",
                  framealpha=0.5, loc="upper right", ncol=2)
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.grid(True, axis="y", color="#333355", linewidth=0.6, alpha=0.5)

    plt.suptitle("Monthly Production Mix by Scenario", color="white",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "monthly_production_mix.png", dpi=200,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Monthly production mix")


def plot_chp_efficiency(scenarios: dict):
    """Monthly average CHP electrical efficiency comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    width = 0.25
    x = np.arange(1, 13)

    for i, (name, df) in enumerate(scenarios.items()):
        df_copy = df.copy()
        df_copy["month"] = df_copy["date"].dt.month
        monthly_eta = df_copy.groupby("month")["eta_el_chp"].mean()
        vals = [monthly_eta.get(m, 0) * 100 for m in x]  # Convert to %

        ax.bar(x + (i - 1) * width, vals, width=width,
               color=SCENARIO_COLORS.get(name, "white"),
               label=SCENARIO_SHORT.get(name, name),
               alpha=0.85, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Month", color="white", fontsize=13)
    ax.set_ylabel("CHP Electrical Efficiency (%)", color="white", fontsize=13)
    ax.set_title("Monthly CHP Efficiency by Scenario", color="white", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=12, labelcolor="white", facecolor="#1a1a2e", framealpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444466")
    ax.grid(True, axis="y", color="#333355", linewidth=0.6, alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "chp_efficiency_monthly.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ✓ CHP efficiency")


def plot_kpi_summary(scenarios: dict):
    """Bar chart comparing key KPIs across scenarios."""
    kpis = {}
    for name, df in scenarios.items():
        short = SCENARIO_SHORT.get(name, name)
        kpis[short] = {
            "Peak Demand\n(MW)": df["modified_load_mw"].max(),
            "Mean T_return\n(°C)": df["T_return_primary"].mean(),
            "Mean η_el\n(%)": df["eta_el_chp"].mean() * 100,
            "Fossil Fuel\n(GWh)": df["Fossil fuel"].sum() / 1000,
            "Load Factor\n(%)": df["modified_load_mw"].mean() / df["modified_load_mw"].max() * 100,
        }

    kpi_names = list(list(kpis.values())[0].keys())
    n_kpis = len(kpi_names)
    n_scenarios = len(kpis)

    fig, axes = plt.subplots(1, n_kpis, figsize=(4 * n_kpis, 5))
    fig.patch.set_facecolor("#1a1a2e")

    scenario_names = list(kpis.keys())
    colors = [SCENARIO_COLORS.get(k, "#888") for k in scenarios.keys()]

    for ax, kpi in zip(axes, kpi_names):
        ax.set_facecolor("#16213e")
        vals = [kpis[s][kpi] for s in scenario_names]
        bars = ax.bar(scenario_names, vals, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", color="white",
                    fontsize=10, fontweight="bold")

        ax.set_title(kpi, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#444466")
        ax.grid(True, axis="y", color="#333355", linewidth=0.6, alpha=0.5)

    plt.suptitle("Key Performance Indicators — Scenario Comparison",
                 color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "kpi_summary.png", dpi=200,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ KPI summary")


def main():
    print("Loading scenario results...")
    scenarios = load_results()

    if not scenarios:
        print("No results found! Run run_simulation.py first.")
        return

    print(f"Found {len(scenarios)} scenarios: {list(scenarios.keys())}")
    print("\nGenerating figures...")

    plot_load_duration_curves(scenarios)
    plot_weekly_profile(scenarios)
    plot_return_temperature_distribution(scenarios)
    plot_monthly_production_mix(scenarios)
    plot_chp_efficiency(scenarios)
    plot_kpi_summary(scenarios)

    print(f"\nAll figures saved to: {FIGS_DIR}")


if __name__ == "__main__":
    main()
