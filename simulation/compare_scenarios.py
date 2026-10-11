import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "simulation" / "results"
FIGS_DIR = BASE_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

PRIORITY_SOURCES = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]

# Palette: validated categorical slots in fixed order (light surface)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BASELINE_COLOR = "#52514e"

COMBOS = [("water", "peak_shaving"), ("water", "production_aware"),
          ("pcm", "peak_shaving"), ("pcm", "production_aware")]
COMBO_COLOR = dict(zip(COMBOS, SLOTS))
COMBO_LABEL = {
    ("water", "peak_shaving"): "Water, peak shaving",
    ("water", "production_aware"): "Water, production-aware",
    ("pcm", "peak_shaving"): "PCM, peak shaving",
    ("pcm", "production_aware"): "PCM, production-aware",
}
SOURCE_COLOR = dict(zip(PRIORITY_SOURCES, SLOTS))
MODE_FILL = {"charge": "#2a78d6", "discharge": "#eb6834"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------
def tag_of(storage, strategy, size_h):
    return f"{storage}_{strategy}_{int(size_h)}h"


def load_summary():
    path = RESULTS_DIR / "summary.csv"
    if not path.exists():
        raise SystemExit("summary.csv not found - run run_simulation.py first.")
    return pd.read_csv(path)


def load_network(tag, p=None):
    name = "network_baseline.csv" if tag == "baseline" else f"network_{tag}_p{int(round(p * 100))}.csv"
    path = RESULTS_DIR / name
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else None


def load_building(tag):
    path = RESULTS_DIR / f"building_{tag}.csv"
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else None


def peak_week(base_net):
    """Start/end of the 7-day window with the highest baseline network energy."""
    s = base_net.set_index("date")["network_load_mw"].rolling("7D").sum()
    end = s.idxmax()
    return end - pd.Timedelta(days=7), end


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS_DIR / name, dpi=200)
    plt.close(fig)
    print(f"  saved figs/{name}")


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------
def plot_penetration(summary):
    """Small multiples: effect vs penetration, one line per storage x strategy x size."""
    base = summary[summary["tag"] == "baseline"].iloc[0]
    tes = summary[summary["tag"] != "baseline"].copy()
    tes["Peak reduction (%)"] = (1 - tes["Peak demand (MW)"] / base["Peak demand (MW)"]) * 100
    tes["Fossil fuel change (GWh)"] = tes["Fossil fuel (GWh)"] - base["Fossil fuel (GWh)"]
    tes["Return temperature change (K)"] = tes["T_return network (degC)"] - base["T_return network (degC)"]
    metrics = ["Peak reduction (%)", "Fossil fuel change (GWh)", "Return temperature change (K)"]

    sizes = sorted(tes["size_h"].unique())
    styles = {s: ls for s, ls in zip(sizes, ["--", "-", ":", "-."])}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric in zip(axes, metrics):
        for combo in COMBOS:
            for size in sizes:
                d = tes[(tes["storage"] == combo[0]) & (tes["strategy"] == combo[1]) & (tes["size_h"] == size)]
                if d.empty:
                    continue
                d = d.sort_values("penetration")
                ax.plot(d["penetration"] * 100, d[metric], color=COMBO_COLOR[combo],
                        linestyle=styles[size], marker="o", markersize=5,
                        label=f"{COMBO_LABEL[combo]}, {size} h")
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_title(metric)
        ax.set_xlabel("Penetration (% of buildings with TES)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.25)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIGS_DIR / "penetration_effects.png", dpi=200)
    plt.close(fig)
    print("  saved figs/penetration_effects.png")


def plot_load_duration(base_net, nets, p, size):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(np.sort(base_net["network_load_mw"].values)[::-1], color=BASELINE_COLOR, label="Baseline")
    for combo, net in nets.items():
        ax.plot(np.sort(net["network_load_mw"].values)[::-1], color=COMBO_COLOR[combo], label=COMBO_LABEL[combo])
    ax.set_xlabel("Hours (sorted)")
    ax.set_ylabel("Network demand (MW)")
    ax.set_title(f"Load duration curves, {size} h tanks, {p:.0%} penetration")
    ax.legend()
    save(fig, "load_duration_curves.png")


def plot_peak_week(base_net, nets, p, size):
    t0, t1 = peak_week(base_net)
    fig, ax = plt.subplots(figsize=(13, 5))
    w = (base_net["date"] >= t0) & (base_net["date"] < t1)
    ax.plot(base_net.loc[w, "date"], base_net.loc[w, "network_load_mw"], color=BASELINE_COLOR, label="Baseline")
    for combo, net in nets.items():
        w = (net["date"] >= t0) & (net["date"] < t1)
        ax.plot(net.loc[w, "date"], net.loc[w, "network_load_mw"], color=COMBO_COLOR[combo],
                label=COMBO_LABEL[combo], linewidth=1.6)
    ax.set_ylabel("Network demand (MW)")
    ax.set_title(f"Peak week, {size} h tanks, {p:.0%} penetration")
    ax.legend(ncol=5, fontsize=9)
    save(fig, "peak_week_load.png")


def plot_return_by_mode(summary, size):
    """Flow-weighted primary return of a TES building in charge / direct / discharge hours."""
    base = summary[summary["tag"] == "baseline"].iloc[0]
    tes = summary[(summary["tag"] != "baseline") & (summary["size_h"] == size)]
    tes = tes.drop_duplicates(subset=["storage", "strategy"])      # building level: same for every p
    modes = ["charge", "direct", "discharge"]
    x = np.arange(len(modes))
    width = 0.8 / len(COMBOS)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, combo in enumerate(COMBOS):
        d = tes[(tes["storage"] == combo[0]) & (tes["strategy"] == combo[1])]
        if d.empty:
            continue
        vals = [d.iloc[0][f"T_p_out TES bldg, {m} hours (degC)"] for m in modes]
        ax.bar(x - 0.4 + width * (k + 0.5), vals, width=width * 0.92, color=COMBO_COLOR[combo],
               label=COMBO_LABEL[combo])
    ax.axhline(base["T_p_out TES bldg, direct hours (degC)"], color=BASELINE_COLOR, linestyle="--",
               linewidth=1.5, label="Baseline building (all hours)")
    ax.set_xticks(x, [f"{m} hours" for m in modes])
    ax.set_ylabel("Primary return, flow-weighted (degC)")
    ax.set_title(f"Primary return of a TES building by operating mode, {size} h tanks")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "return_temperature_by_mode.png")


def _shade_modes(ax, bld):
    """Shade charge / discharge hours behind a time series."""
    for mode, color in MODE_FILL.items():
        on = (bld["mode_scheduled"] == mode).values
        if not on.any():
            continue
        dates = bld["date"].values
        start = None
        for i, flag in enumerate(on):
            if flag and start is None:
                start = dates[i]
            if start is not None and (not flag or i == len(on) - 1):
                end = dates[i] if not flag else dates[i] + np.timedelta64(1, "h")
                ax.axvspan(start, end, color=color, alpha=0.10, linewidth=0)
                start = None


def plot_tank_operation(base_net, size, strategy="peak_shaving"):
    t0, t1 = peak_week(base_net)
    blds = {s: load_building(tag_of(s, strategy, size)) for s in ["water", "pcm"]}
    blds = {s: b[(b["date"] >= t0) & (b["date"] < t1)] for s, b in blds.items() if b is not None}
    if not blds:
        return
    has_pcm = "pcm" in blds and "liquid_fraction" in blds["pcm"]
    n_rows = len(blds) + int(has_pcm)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 3.2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (storage, b) in zip(axes, blds.items()):
        _shade_modes(ax, b)
        ax.plot(b["date"], b["T_top"], color=SLOTS[0], label="Tank top")
        ax.plot(b["date"], b["T_bottom"], color=SLOTS[1], label="Tank bottom")
        if storage == "pcm" and "T_pcm_mean" in b:
            ax.plot(b["date"], b["T_pcm_mean"], color=SLOTS[2], label="PCM (mean)")
        ax.set_ylabel("Temperature (degC)")
        ax.set_title(f"{'Water' if storage == 'water' else 'PCM'} tank, {size} h, {strategy.replace('_', ' ')} "
                     f"(blue band = charge, orange band = discharge)")
        ax.legend(ncol=3, fontsize=9, loc="upper right")
    if has_pcm:
        ax = axes[-1]
        b = blds["pcm"]
        _shade_modes(ax, b)
        ax.plot(b["date"], b["liquid_fraction"], color=SLOTS[2])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Liquid fraction (-)")
        ax.set_title("PCM liquid fraction (mean over layers)")
    save(fig, "tank_operation_peak_week.png")


def plot_monthly_mix(base_net, nets, p, size):
    series = {"Baseline": base_net}
    series.update({COMBO_LABEL[c]: n for c, n in nets.items()})
    monthly = {k: v.groupby(v["date"].dt.month)[PRIORITY_SOURCES].sum() / 1000.0 for k, v in series.items()}
    months = sorted(set().union(*[m.index for m in monthly.values()]))
    month_names = [pd.Timestamp(2000, m, 1).strftime("%b") for m in months]
    n = len(series)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for k, (name, m) in enumerate(monthly.items()):
        xs = np.arange(len(months)) - 0.4 + width * (k + 0.5)
        bottom = np.zeros(len(months))
        for src in PRIORITY_SOURCES:
            vals = np.array([m.loc[mo, src] if mo in m.index else 0.0 for mo in months])
            ax.bar(xs, vals, bottom=bottom, width=width * 0.9, color=SOURCE_COLOR[src],
                   edgecolor=SURFACE, linewidth=1.0, label=src if k == 0 else None)
            bottom += vals
        for x in xs:
            ax.text(x, -0.02, str(k + 1), ha="center", va="top", fontsize=8, color=TEXT_2,
                    transform=ax.get_xaxis_transform())
    ax.set_xticks(np.arange(len(months)), month_names)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylabel("Heat production (GWh)")
    key = ", ".join(f"{k + 1} = {name}" for k, name in enumerate(series))
    ax.set_title(f"Monthly production mix, {size} h tanks, {p:.0%} penetration\n{key}", fontsize=9)
    ax.legend(ncol=4, fontsize=9, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "monthly_production_mix.png")


def energy_check(summary):
    cols = ["tag", "penetration", "Heat supplied (GWh)", "Heat supplied, dE-corrected (GWh)",
            "Energy charged (MWh)", "Energy discharged (MWh)", "Tank losses (MWh)",
            "Tank dE end-start (MWh)", "Tank residual (kWh, 1 bldg)", "Coupling error (kWh, 1 bldg)",
            "Unmet demand (MWh)", "Saturated hours (TES bldg)"]
    table = summary[[c for c in cols if c in summary.columns]]
    table.to_csv(RESULTS_DIR / "energy_check.csv", index=False)
    print("\nEnergy check:")
    print(table.to_string(index=False, float_format="{:.3f}".format))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=None, help="penetration for the detail figures")
    ap.add_argument("--size", type=float, default=None, help="tank size (h) for the detail figures")
    args = ap.parse_args()

    summary = load_summary()
    tes = summary[summary["tag"] != "baseline"]
    pens = sorted(tes["penetration"].unique())
    sizes = sorted(tes["size_h"].unique())
    p = args.p if args.p is not None else (0.25 if 0.25 in pens else pens[len(pens) // 2])
    size = int(args.size if args.size is not None else sizes[-1])
    print(f"Detail figures for {size} h tanks at {p:.0%} penetration")

    base_net = load_network("baseline")
    nets = {}
    for combo in COMBOS:
        net = load_network(tag_of(*combo, size), p)
        if net is not None:
            nets[combo] = net

    plot_penetration(summary)
    plot_load_duration(base_net, nets, p, size)
    plot_peak_week(base_net, nets, p, size)
    plot_return_by_mode(summary, size)
    plot_tank_operation(base_net, size)
    plot_monthly_mix(base_net, nets, p, size)
    energy_check(summary)
    print(f"\nFigures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
