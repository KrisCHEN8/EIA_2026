import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Settings ───────────────────────────────────────────────────────────────────
priority_sources = ['Waste heat', 'Electricity', 'Biomass', 'Fossil fuel']
source_colors    = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
years            = range(2021, 2026)

# ── Load real monthly reference (agg sheet) ────────────────────────────────────
ref = pd.read_excel('./DH production mix/four_category.xlsx', sheet_name='agg')
ref['month'] = pd.to_datetime(ref['month'].astype(str).str.strip(), format='%Y-%m')
ref['year']  = ref['month'].dt.year
ref['mo']    = ref['month'].dt.month
for src in priority_sources:
    ref[src] = pd.to_numeric(ref[src], errors='coerce').fillna(0)
ref['total'] = ref[priority_sources].sum(axis=1)

# ── Load & aggregate daily CSVs → monthly ─────────────────────────────────────
daily_frames = []
for yr in years:
    path = f'./DH production mix/daily_{yr}.csv'
    try:
        d = pd.read_csv(path, parse_dates=['date'])
        d['year'] = d['date'].dt.year
        d['mo']   = d['date'].dt.month
        daily_frames.append(d)
    except FileNotFoundError:
        print(f'Warning: {path} not found, skipping.')

daily_all = pd.concat(daily_frames, ignore_index=True)

agg_cols = priority_sources + ['daily_total_mwh']
monthly_agg = (
    daily_all
    .groupby(['year', 'mo'])[agg_cols]
    .sum()
    .reset_index()
    .rename(columns={'daily_total_mwh': 'total'})
)

# ── Optional: print numeric comparison ────────────────────────────────────────
print("=== Simulated monthly totals (aggregated from daily) ===")
print(monthly_agg.to_string(index=False))

# ── Plot: one figure per year ──────────────────────────────────────────────────
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
x      = list(range(1, 13))
width  = 0.35        # width of each bar group
offset = width / 2   # shift left (sim) / right (real) from month tick

for yr in years:
    sim_yr = monthly_agg[monthly_agg['year'] == yr].set_index('mo')
    ref_yr = ref[ref['year'] == yr].set_index('mo')

    if sim_yr.empty and ref_yr.empty:
        continue

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # ── Simulated bars (left of each month tick) ───────────────────────────
    bottom_sim = [0.0] * 12
    for src, col in zip(priority_sources, source_colors):
        vals = [float(sim_yr.loc[m, src]) if m in sim_yr.index else 0.0 for m in x]
        ax.bar(
            [m - offset for m in x], vals, width=width,
            bottom=bottom_sim, color=col, alpha=0.9,
            label=f'{src} (sim)'
        )
        bottom_sim = [b + v for b, v in zip(bottom_sim, vals)]

    # ── Reference bars (right of each month tick, hatched) ─────────────────
    bottom_ref = [0.0] * 12
    for i, (src, col) in enumerate(zip(priority_sources, source_colors)):
        vals = [float(ref_yr.loc[m, src]) if m in ref_yr.index else 0.0 for m in x]
        ax.bar(
            [m + offset for m in x], vals, width=width,
            bottom=bottom_ref, color=col, alpha=0.45,
            hatch='//', edgecolor='white', linewidth=0.4,
            label=f'{src} (real)'
        )
        bottom_ref = [b + v for b, v in zip(bottom_ref, vals)]

    # ── Total lines ────────────────────────────────────────────────────────
    sim_totals = [float(sim_yr.loc[m, 'total']) if m in sim_yr.index else None for m in x]
    ref_totals = [float(ref_yr.loc[m, 'total']) if m in ref_yr.index else None for m in x]

    ax.plot([m - offset for m in x], sim_totals,
            'o-', color='white', lw=1.8, ms=5, label='Total (sim)', zorder=5)
    ax.plot([m + offset for m in x], ref_totals,
            's--', color='#FFD700', lw=1.8, ms=5, label='Total (real)', zorder=5)

    # ── Styling ────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, color='white')
    ax.tick_params(colors='white')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1e3:.0f}k'))
    ax.set_ylabel('Energy (MWh)', color='white')
    ax.set_xlabel('Month', color='white')
    ax.set_title(
        f'{yr}  —  Simulated vs Real monthly production by source',
        color='white', fontsize=13, pad=12
    )
    for spine in ax.spines.values():
        spine.set_color('#444466')
    ax.grid(axis='y', color='#333355', linewidth=0.6)

    # Deduplicate legend (keep first occurrence of each label)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = h
    ax.legend(
        seen.values(), seen.keys(),
        loc='upper right', framealpha=0.3,
        labelcolor='white', facecolor='#1a1a2e',
        fontsize=8.5, ncol=2
    )

    plt.tight_layout()
    out_path = f'./DH production mix/comparison_{yr}.png'
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)   # free memory; prevents blocking when run as a script
    print(f'Year {yr} saved → {out_path}')
