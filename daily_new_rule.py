import pandas as pd
import numpy as np
import calendar
from pathlib import Path
import matplotlib.pyplot as plt

# ==================================================
# User settings
# ==================================================

# Use paths relative to this script's directory
base_dir = Path(__file__).resolve().parent

weather_file = base_dir / "./DH production mix/2024_weather.csv"
mix_file = base_dir / "./DH production mix/four_category.xlsx"
output_file = base_dir / "./DH production mix/daily_priority_dispatch_2024.csv"

year = 2024

# Load-index equation:
# Below 17°C, load increases linearly.
# Above 17°C, load remains at base-load level 6.6.
base_load_index = 6.6
balance_temperature_C = 17.0

priority_sources = [
    "Waste heat",
    "Electricity",
    "Biomass",
    "Fossil fuel",
]


# ==================================================
# 1. Read weather data
# ==================================================

weather = pd.read_csv(
    weather_file,
    sep=";",
    decimal=","
)

weather.columns = [str(c).strip() for c in weather.columns]

weather = weather.rename(columns={
    "Tid(norsk normaltid)": "date",
    "Middeltemperatur, köppens formel (døgn)": "outdoor_temp_C",
})

weather["date"] = pd.to_datetime(
    weather["date"],
    format="%d.%m.%Y",
    errors="coerce"
)

weather["outdoor_temp_C"] = pd.to_numeric(
    weather["outdoor_temp_C"],
    errors="coerce"
)

weather = weather.dropna(subset=["date", "outdoor_temp_C"]).copy()
weather = weather[weather["date"].dt.year == year].copy()

weather["month"] = weather["date"].dt.month

weather["daily_load_index"] = (
    base_load_index
    + (balance_temperature_C - weather["outdoor_temp_C"]).clip(lower=0)
)


# ==================================================
# 2. Read monthly four-category energy mix
# ==================================================

mix = pd.read_excel(mix_file)

mix.columns = [str(c).strip() for c in mix.columns]

# The first column should contain the source/category names
mix = mix.rename(columns={mix.columns[0]: "source_original"})

month_map = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "Mai": 5,
    "Juni": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Okt": 10,
    "Nov": 11,
    "Des": 12,
}

month_cols = list(month_map.keys())

# Remove total row if it exists
mix = mix[
    ~mix["source_original"]
    .astype(str)
    .str.lower()
    .str.contains("sum|total", na=False)
].copy()

mix["source_original"] = mix["source_original"].astype(str).str.strip()

# Standardize source names
source_translation = {
    "Waste heat": "Waste heat",
    "waste heat": "Waste heat",
    "Waste Heat": "Waste heat",

    "Electricity": "Electricity",
    "electricity": "Electricity",
    "Electricity ": "Electricity",

    "Biomass": "Biomass",
    "biomass": "Biomass",

    "Fossil fuel": "Fossil fuel",
    "fossil fuel": "Fossil fuel",
    "Fossil Fuel": "Fossil fuel",
}

mix["source"] = (
    mix["source_original"]
    .map(source_translation)
    .fillna(mix["source_original"])
)

for col in month_cols:
    if col not in mix.columns:
        raise ValueError(f"Missing month column in four_category.xlsx: {col}")
    mix[col] = pd.to_numeric(mix[col], errors="coerce").fillna(0)

# If repeated category rows exist, aggregate them
mix_grouped = mix.groupby("source")[month_cols].sum().reset_index()

missing_sources = [
    source for source in priority_sources
    if source not in mix_grouped["source"].values
]

if missing_sources:
    raise ValueError(
        f"Missing required source categories in four_category.xlsx: {missing_sources}"
    )


# ==================================================
# 3. Dispatch helper function
# ==================================================

def dispatch_one_source(month_df, source_name, monthly_source_mwh):
    """
    Dispatch one source to daily load by priority.

    Logic:
    1. Source is only used where remaining load > 0.
    2. Active days = days with remaining load.
    3. Initial daily limit = monthly source energy / active days.
    4. Allocate min(remaining load, daily limit).
    5. If some monthly energy is unused, redistribute to days that still have remaining load.
    6. Continue until monthly source energy is used or no remaining load is available.
    """

    monthly_source_mwh = float(monthly_source_mwh)

    month_df[source_name] = 0.0

    if monthly_source_mwh <= 0:
        return month_df

    tolerance = 1e-9
    max_iter = 1000

    remaining_monthly_source = monthly_source_mwh

    for _ in range(max_iter):

        if remaining_monthly_source <= tolerance:
            break

        active_mask = month_df["remaining_load"] > tolerance

        if not active_mask.any():
            break

        active_days = active_mask.sum()

        daily_limit = remaining_monthly_source / active_days

        add_amount = np.minimum(
            month_df.loc[active_mask, "remaining_load"],
            daily_limit
        )

        month_df.loc[active_mask, source_name] += add_amount
        month_df.loc[active_mask, "remaining_load"] -= add_amount

        used_this_round = add_amount.sum()
        remaining_monthly_source -= used_this_round

        if used_this_round <= tolerance:
            break

    # Store unused source energy for checking
    month_df[f"{source_name}_unused_monthly_mwh"] = remaining_monthly_source

    return month_df


# ==================================================
# 4. Monthly dispatch loop
# ==================================================

all_months = []

for month_name, month_number in month_map.items():

    month_df = weather[weather["month"] == month_number].copy()

    if month_df.empty:
        raise ValueError(f"No weather data found for month {month_number}")

    monthly_energy = {}

    for source in priority_sources:
        monthly_energy[source] = float(
            mix_grouped.loc[
                mix_grouped["source"] == source,
                month_name
            ].iloc[0]
        )

    monthly_total_mwh = sum(monthly_energy.values())

    # Daily total production follows daily load index
    load_index_sum = month_df["daily_load_index"].sum()

    if load_index_sum <= 0:
        month_df["daily_total_mwh"] = monthly_total_mwh / len(month_df)
    else:
        month_df["daily_total_mwh"] = (
            month_df["daily_load_index"]
            / load_index_sum
            * monthly_total_mwh
        )

    # Remaining load to be covered by sources
    month_df["remaining_load"] = month_df["daily_total_mwh"]

    # Dispatch according to priority
    for source in priority_sources:
        month_df = dispatch_one_source(
            month_df,
            source,
            monthly_energy[source]
        )

    # If small remaining load exists due to numerical issues or source limits,
    # assign it to fossil fuel as final backup.
    month_df["unmet_load_after_dispatch"] = month_df["remaining_load"].clip(lower=0)

    month_df["Fossil fuel"] += month_df["unmet_load_after_dispatch"]
    month_df["remaining_load"] = 0.0

    # Metadata
    month_df["monthly_total_mwh"] = monthly_total_mwh
    month_df["monthly_waste_heat_mwh"] = monthly_energy["Waste heat"]
    month_df["monthly_electricity_mwh"] = monthly_energy["Electricity"]
    month_df["monthly_biomass_mwh"] = monthly_energy["Biomass"]
    month_df["monthly_fossil_fuel_mwh"] = monthly_energy["Fossil fuel"]

    all_months.append(month_df)


daily = pd.concat(all_months, ignore_index=True)


# ==================================================
# 5. Checks
# ==================================================

daily["check_sum_sources_mwh"] = (
    daily["Waste heat"]
    + daily["Electricity"]
    + daily["Biomass"]
    + daily["Fossil fuel"]
)

daily["check_error_daily_mwh"] = (
    daily["check_sum_sources_mwh"] - daily["daily_total_mwh"]
)

monthly_check = daily.groupby("month")[
    [
        "daily_total_mwh",
        "Waste heat",
        "Electricity",
        "Biomass",
        "Fossil fuel",
        "check_sum_sources_mwh",
    ]
].sum()

print("\nMonthly dispatch check:")
print(monthly_check)

print("\nMaximum daily balance error:")
print(daily["check_error_daily_mwh"].abs().max())

# Check against input monthly mix
print("\nInput monthly energy mix:")
print(mix_grouped)


# ==================================================
# 6. Save results
# ==================================================

output_cols = [
    "date",
    "outdoor_temp_C",
    "daily_load_index",
    "daily_total_mwh",
    "Waste heat",
    "Electricity",
    "Biomass",
    "Fossil fuel",
    "check_sum_sources_mwh",
    "check_error_daily_mwh",
]
daily[output_cols].to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")


# ==================================================
# 7. Continuous stacked area plot for selected period
# ==================================================

start_date = "2024-01-01"
end_date = "2024-12-31"

plot_df = daily[
    (daily["date"] >= start_date) &
    (daily["date"] <= end_date)
].copy()

plot_df = plot_df.sort_values("date")

energy_cols = [
    "Waste heat",
    "Electricity",
    "Biomass",
    "Fossil fuel",
]

plt.figure(figsize=(15, 6))

# Use stacked area plot for continuous daily composition
plt.stackplot(
    plot_df["date"],
    [plot_df[col] for col in energy_cols],
    labels=energy_cols,
    alpha=0.85
)

# Plot a clean dashed line for the total daily production
plt.plot(
    plot_df["date"],
    plot_df["daily_total_mwh"],
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Total daily production"
)

plt.xlabel("Date")
plt.ylabel("Daily production (MWh)")
plt.title(f"Daily energy mix from {start_date} to {end_date}")
plt.legend(loc="upper left")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("./figs/daily_stacked_area.png", dpi=300)
plt.show()

