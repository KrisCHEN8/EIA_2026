import pandas as pd
import calendar
from pathlib import Path

# ==================================================
# User settings
# ==================================================

production_file = Path(
    r"C:\Users\nanhu\OneDrive - KTH\2026\energy_informatics_conference\2024_production.csv"
)

weather_file = Path(
    r"C:\Users\nanhu\OneDrive - KTH\2026\energy_informatics_conference\2024_weather.csv"
)

output_wide = Path(
    r"C:\Users\nanhu\OneDrive - KTH\2026\energy_informatics_conference\daily_production_same_monthly_mix_wide_2024.csv"
)

output_long = Path(
    r"C:\Users\nanhu\OneDrive - KTH\2026\energy_informatics_conference\daily_production_same_monthly_mix_long_2024.csv"
)

year = 2024

# Heating base temperature.
# Lower outdoor temperature gives higher daily production.
base_temperature_C = 17.0

# Minimum weight so warm days still receive production.
minimum_weight = 0.05


# ==================================================
# 1. Read monthly production CSV
# ==================================================

prod = pd.read_csv(production_file, sep=",")
prod.columns = [str(c).strip() for c in prod.columns]

prod = prod.rename(columns={"Oppsummering (MWh)": "resource_original"})

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

# Remove Sum row
prod = prod[
    ~prod["resource_original"].astype(str).str.lower().str.contains("sum|total", na=False)
].copy()

resource_translation = {
    "Avfall": "Waste incineration heat",
    "Biogass": "Biogas",
    "Biokjel": "Wood / bio boiler",
    "Bioolje": "Bio-oil",
    "Spillvarme (Rockwool)": "Waste heat / surplus heat (Rockwool)",
    "Varmepumpe": "Heat pump",
    "El-kjeler": "Electric boilers",
    "LNG": "LNG",
    "LPG": "LPG",
    "Oljekjeler": "Oil boilers / fossil oil boilers",
}

prod["resource"] = (
    prod["resource_original"]
    .astype(str)
    .str.strip()
    .map(resource_translation)
    .fillna(prod["resource_original"])
)

for col in month_cols:
    prod[col] = pd.to_numeric(prod[col], errors="coerce").fillna(0)


# ==================================================
# 2. Read weather CSV
# ==================================================

weather = pd.read_csv(weather_file, sep=";", decimal=",")
weather.columns = [str(c).strip() for c in weather.columns]

weather = weather.rename(columns={
    "Tid(norsk normaltid)": "date",
    "Middeltemperatur, köppens formel (døgn)": "outdoor_temp_C",
})

weather["date"] = pd.to_datetime(weather["date"], format="%d.%m.%Y", errors="coerce")
weather["outdoor_temp_C"] = pd.to_numeric(weather["outdoor_temp_C"], errors="coerce")

weather = weather.dropna(subset=["date", "outdoor_temp_C"]).copy()
weather = weather[weather["date"].dt.year == year].copy()
weather["month"] = weather["date"].dt.month

weather = weather[["date", "month", "outdoor_temp_C"]].copy()


# ==================================================
# 3. Temperature weighting for total daily production
# ==================================================

weather["temperature_weight"] = base_temperature_C - weather["outdoor_temp_C"]
weather["temperature_weight"] = weather["temperature_weight"].clip(lower=minimum_weight)


# ==================================================
# 4. Calculate daily production with same monthly mix
# ==================================================

daily_rows = []

for month_name, month_number in month_map.items():

    month_weather = weather[weather["month"] == month_number].copy()

    if month_weather.empty:
        raise ValueError(f"No weather data found for month {month_number}")

    # Monthly total production across all resources
    monthly_total_mwh = prod[month_name].sum()

    # Temperature-weighted daily total production
    weight_sum = month_weather["temperature_weight"].sum()

    if weight_sum <= 0:
        month_weather["total_daily_mwh"] = monthly_total_mwh / len(month_weather)
    else:
        month_weather["total_daily_mwh"] = (
            month_weather["temperature_weight"] / weight_sum * monthly_total_mwh
        )

    # Monthly resource shares
    if monthly_total_mwh <= 0:
        prod[f"{month_name}_share"] = 0.0
    else:
        prod[f"{month_name}_share"] = prod[month_name] / monthly_total_mwh

    # Split each daily total by same monthly mix
    for _, day_row in month_weather.iterrows():
        for _, resource_row in prod.iterrows():

            resource_daily_mwh = (
                day_row["total_daily_mwh"] * resource_row[f"{month_name}_share"]
            )

            daily_rows.append({
                "date": day_row["date"],
                "month": month_number,
                "resource_original": resource_row["resource_original"],
                "resource": resource_row["resource"],
                "outdoor_temp_C": day_row["outdoor_temp_C"],
                "temperature_weight": day_row["temperature_weight"],
                "monthly_total_mwh": monthly_total_mwh,
                "monthly_resource_mwh": resource_row[month_name],
                "monthly_resource_share": resource_row[f"{month_name}_share"],
                "total_daily_mwh": day_row["total_daily_mwh"],
                "daily_mwh": resource_daily_mwh,
            })

daily_long = pd.DataFrame(daily_rows)


# ==================================================
# 5. Save long format
# ==================================================

daily_long.to_csv(output_long, index=False, encoding="utf-8-sig")


# ==================================================
# 6. Save wide format
# ==================================================

daily_wide = daily_long.pivot_table(
    index=["date", "outdoor_temp_C", "total_daily_mwh"],
    columns="resource",
    values="daily_mwh",
    aggfunc="sum"
).reset_index()

resource_cols = [
    c for c in daily_wide.columns
    if c not in ["date", "outdoor_temp_C", "total_daily_mwh"]
]

daily_wide["check_sum_resources_mwh"] = daily_wide[resource_cols].sum(axis=1)

daily_wide.to_csv(output_wide, index=False, encoding="utf-8-sig")


# ==================================================
# 7. Checks
# ==================================================

print("Saved long format:")
print(output_long)

print("\nSaved wide format:")
print(output_wide)

print("\nPreview:")
print(daily_wide.head())

# Check 1: monthly total should match original Sum row/month total
daily_long["check_month"] = daily_long["date"].dt.month

monthly_total_check = (
    daily_long
    .drop_duplicates(subset=["date", "total_daily_mwh"])
    .groupby("check_month")["total_daily_mwh"]
    .sum()
    .reset_index()
)

print("\nMonthly total production check:")
print(monthly_total_check)

# Check 2: each resource monthly sum should match original monthly resource amount
monthly_resource_check = (
    daily_long
    .groupby(["resource", "check_month"])["daily_mwh"]
    .sum()
    .reset_index()
)

print("\nMonthly resource production check:")
print(monthly_resource_check.head(40))

# Check 3: monthly mix should be constant every day within each month
mix_check = daily_long.copy()
mix_check["daily_resource_share"] = (
    mix_check["daily_mwh"] / mix_check["total_daily_mwh"]
)

print("\nDaily mix check:")
print(
    mix_check[
        ["date", "resource", "monthly_resource_share", "daily_resource_share"]
    ].head(40)
)