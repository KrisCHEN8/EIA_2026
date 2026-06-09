import pandas as pd
import numpy as np
from pyearth import Earth
import matplotlib.pyplot as plt


weather_file = "./DH production mix/weather.csv"
mix_file = "./DH production mix/four_category.xlsx"
output_file = "./DH production mix/daily_priority_dispatch.csv"

priority_sources = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]

# Read weather and production mix
weather_df = pd.read_csv(weather_file)
weather_df.index = pd.to_datetime(weather_df["date"])
weather_df = weather_df.resample('D').mean()
weather_df["year"] = weather_df["date"].dt.year
weather_df["month"] = weather_df["date"].dt.month
weather_df["month_period"] = weather_df["date"].dt.to_period("M")
weather_monthly = weather_df.resample("M").mean()

# Read production mix
monthly_mix = pd.read_excel(mix_file, sheet_name='agg')

# Convert 'month' column to datetime (e.g., 2021-01-01)
monthly_mix['month'] = pd.to_datetime(monthly_mix['month'], format='%Y-%m')
monthly_mix['month_period'] = monthly_mix['month'].dt.to_period('M')
monthly_mix["Total_Monthly_MWh"] = monthly_mix[priority_sources].sum(axis=1)

# Extract year and month number if needed
monthly_mix['year'] = monthly_mix['month'].dt.year
monthly_mix['month_num'] = monthly_mix['month'].dt.month


"""
# Map monthly historical totals back to daily weather for training targets
weather_df = weather_df.merge(monthly_mix, on="month_period", how="left")

# Create a proxy daily load weight using historical logic to train MARS
weather_df["HDD"] = (17.0 - weather_df["outdoor_temp_C"]).clip(lower=0) + 6.6
monthly_hdd_sum = weather_df.groupby("month_period")["HDD"].transform("sum")
weather_df["Target_Daily_MWh"] = (weather_df["HDD"] / monthly_hdd_sum) * weather_df["Total_Monthly_MWh"]

# Split Train set (2021-2024), and Validation set (2025)
train_df = weather_df[weather_df["year"].isin([2021, 2022, 2023, 2024])].copy()
val_df = weather_df[weather_df["year"] == 2025].copy()

X_train = train_df[["temperature"]].values
y_train = train_df["Target_Daily_MWh"].values

X_val = val_df[["temperature"]].values


# Train MARS model
mars_model = Earth(max_terms=10, max_degree=1)
mars_model.fit(X_train, y_train)

# Predict raw daily values for 2025 validation set
val_df["MARS_Raw_Daily_MWh"] = mars_model.predict(X_val)
val_df["MARS_Raw_Daily_MWh"] = val_df["MARS_Raw_Daily_MWh"].clip(lower=0)

val_df["Predicted_Month_Sum"] = val_df.groupby("month_period")["MARS_Raw_Daily_MWh"].transform("sum")

# Create a scaling factor: Real Monthly Total / MARS Predicted Monthly Total
val_df["Scaling_Factor"] = val_df["Total_Monthly_MWh"] / val_df["Predicted_Month_Sum"]

# Final daily prediction perfectly matching monthly totals (Zero Error Objective)
val_df["daily_total_mwh"] = val_df["MARS_Raw_Daily_MWh"] * val_df["Scaling_Factor"]
"""