import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


weather_2019 = pd.read_excel("DH production mix/nursing homes with DH 2016-2019 Trondheim.xlsx", sheet_name="outdoor temp-2019")
weather_2019.index = pd.to_datetime(weather_2019["Time(Norwegian mean time)"], format="%d.%m.%Y %H:%M")

load_2019 = pd.read_excel("DH production mix/nursing homes with DH 2016-2019 Trondheim.xlsx", sheet_name="DH-2019")
load_2019.index = pd.to_datetime(load_2019["DateTime"])

load = load_2019["Bromstad (W/m2)"] * 1350 * 0.001   # kW
temperature = weather_2019["Air temperature (°C)"]

# Heating Curve for temperature less than 21°C
df = pd.DataFrame({"temperature": temperature, "load": load}).dropna()
df_heating = df[df["temperature"] < 21]

coeffs = np.polyfit(df_heating["temperature"], df_heating["load"], deg=2)
poly = np.poly1d(coeffs)

print(f"Quadratic fit: load = {coeffs[0]:.4f}·T² + {coeffs[1]:.4f}·T + {coeffs[2]:.4f}")

# Error distribution
load_pred = poly(df_heating["temperature"])
residuals = df_heating["load"] - load_pred

err_mean = residuals.mean()
err_var  = residuals.var()
err_std  = residuals.std()

print(f"Error mean:     {err_mean:.4f} kW")
print(f"Error variance: {err_var:.4f} kW²")
print(f"Error std dev:  {err_std:.4f} kW")

# Plot
T_range = np.linspace(df_heating["temperature"].min(), 21, 200)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_heating["temperature"], df_heating["load"], alpha=0.15, s=8, color="#4C9BE8", label="Hourly data")
ax.plot(T_range, poly(T_range), color="#E84C4C", linewidth=2.5, label="Quadratic fit")
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
ax.set_xlabel("Outdoor temperature (°C)", fontsize=14)
ax.set_ylabel("Building load (kW)", fontsize=14)
# ax.set_title("Heating Curve — Bromstad Nursing Home")
ax.legend(fontsize=14)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("./figs/heating_curve.png", dpi=300)
plt.close()


# mapping the 2024 temperature
weather_df = pd.read_csv("./DH production mix/weather.csv")
weather_df.index = pd.to_datetime(weather_df['date'], format='%Y-%m-%d %H:%M:%S')
weather_df['load'] = poly(weather_df['temperature'])

rng = np.random.default_rng(seed=42)   
noise = rng.normal(loc=err_mean, scale=err_std, size=len(weather_df))
weather_df['load'] = (weather_df['load'] + noise).clip(lower=0)

weather_df[['date', 'load']].to_csv('./DH production mix/building_load.csv')
