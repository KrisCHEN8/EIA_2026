import numpy as np
import pandas as pd


def heating_curve(
    t_out,
    t_supply_min=60.0,
    t_supply_max=110.0,
    t_out_summer=17.0,
    t_out_dot=-20.0,
    noise_variance=3.0,
    seed=None,
):
    """
    Map outdoor temperature to DH primary supply temperature.

    Parameters
    t_out: Outdoor temperature in degC.
    t_supply_min: Supply temperature at/above the summer threshold [degC].
    t_supply_max: Supply temperature at/below the design outdoor temperature [degC].
    t_out_summer: Outdoor temperature above which the curve flattens to t_supply_min [degC].
    t_out_dot: design point design temp [degC].

    Returns
        Primary side supply temperature in degC.
    """
    t_out_arr = np.asarray(t_out, dtype=float)

    slope = (t_supply_max - t_supply_min) / (t_out_summer - t_out_dot)
    t_supply = t_supply_min + slope * (t_out_summer - t_out_arr)

    if noise_variance > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=np.sqrt(noise_variance), size=t_out_arr.shape)
        t_supply += noise

    t_supply = np.clip(t_supply, t_supply_min, t_supply_max)
    t_supply = np.round(t_supply, decimals=1)

    if isinstance(t_out, pd.Series):
        return pd.Series(t_supply, index=t_out.index, name="T_supply")
    if np.isscalar(t_out):
        return float(t_supply)
    return t_supply


if __name__ == "__main__":
    weather_df = pd.read_csv("./DH production mix/weather.csv")
    weather_df.index = pd.to_datetime(weather_df['date'], format='%Y-%m-%d %H:%M:%S')

    weather_df["T_supply"] = heating_curve(weather_df["temperature"], seed=42)

    # Create another csv file with date and supply temperature
    output_path = "./DH production mix/DH_supply_temp.csv"
    weather_df[["date","T_supply"]].to_csv(output_path)
    print(f"Saved mapped supply temperatures to {output_path}")
    print(weather_df[["date","T_supply"]].head())
