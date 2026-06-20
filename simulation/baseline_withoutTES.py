import pandas as pd
import numpy as np


# building load
load_df = pd.read_csv("./DH production mix/building_load.csv")
load_df.index = pd.to_datetime(load_df['date'])

# weather data
weather_df = pd.read_csv("./DH production mix/weather.csv")
weather_df.index = pd.to_datetime(weather_df['date'])



# Simulation configuration
