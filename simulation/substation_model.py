import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Constants
CP_WATER = 4.186  # kJ/(kg·K)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DH production mix"

# Add parent directory for imports if not already there
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


class Substation:
    def __init__(
        self,
        secondary_supply_temp = 55.0,
        max_primary_flow_kgs = 0.5,
        T_approach = 4.0,
        **kwargs,
    ):
        """
        secondary_supply_temp: Design secondary supply temperature (degC). Default is 55.0.
        max_primary_flow_kgs: Maximum primary flow rate (kg/s). Default is 0.5.
        T_approach: Minimum temperature difference (degC) between primary return and
            secondary return, representing a realistic finite-size HEX. Default is 4.0.
        """
        self.T_sec_supply_design = secondary_supply_temp
        self.max_primary_flow = max_primary_flow_kgs
        self.T_approach = T_approach

    def compute_hex(
        self,
        T_supply_primary,
        m_dot_primary,
        T_sec_return,
        T_sec_supply_desired,
        out_temp,
        eta_HEX = 0.85,
        m_dot_sec = 0.3,
    ):
        """
        Under baseline scenario, T_sec_supply_desired will be T_to_building, and that is secondary_supply_temp

        Under TES scenario, T_sec_supply_desired is the temperatrue entering TES, and it will be calculated from DSM,
        so that the T_to_building (water leaving TES) is close to secondary_supply_temp

        m_dot_primary needds to be calculated from DSM to ensure T_sec_supply_desired

        eta_HEX is the HEX efficiency

        T_sec_return will also be calculated during simulation by T_to_building and load and m_dot_sec

        A minimum approach temperature (self.T_approach) is enforced between the primary
        return and the secondary return, avoiding the zero-approach-temperature (infinite
        HEX) idealisation.
        """
        if m_dot_primary <= 0.0 or eta_HEX <= 0.0:
            return {
                "T_return_primary": T_supply_primary,
                "T_sec_supply_HEX": T_sec_return,
                "Q_primary_kw": 0.0,
            }

        # Solve for the primary return temperature based on energy balance
        dT_sec = max(0.0, T_sec_supply_desired - T_sec_return)

        m_dot_primary = np.clip(m_dot_primary, 0.0, self.max_primary_flow)

        T_return_primary = T_supply_primary - (m_dot_sec * dT_sec) / (m_dot_primary * eta_HEX)

        # Approach-temperature constraint: primary return must stay at least T_approach above secondary return
        T_return_min = T_sec_return + self.T_approach
        if T_return_primary < T_return_min:
            T_return_primary = T_return_min

        # Calculate actual secondary supply temperature leaving HEX
        Q_primary_kw = m_dot_primary * CP_WATER * (T_supply_primary - T_return_primary)
        Q_primary_kw = max(Q_primary_kw, 0.0)

        if m_dot_sec > 0.0:
            T_sec_supply_HEX = T_sec_return + eta_HEX * Q_primary_kw / (m_dot_sec * CP_WATER)
        else:
            T_sec_supply_HEX = T_sec_return

        return {
            "T_return_primary": T_return_primary,
            "T_sec_supply_HEX": T_sec_supply_HEX,
            "Q_primary_kw": Q_primary_kw
        }


def load_supply_temperature(year = 2024):
    """Load the hourly primary supply temperature from the pre-computed CSV."""
    df = pd.read_csv(DATA_DIR / "DH_supply_temp.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year == year].copy()

    return df.sort_values("date").reset_index(drop=True)['T_supply'].values
