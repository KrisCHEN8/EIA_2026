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

from TES_model import WaterTank, PCMStorageTank


class Substation:
    def __init__(
        self,
        secondary_supply_temp = 55.0,
        secondary_return_temp = 35.0,
        max_primary_flow_kgs = 1.0,
        tes_type = "no TES",
        **kwargs,
    ):
        """
        secondary_supply_temp: Design secondary supply temperature (°C). Default is 55.0.
        secondary_return_temp: Design secondary return temperature (°C).
        max_primary_flow_kgs: Maximum primary flow rate (kg/s). Default is 0.5.
        tes_type: Type of Thermal Energy Storage: "no TES", "Water Tank", or "PCM Storage"
        """
        self.T_sec_supply_design = secondary_supply_temp
        self.T_sec_return_design = secondary_return_temp
        self.max_primary_flow = max_primary_flow_kgs
        self.tes_type = tes_type

        # Sizing the tank based on the tes_type
        if self.tes_type == "Water Tank":
            self.tank = WaterTank(water_mass_kg=1000.0, initial_temperature=50.0)
        elif self.tes_type == "PCM Storage":
            self.tank = PCMStorageTank(water_mass_kg=500.0, pcm_mass_kg=578.0, initial_temperature=50.0)
        else:
            self.tank = None

    def compute_secondary_return_temperature(self, Q_demand_kw, T_sec_supply = 55.0):
        """
        Calculate secondary radiator return temperature based on radiator energy balance.
        """
        m_dot_sec = 0.2

        # Calculate the secondary return temperature via energy balance
        T_sec_return = T_sec_supply - Q_demand_kw / (m_dot_sec * CP_WATER)

        return np.clip(T_sec_return, 25.0, T_sec_supply)

    def compute_return_temperature(
        self,
        T_supply_primary: float,
        Q_demand_kw: float,
        mode: str = "standby",
        m_dot_charge: float = 0.0,
        T_outdoor: float = None,
    ):
        """
        Compute primary return temperature and flow rate based on the mode and load.
        """
        # Secondary flow rate
        m_dot_sec_nominal = 0.2
        m_dot_sec = m_dot_sec_nominal if Q_demand_kw > 0 else 0.0

        n_steps = 60
        dt_sub = 60.0

        # Initialize outputs
        Q_charge_kw = 0.0
        Q_discharge_kw = 0.0
        T_water = np.nan
        T_pcm = np.nan

        # Without TES
        if self.tank is None:
            T_sec_supply = self.T_sec_supply_design  # 55.0
            T_sec_return = self.compute_secondary_return_temperature(Q_demand_kw, T_sec_supply)
            T_sec_inlet = T_sec_return
            Q_HEX = Q_demand_kw

            # Substation HEX calculation
            delta_T = T_supply_primary - T_sec_inlet
            if delta_T <= 0:
                m_dot_primary = 0.0
                T_return = T_supply_primary
                Q_delivered_kw = 0.0
            else:
                m_dot_needed = Q_HEX / (CP_WATER * delta_T)
                if self.max_primary_flow is not None and m_dot_needed > self.max_primary_flow:
                    m_dot_primary = self.max_primary_flow
                    Q_delivered_kw = m_dot_primary * CP_WATER * delta_T
                    # Re-calculate temperatures based on actual delivered heat
                    T_sec_return = self.compute_secondary_return_temperature(Q_delivered_kw, T_sec_supply)
                    T_return = T_sec_return
                else:
                    m_dot_primary = m_dot_needed
                    Q_delivered_kw = Q_demand_kw
                    T_return = T_supply_primary / (m_dot_primary * (T_sec_supply - T_sec_return))

        # Tank scenario
        else:
            T_water_current = self.tank.T_water

            if mode == "charging":
                # Step the tank: inlet is 55.0°C (from HEX)
                T_out_sum = 0.0
                for _ in range(n_steps):
                    T_out_sub = self.tank.step(m_dot=m_dot_charge, T_in=self.T_sec_supply_design, dt=dt_sub)
                    T_out_sum += T_out_sub
                T_out_tank = T_out_sum / n_steps

                Q_charge_kw = m_dot_charge * CP_WATER * (self.T_sec_supply_design - T_out_tank)
                Q_charge_kw = max(Q_charge_kw, 0.0)

                # Substation heats building load plus charging load
                T_sec_supply = self.T_sec_supply_design  # 55.0
                T_sec_return = self.compute_secondary_return_temperature(Q_demand_kw, T_sec_supply)

                # Secondary inlet to HEX is mixed radiator return and tank return
                m_dot_total_sec = m_dot_sec + m_dot_charge
                if m_dot_total_sec > 0:
                    T_sec_inlet = (m_dot_sec * T_sec_return + m_dot_charge * T_out_tank) / m_dot_total_sec
                else:
                    T_sec_inlet = T_sec_return

                Q_HEX = Q_demand_kw + Q_charge_kw

                delta_T = T_supply_primary - T_sec_inlet
                if delta_T <= 0:
                    m_dot_primary = 0.0
                    T_return = T_supply_primary
                    Q_delivered_kw = 0.0
                else:
                    m_dot_needed = Q_HEX / (CP_WATER * delta_T)
                    if self.max_primary_flow is not None and m_dot_needed > self.max_primary_flow:
                        m_dot_primary = self.max_primary_flow
                        Q_delivered_kw = m_dot_primary * CP_WATER * delta_T
                        # Note: under constrained flow, we deliver Q_delivered_kw in total (HEX is maxed)
                        # The return temperature is the secondary inlet
                        T_return = T_sec_inlet
                    else:
                        m_dot_primary = m_dot_needed
                        Q_delivered_kw = Q_HEX
                        T_return = T_sec_inlet

            elif mode == "discharging":
                # Building design supply temperature is the temperature leaving TES
                T_sec_supply = T_water_current
                T_sec_return = self.compute_secondary_return_temperature(Q_demand_kw, T_sec_supply)

                # Step the tank: inlet is radiator return temperature
                T_out_sum = 0.0
                for _ in range(n_steps):
                    T_out_sub = self.tank.step(m_dot=m_dot_sec, T_in=T_sec_return, dt=dt_sub)
                    T_out_sum += T_out_sub
                T_out_tank = T_out_sum / n_steps

                Q_discharge_kw = m_dot_sec * CP_WATER * (T_sec_supply - T_sec_return)
                Q_discharge_kw = np.clip(Q_discharge_kw, 0.0, Q_demand_kw)

                # The primary side of the HEX is bypassed during discharging
                m_dot_primary = 0.0
                Q_delivered_kw = Q_discharge_kw
                T_return = T_supply_primary
                T_sec_inlet = T_sec_return

            else:
                # Standby: only ambient losses in the tank
                T_out_sum = 0.0
                for _ in range(n_steps):
                    T_out_sub = self.tank.step(m_dot=0.0, T_in=0.0, dt=dt_sub)
                    T_out_sum += T_out_sub
                T_out_tank = T_out_sum / n_steps

                # Normal substation operation
                T_sec_supply = self.T_sec_supply_design  # 55.0
                T_sec_return = self.compute_secondary_return_temperature(Q_demand_kw, T_sec_supply)
                T_sec_inlet = T_sec_return
                Q_HEX = Q_demand_kw

                delta_T = T_supply_primary - T_sec_inlet
                if delta_T <= 0:
                    m_dot_primary = 0.0
                    T_return = T_supply_primary
                    Q_delivered_kw = 0.0
                else:
                    m_dot_needed = Q_HEX / (CP_WATER * delta_T)
                    if self.max_primary_flow is not None and m_dot_needed > self.max_primary_flow:
                        m_dot_primary = self.max_primary_flow
                        Q_delivered_kw = m_dot_primary * CP_WATER * delta_T
                        T_sec_return = self.compute_secondary_return_temperature(Q_delivered_kw, T_sec_supply)
                        T_return = T_sec_return
                    else:
                        m_dot_primary = m_dot_needed
                        Q_delivered_kw = Q_demand_kw
                        T_return = T_sec_inlet

            # Export current tank variables
            T_water = self.tank.T_water
            if hasattr(self.tank, "pcm"):
                T_pcm = self.tank.pcm.T

        # Clip primary return to physically meaningful range
        T_return = np.clip(T_return, T_sec_inlet, T_supply_primary)

        return {
            "T_return_primary": T_return,
            "T_secondary_supply": T_sec_supply,
            "T_secondary_return": T_sec_return,
            "m_dot_primary": m_dot_primary,
            "Q_delivered_kw": Q_delivered_kw,
            "Q_charge_kw": Q_charge_kw,
            "Q_discharge_kw": Q_discharge_kw,
            "T_water": T_water,
            "T_pcm": T_pcm,
        }


def load_supply_temperature(year: int = 2024) -> pd.DataFrame:
    """Load the hourly primary supply temperature from the pre-computed CSV."""
    df = pd.read_csv(DATA_DIR / "DH_supply_temp.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year == year].copy()

    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    sub = Substation()

    # Test cases
    tests = [
        (90.0, 100.0),
        (90.0, 50.0),
        (70.0, 20.0),
        (65.0, 5.0),
    ]
    print(f"{'T_supply':>8} {'Q_kW':>6} → {'T_return':>8} {'m_dot':>7}")
    print("-" * 35)
    for t_s, q in tests:
        r = sub.compute_return_temperature(t_s, q)
        print(
            f"{t_s:8.1f} {q:6.1f} → "
            f"{r['T_return_primary']:8.1f} {r['m_dot_primary']:7.3f}"
        )
