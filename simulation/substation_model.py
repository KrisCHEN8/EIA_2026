"""
Substation & Return Temperature Model
======================================
Models the heat exchanger (substation) connecting the DH primary network
to the building's secondary circuit.

Physics:
    The substation is a counter-flow heat exchanger. Given:
      - Primary supply temperature T_s (from heating curve)
      - Heat demand Q (kW)
      - Primary mass flow rate m_dot (kg/s)

    The return temperature is:
        T_return = T_supply - Q / (m_dot × cp)

    For the network level, the return temperature is computed as a
    flow-weighted average across the network, but since we're using one
    building type scaled up, this simplifies to the same equation.

    We also model the secondary side temperatures and the substation
    approach temperature (pinch).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Physical constants ──────────────────────────────────────────────────────
CP_WATER = 4.186  # kJ/(kg·K)

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DH production mix"


class Substation:
    """
    District heating substation (heat exchanger) model.

    Convention:
        Primary side: DH network water (hot supply in, cooled return out)
        Secondary side: Building circuit (cool return in, warm supply out)

    The substation transfers Q_demand from primary to secondary.

    Key physical effect modeled:
        At partial load the heat exchanger has excess capacity → T_return
        approaches T_secondary_return + approach_temp (the ideal minimum).
        At overload the HX is undersized → T_return rises because the
        primary water passes through too quickly to cool fully.
    """

    def __init__(
        self,
        approach_temp: float = 5.0,
        secondary_supply_temp: float = 60.0,
        secondary_return_temp: float = 35.0,
        max_primary_flow_kgs: float = None,
        design_capacity_kw: float = None,
        overload_penalty_degC_per_pct: float = 0.08,
    ):
        """
        Parameters
        ----------
        approach_temp : float
            Minimum temperature difference in the heat exchanger (°C).
        secondary_supply_temp : float
            Design secondary supply temperature (°C).
        secondary_return_temp : float
            Design secondary return temperature (°C).
        max_primary_flow_kgs : float or None
            Maximum primary flow rate (kg/s). If None, no limit.
        design_capacity_kw : float or None
            Design heat transfer capacity (kW). If None, assumed perfectly sized.
        overload_penalty_degC_per_pct : float
            T_return increase per 1% overload above design capacity.
            E.g. 0.08 means at 10% overload, T_return rises by 0.8°C.
        """
        self.approach_temp = approach_temp
        self.T_sec_supply_design = secondary_supply_temp
        self.T_sec_return_design = secondary_return_temp
        self.max_primary_flow = max_primary_flow_kgs
        self.design_capacity_kw = design_capacity_kw
        self.overload_penalty = overload_penalty_degC_per_pct

    def compute_return_temperature(
        self,
        T_supply_primary: float,
        Q_demand_kw: float,
        T_outdoor: float = None,
    ) -> dict:
        """
        Compute the primary return temperature given supply temp and heat demand.

        The secondary return temperature varies with outdoor temperature:
        at design conditions (coldest), it equals T_sec_return_design;
        at milder conditions, it rises (less ΔT needed across radiators).

        Load-dependent effect:
        - At partial load (Q < design): T_return approaches ideal minimum
        - At high load (Q > design): T_return increases due to HX limitation
        - This means peak shaving (reducing Q peaks) directly lowers T_return

        Parameters
        ----------
        T_supply_primary : float
            Primary DH supply temperature (°C).
        Q_demand_kw : float
            Heat demand (kW). If 0, return temperature = supply temperature.
        T_outdoor : float or None
            Outdoor temperature for secondary return estimation.

        Returns
        -------
        dict with:
            T_return_primary : float — the key output
            T_secondary_supply : float
            T_secondary_return : float
            m_dot_primary : float — mass flow rate (kg/s)
            Q_delivered_kw : float — actual heat delivered
        """
        if Q_demand_kw <= 0:
            return {
                "T_return_primary": T_supply_primary,
                "T_secondary_supply": self.T_sec_supply_design,
                "T_secondary_return": self.T_sec_return_design,
                "m_dot_primary": 0.0,
                "Q_delivered_kw": 0.0,
            }

        # ── Secondary side temperatures ────────────────────────────────────
        T_sec_supply = min(
            T_supply_primary - self.approach_temp,
            self.T_sec_supply_design,
        )

        # Secondary return rises at partial load (warmer outdoor temps)
        # At full load (design outdoor ≈ -20°C): T_sec_return = 35°C
        # At zero load (outdoor ≈ 17°C): T_sec_return → T_sec_supply (no ΔT needed)
        if T_outdoor is not None:
            T_out_design = -20.0
            T_out_balance = 17.0
            load_fraction = np.clip(
                (T_out_balance - T_outdoor) / (T_out_balance - T_out_design),
                0.0, 1.0,
            )
            T_sec_return = (
                self.T_sec_return_design * load_fraction
                + T_sec_supply * (1 - load_fraction)
            )
        else:
            T_sec_return = self.T_sec_return_design

        # ── Primary side ───────────────────────────────────────────────────
        # Ideal minimum return temperature (perfect HX at low load)
        T_return_min = T_sec_return + self.approach_temp

        # Load-dependent return temperature penalty:
        # Higher instantaneous demand → higher flow rate → less effective HX
        # → return temperature rises above the ideal minimum
        T_return_penalty = 0.0
        if self.design_capacity_kw is not None and self.design_capacity_kw > 0:
            load_ratio = Q_demand_kw / self.design_capacity_kw
            if load_ratio > 0.5:
                # Penalty grows quadratically with load above 50%
                # At design load (100%): penalty ≈ overload_penalty * 50 * 0.5 = small
                # At 150% load: penalty ≈ overload_penalty * 100 * 1.0 = significant
                excess_pct = (load_ratio - 0.5) * 100.0
                T_return_penalty = self.overload_penalty * excess_pct * load_ratio
        else:
            # No design capacity set: use a simplified load-fraction approach
            # based on outdoor temperature as proxy for load fraction
            if T_outdoor is not None:
                # Higher load_fraction → higher penalty
                T_return_penalty = load_fraction * 3.0  # Up to 3°C at design conditions

        T_return_ideal = T_return_min + T_return_penalty

        delta_T_max = T_supply_primary - T_return_ideal

        if delta_T_max <= 0:
            T_return = T_supply_primary
            m_dot = 0.0
            Q_delivered = 0.0
        else:
            m_dot_needed = Q_demand_kw / (CP_WATER * delta_T_max)  # kg/s

            if self.max_primary_flow is not None and m_dot_needed > self.max_primary_flow:
                m_dot = self.max_primary_flow
                Q_delivered = m_dot * CP_WATER * delta_T_max
                T_return = T_return_ideal
            else:
                m_dot = m_dot_needed
                Q_delivered = Q_demand_kw
                T_return = T_return_ideal

        # Clip return temperature to physically meaningful range
        T_return = np.clip(T_return, T_return_min, T_supply_primary)

        return {
            "T_return_primary": T_return,
            "T_secondary_supply": T_sec_supply,
            "T_secondary_return": T_sec_return,
            "m_dot_primary": m_dot,
            "Q_delivered_kw": Q_delivered,
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
        (90.0, 100.0, -10.0),
        (90.0, 50.0, 5.0),
        (70.0, 20.0, 12.0),
        (65.0, 5.0, 16.0),
    ]
    print(f"{'T_supply':>8} {'Q_kW':>6} {'T_out':>6} → {'T_return':>8} {'T_sec_ret':>9} {'m_dot':>7}")
    print("-" * 55)
    for t_s, q, t_out in tests:
        r = sub.compute_return_temperature(t_s, q, t_out)
        print(
            f"{t_s:8.1f} {q:6.1f} {t_out:6.1f} → "
            f"{r['T_return_primary']:8.1f} {r['T_secondary_return']:9.1f} "
            f"{r['m_dot_primary']:7.3f}"
        )
