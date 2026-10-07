import numpy as np
import pandas as pd
from pathlib import Path

CP = 4.186  # kJ/(kg K)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DH production mix"


class PlateHEX:
    def __init__(self, Q_design_kw, T_p_in_d=90.0, T_p_out_d=45.0,
                 T_s_in_d=40.0, T_s_out_d=60.0, exp_h=0.7):
        self.m_p_d = Q_design_kw / (CP * (T_p_in_d - T_p_out_d))
        self.m_s_d = Q_design_kw / (CP * (T_s_out_d - T_s_in_d))
        dT1, dT2 = T_p_in_d - T_s_out_d, T_p_out_d - T_s_in_d
        self.UA_d = Q_design_kw / ((dT1 - dT2) / np.log(dT1 / dT2))
        self.exp_h = exp_h

    def UA(self, m_p, m_s):
        R = 0.5 / self.UA_d
        return 1.0 / (R * (self.m_p_d / m_p) ** self.exp_h
                      + R * (self.m_s_d / m_s) ** self.exp_h)

    def solve(self, T_p_in, m_p, T_s_in, m_s):
        """Both inlets + both flows -> (T_p_out, T_s_out, Q_kw)."""
        if m_p <= 0 or m_s <= 0 or T_p_in <= T_s_in:
            return (T_p_in if m_p > 0 else T_s_in), T_s_in, 0.0
        C_p, C_s = m_p * CP, m_s * CP
        C_min, C_max = min(C_p, C_s), max(C_p, C_s)
        Cr = C_min / C_max
        NTU = self.UA(m_p, m_s) / C_min
        if abs(1.0 - Cr) < 1e-9:
            eps = NTU / (1.0 + NTU)
        else:
            e = np.exp(-NTU * (1.0 - Cr))
            eps = (1.0 - e) / (1.0 - Cr * e)
        Q = eps * C_min * (T_p_in - T_s_in)
        return T_p_in - Q / C_p, T_s_in + Q / C_s, Q

    def control_to_setpoint(self, T_p_in, T_s_in, m_s, T_s_set, m_p_max):
        """Primary valve: smallest primary flow that brings secondary outlet to setpoint."""
        T_p_out, T_s_out, Q = self.solve(T_p_in, m_p_max, T_s_in, m_s)
        if T_s_out <= T_s_set:  # valve fully open, setpoint unreachable
            return dict(m_p=m_p_max, T_p_out=T_p_out, T_s_out=T_s_out, Q=Q, saturated=True)
        lo, hi = 0.0, m_p_max
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.solve(T_p_in, mid, T_s_in, m_s)[1] < T_s_set:
                lo = mid
            else:
                hi = mid
        T_p_out, T_s_out, Q = self.solve(T_p_in, hi, T_s_in, m_s)
        return dict(m_p=hi, T_p_out=T_p_out, T_s_out=T_s_out, Q=Q, saturated=False)


class Radiators:
    """Radiator circuit with thermostatic valves: Q = Q_d * (LMTD/LMTD_d)^n."""

    def __init__(self, Q_design_kw, T_s_d=60.0, T_r_d=40.0, T_room=21.0, n=1.3, m_max=None):
        self.Q_d, self.T_room, self.n = Q_design_kw, T_room, n
        self.lmtd_d = (T_s_d - T_r_d) / np.log((T_s_d - T_room) / (T_r_d - T_room))
        self.m_max = m_max if m_max else 1.2 * Q_design_kw / (CP * (T_s_d - T_r_d))

    def _lmtd(self, T_s, T_r):
        return (T_s - T_r) / np.log((T_s - self.T_room) / (T_r - self.T_room))

    def operate(self, T_supply, Q_demand):
        """Valves throttle flow to meet demand. Returns (T_return, m_dot, Q_delivered)."""
        if Q_demand <= 0 or T_supply <= self.T_room + 0.5:
            return T_supply, 0.0, 0.0
        lmtd_req = self.lmtd_d * (Q_demand / self.Q_d) ** (1.0 / self.n)
        # LMTD falls as T_r falls, so bisect on T_r
        lo, hi = self.T_room + 1e-3, T_supply - 1e-3
        if self._lmtd(T_supply, hi) < lmtd_req:      # supply too cold: valves fully open
            return self._flow_limited(T_supply)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self._lmtd(T_supply, mid) > lmtd_req:
                hi = mid
            else:
                lo = mid
        T_r = 0.5 * (lo + hi)
        m = Q_demand / (CP * (T_supply - T_r))
        if m > self.m_max:
            return self._flow_limited(T_supply)
        return T_r, m, Q_demand

    def _flow_limited(self, T_supply):
        """Max flow: solve Q_water = Q_radiator for T_r."""
        m = self.m_max
        lo, hi = self.T_room + 1e-3, T_supply - 1e-3
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            q_w = m * CP * (T_supply - mid)
            q_r = self.Q_d * (self._lmtd(T_supply, mid) / self.lmtd_d) ** self.n
            lo, hi = (mid, hi) if q_w > q_r else (lo, mid)
        T_r = 0.5 * (lo + hi)
        return T_r, m, m * CP * (T_supply - T_r)


class Substation:
    def __init__(self, Q_design_kw, m_p_max=None, hex_kwargs=None, rad_kwargs=None,
                 loss_fraction=0.01):
        self.hex = PlateHEX(Q_design_kw, **(hex_kwargs or {}))
        self.rad = Radiators(Q_design_kw, **(rad_kwargs or {}))
        # Valve sized ~30% above design primary flow so peak is reachable at low supply T
        self.m_p_max = m_p_max if m_p_max else 1.3 * self.hex.m_p_d
        self.loss_fraction = loss_fraction

    def step_direct(self, T_p_in, T_s_set, Q_demand, n_iter=5):
        """Baseline: HEX feeds radiators directly. Iterates HEX <-> radiator coupling."""
        T_s = min(T_s_set, T_p_in - 0.5)
        res = None
        for _ in range(n_iter):
            T_r, m_s, Q_del = self.rad.operate(T_s, Q_demand)
            if m_s <= 0:
                return dict(T_p_out=T_p_in, m_p=0.0, Q_primary=0.0, T_s_supply=T_s,
                            T_s_return=T_r, m_s=0.0, Q_delivered=0.0, saturated=False)
            res = self.hex.control_to_setpoint(T_p_in, T_r, m_s, T_s_set, self.m_p_max)
            if abs(res["T_s_out"] - T_s) < 0.01:
                break
            T_s = res["T_s_out"]
        return dict(T_p_out=res["T_p_out"], m_p=res["m_p"],
                    Q_primary=res["Q"] / (1.0 - self.loss_fraction),
                    T_s_supply=res["T_s_out"], T_s_return=T_r, m_s=m_s,
                    Q_delivered=Q_del, saturated=res["saturated"])

    def step_charge(self, T_p_in, T_tank_bottom, m_charge, T_charge_set):
        """TES charging loop: independent charging pump flow through HEX."""
        res = self.hex.control_to_setpoint(T_p_in, T_tank_bottom, m_charge,
                                           T_charge_set, self.m_p_max)
        res["Q_primary"] = res["Q"] / (1.0 - self.loss_fraction)
        return res


def load_supply_temperature(year=2024):
    """Load the hourly primary supply temperature from the pre-computed CSV."""
    df = pd.read_csv(DATA_DIR / "DH_supply_temp.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year == year].copy()
    return df.sort_values("date").reset_index(drop=True)["T_supply"].values