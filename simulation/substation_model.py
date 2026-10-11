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
        self.UA_d = Q_design_kw / ((dT1 - dT2) / np.log(dT1 / dT2))   # Use LMTD for design value of UA
        self.exp_h = exp_h

    def UA(self, m_p, m_s):
        R = 0.5 / self.UA_d
        return 1.0 / (R * (self.m_p_d / m_p) ** self.exp_h
                      + R * (self.m_s_d / m_s) ** self.exp_h)

    def solve(self, T_p_in, m_p, T_s_in, m_s):
        """Both inlets + both flows for T_p_out, T_s_out, Q_kw."""
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
        """Primary valve: smallest primary flow that satisfy secondary supply teperature"""
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
        lo, hi = self.T_room + 1e-3, T_supply - 1e-3
        if self._lmtd(T_supply, hi) < lmtd_req:
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
    """
    Substation with optional parallel buffer tank (three modes).
 
        DH supply -> HEX -> (a) bypass -> mixing valve -> radiators
                         -> (b) tank top            (charge)
        tank top  -> mixing valve -> radiators          (discharge)
        radiator return -> HEX and/or tank bottom
 
    The tank model itself is NOT here: every step only needs the tank's top/bottom
    temperature as input and returns the flow + inlet temperature for the tank.
    Use the SAME Substation object (same HEX size) for baseline and TES runs.
    """
 
    def __init__(self, Q_design_kw, Q_hex_design_kw=None, m_p_max=None,
                 hex_kwargs=None, rad_kwargs=None, loss_fraction=0.01, dT_min_charge=3.0):
        # Radiators are sized for the building; the HEX may be sized larger for charging
        self.rad = Radiators(Q_design_kw, **(rad_kwargs or {}))
        self.hex = PlateHEX(Q_hex_design_kw or Q_design_kw, **(hex_kwargs or {}))
        self.m_p_max = m_p_max if m_p_max else 1.3 * self.hex.m_p_d
        self.loss_fraction = loss_fraction
        self.dT_min_charge = dT_min_charge   # HEX outlet can't be closer than this to T_p_in
 
    # helpers
    def _q_primary(self, Q_hex):
        return Q_hex / (1.0 - self.loss_fraction)
 
    def _radiators(self, T_hot, T_set, Q_demand):
        """Radiators behind a mixing valve fed by a hot source at T_hot.
        Returns T_r, m_rad, Q_delivered, m_hot (flow drawn from the hot source)."""
        T_supply = min(T_set, T_hot)
        T_r, m_rad, Q_del = self.rad.operate(T_supply, Q_demand)
        if m_rad <= 0:
            return T_r, 0.0, 0.0, 0.0
        if T_hot <= T_r + 1e-6:
            return T_r, m_rad, Q_del, m_rad
        m_hot = m_rad * (T_supply - T_r) / (T_hot - T_r)
        return T_r, m_rad, Q_del, m_hot
 
    @staticmethod
    def _out(mode, hexres, Q_primary, rad, tank):
        T_r, m_rad, Q_del, T_s = rad
        return dict(mode=mode, m_p=hexres["m_p"], T_p_out=hexres["T_p_out"], Q_primary=Q_primary,
                    T_s_supply=T_s, T_s_return=T_r, m_s=m_rad, Q_delivered=Q_del,
                    saturated=hexres.get("saturated", False), **tank)
 
    # baseline: no tank
    def step_direct(self, T_p_in, T_s_set, Q_demand, n_iter=60):
        """HEX feeds radiators directly. Tank (if any) isolated."""
        T_s = min(T_s_set, T_p_in - 0.5)
        no_tank = dict(m_tank=0.0, T_tank_in=np.nan, Q_tank=0.0)
        for _ in range(n_iter):
            T_r, m_s, Q_del = self.rad.operate(T_s, Q_demand)
            if m_s <= 0:
                return self._out("direct", dict(m_p=0.0, T_p_out=T_p_in), 0.0,
                                 (T_r, 0.0, 0.0, T_s), no_tank)
            res = self.hex.control_to_setpoint(T_p_in, T_r, m_s, T_s_set, self.m_p_max)
            if abs(res["T_s_out"] - T_s) < 0.01:
                break
            T_s = 0.5 * (T_s + res["T_s_out"])     # damped: only iterates when saturated
        if res["saturated"]:
            # Rebuild the secondary side from the HEX solution so energy balances exactly
            T_s = res["T_s_out"]
            Q_del = res["Q"]
            T_r = T_s - Q_del / (CP * m_s)
        return self._out("direct", res, self._q_primary(res["Q"]),
                         (T_r, m_s, Q_del, res["T_s_out"]), no_tank)
 
    # charge
    def step_charge(self, T_p_in, T_s_set, Q_demand, T_tank_bottom, Q_charge_target,
                    T_charge_set, T_bottom_cutoff=None):
        """HEX heats radiators (via mixing valve) AND charges the tank top.
        The charging pump is throttled so the water entering the tank top stays at
        T_charge_set; if HEX/valve capacity runs out, charging power is reduced
        (heating demand always has priority)."""
        T_charge_set = min(T_charge_set, T_p_in - self.dT_min_charge)
        if (T_bottom_cutoff is not None and T_tank_bottom >= T_bottom_cutoff) \
                or T_charge_set <= T_tank_bottom + 1.0 or Q_charge_target <= 0:
            return self.step_direct(T_p_in, T_s_set, Q_demand)   # tank full / can't charge
 
        T_r, m_rad, Q_del, m_hot = self._radiators(T_charge_set, T_s_set, Q_demand)
 
        def run(m_ch):
            m_s = m_hot + m_ch
            T_s_in = (m_hot * T_r + m_ch * T_tank_bottom) / m_s
            return self.hex.control_to_setpoint(T_p_in, T_s_in, m_s, T_charge_set, self.m_p_max)
 
        m_ch_max = Q_charge_target / (CP * (T_charge_set - T_tank_bottom))
        res = run(m_ch_max)
        m_ch = m_ch_max
        if res["saturated"]:                 # not enough capacity: throttle charging pump
            lo, hi = 0.0, m_ch_max
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if run(mid)["saturated"]:
                    hi = mid
                else:
                    lo = mid
            m_ch = lo
            if m_ch < 1e-4:
                return self.step_direct(T_p_in, T_s_set, Q_demand)
            res = run(m_ch)
        Q_tank = m_ch * CP * (res["T_s_out"] - T_tank_bottom)
        tank = dict(m_tank=m_ch, T_tank_in=res["T_s_out"], Q_tank=Q_tank)   # in at TOP
        return self._out("charge", res, self._q_primary(res["Q"]),
                         (T_r, m_rad, Q_del, min(T_s_set, res["T_s_out"])), tank)
 
    # discharge
    def step_discharge(self, T_p_in, T_s_set, Q_demand, T_tank_top, Q_discharge_target,
                       T_top_cutoff=None):
        """Tank covers up to Q_discharge_target via the mixing valve; HEX covers the rest
        at the normal setpoint. Radiator return is split between tank bottom and HEX."""
        T_top_cutoff = T_s_set + 1.0 if T_top_cutoff is None else T_top_cutoff
        if T_tank_top < T_top_cutoff or Q_discharge_target <= 0 or Q_demand <= 0:
            return self.step_direct(T_p_in, T_s_set, Q_demand)   # tank empty
 
        T_r, m_rad, Q_del = self.rad.operate(T_s_set, Q_demand)
        Q_from_tank = min(Q_discharge_target, Q_del)
        m_tank = Q_from_tank / (CP * (T_tank_top - T_r))
        tank = dict(m_tank=-m_tank, T_tank_in=T_r, Q_tank=-Q_from_tank)   # in at BOTTOM
 
        Q_hex_needed = Q_del - Q_from_tank
        if Q_hex_needed <= 1e-6:
            res = dict(m_p=0.0, T_p_out=T_p_in, Q=0.0, saturated=False)
        else:
            m_hex_s = Q_hex_needed / (CP * (T_s_set - T_r))
            res = self.hex.control_to_setpoint(T_p_in, T_r, m_hex_s, T_s_set, self.m_p_max)
        return self._out("discharge", res, self._q_primary(res["Q"]),
                         (T_r, m_rad, Q_del, T_s_set), tank)


def load_supply_temperature(year=2024):
    """Load the hourly primary supply temperature from the pre-computed CSV."""
    df = pd.read_csv(DATA_DIR / "DH_supply_temp.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year == year].copy()
    return df.sort_values("date").reset_index(drop=True)["T_supply"].values
