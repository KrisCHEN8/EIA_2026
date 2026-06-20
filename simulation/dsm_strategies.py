"""
DSM Strategies
==============
Demand-side management control strategies applied at the building/network level.

Two strategies are implemented:
1. **Peak Shaving**: Cap instantaneous demand at a threshold; excess heat is
   pre-stored in TES during lower-demand hours and discharged during peaks.
2. **Load Shifting**: Shift demand from peak hours to off-peak hours by
   pre-charging TES (e.g., at night when base-load waste heat is cheap).

Both strategies modify the hourly load profile that the DH network "sees",
while ensuring the building still receives the heat it needs (from TES).
"""

import numpy as np
import pandas as pd


class PeakShavingStrategy:
    """
    Peak Shaving DSM Strategy.

    Caps the network demand at a threshold (e.g., 80th percentile of daily load).
    When demand exceeds the cap, the excess is served from TES.
    When demand is below the cap, TES is recharged.

    The threshold can be:
    - A fixed MW value
    - A percentile of the load profile (recommended: 70–85th)
    """

    def __init__(
        self,
        cap_percentile: float = 80.0,
        cap_absolute_mw: float = None,
        tes_capacity_mwh: float = None,
        tes_max_charge_rate_mw: float = None,
        tes_efficiency: float = 0.95,
    ):
        """
        Parameters
        ----------
        cap_percentile : float
            Percentile of the load profile to use as the cap (0–100).
            Only used if cap_absolute_mw is None.
        cap_absolute_mw : float or None
            Fixed cap in MW. Overrides cap_percentile if set.
        tes_capacity_mwh : float or None
            TES storage capacity (MWh). If None, assumed unlimited.
        tes_max_charge_rate_mw : float or None
            Maximum TES charge/discharge rate (MW).
        tes_efficiency : float
            Round-trip efficiency of TES (0–1).
        """
        self.cap_percentile = cap_percentile
        self.cap_absolute_mw = cap_absolute_mw
        self.tes_capacity_mwh = tes_capacity_mwh
        self.tes_max_charge_rate_mw = tes_max_charge_rate_mw
        self.tes_efficiency = tes_efficiency

    def apply(self, load_mw: np.ndarray, dt_hours: float = 1.0) -> dict:
        """
        Apply peak shaving to a load profile.

        Parameters
        ----------
        load_mw : np.ndarray
            Hourly network demand (MW), shape (N,).
        dt_hours : float
            Time step (hours).

        Returns
        -------
        dict with:
            modified_load_mw : np.ndarray — what the network actually sees
            tes_charge_mw : np.ndarray — positive = charging, negative = discharging
            tes_soc_mwh : np.ndarray — state of charge
            cap_mw : float — the cap value used
        """
        N = len(load_mw)
        load = np.array(load_mw, dtype=float)

        # Determine cap
        if self.cap_absolute_mw is not None:
            cap = self.cap_absolute_mw
        else:
            cap = np.percentile(load, self.cap_percentile)

        # TES parameters — sized to handle the peak excess
        peak_excess = max(load.max() - cap, 0)
        capacity = self.tes_capacity_mwh or max(peak_excess * 12, cap * 4)
        max_rate = self.tes_max_charge_rate_mw or max(peak_excess * 1.2, cap * 0.3)
        eta = self.tes_efficiency

        modified_load = np.copy(load)
        tes_charge = np.zeros(N)
        tes_soc = np.zeros(N)
        soc = capacity * 0.5  # Start at 50% SOC

        for i in range(N):
            if load[i] > cap:
                # Discharge TES to cover excess
                excess = load[i] - cap
                discharge = min(excess, max_rate, soc / dt_hours)
                tes_charge[i] = -discharge
                soc -= discharge * dt_hours
                modified_load[i] = load[i] - discharge
            else:
                # Charge TES with available headroom
                headroom = cap - load[i]
                charge = min(headroom, max_rate, (capacity - soc) / dt_hours / eta)
                tes_charge[i] = charge
                soc += charge * dt_hours * eta
                modified_load[i] = load[i] + charge  # Network sees higher load

            tes_soc[i] = soc

        return {
            "modified_load_mw": modified_load,
            "tes_charge_mw": tes_charge,
            "tes_soc_mwh": tes_soc,
            "cap_mw": cap,
        }


class LoadShiftingStrategy:
    """
    Load Shifting DSM Strategy.

    Identifies peak and off-peak periods, then:
    - During off-peak (e.g., 23:00–06:00): increase network demand to charge TES
    - During peak (e.g., 07:00–09:00, 17:00–20:00): reduce demand using TES

    This shifts demand from expensive peak-production hours (when fossil fuel
    is dispatched) to off-peak hours (when base-load waste heat is available).
    """

    def __init__(
        self,
        off_peak_hours: list = None,
        peak_hours: list = None,
        shift_fraction: float = 0.20,
        tes_capacity_mwh: float = None,
        tes_max_charge_rate_mw: float = None,
        tes_efficiency: float = 0.95,
    ):
        """
        Parameters
        ----------
        off_peak_hours : list of int
            Hours of the day (0–23) considered off-peak for charging.
            Default: [23, 0, 1, 2, 3, 4, 5, 6].
        peak_hours : list of int
            Hours of the day considered peak for discharging.
            Default: [7, 8, 9, 17, 18, 19, 20].
        shift_fraction : float
            Fraction of peak-hour demand to shift (0–1).
        tes_capacity_mwh : float or None
            TES capacity (MWh).
        tes_max_charge_rate_mw : float or None
            Max charge/discharge rate (MW).
        tes_efficiency : float
            Round-trip efficiency.
        """
        self.off_peak_hours = off_peak_hours or [23, 0, 1, 2, 3, 4, 5, 6]
        self.peak_hours = peak_hours or [7, 8, 9, 17, 18, 19, 20]
        self.shift_fraction = shift_fraction
        self.tes_capacity_mwh = tes_capacity_mwh
        self.tes_max_charge_rate_mw = tes_max_charge_rate_mw
        self.tes_efficiency = tes_efficiency

    def apply(
        self,
        load_mw: np.ndarray,
        hours_of_day: np.ndarray,
        dt_hours: float = 1.0,
    ) -> dict:
        """
        Apply load shifting to a load profile.

        Parameters
        ----------
        load_mw : np.ndarray
            Hourly network demand (MW).
        hours_of_day : np.ndarray
            Hour of day for each time step (0–23).
        dt_hours : float
            Time step (hours).

        Returns
        -------
        dict — same structure as PeakShavingStrategy.apply()
        """
        N = len(load_mw)
        load = np.array(load_mw, dtype=float)
        hours = np.array(hours_of_day, dtype=int)

        # Default TES sizing: enough to store shift_fraction × average daily peak energy
        mean_load = np.mean(load)
        capacity = self.tes_capacity_mwh or (mean_load * len(self.peak_hours) * self.shift_fraction * 2)
        max_rate = self.tes_max_charge_rate_mw or (mean_load * 0.5)
        eta = self.tes_efficiency

        modified_load = np.copy(load)
        tes_charge = np.zeros(N)
        tes_soc = np.zeros(N)
        soc = capacity * 0.5  # Start at 50%

        for i in range(N):
            h = hours[i]

            if h in self.peak_hours:
                # Discharge: reduce network demand
                target_reduction = load[i] * self.shift_fraction
                discharge = min(target_reduction, max_rate, soc / dt_hours)
                tes_charge[i] = -discharge
                soc -= discharge * dt_hours
                modified_load[i] = load[i] - discharge

            elif h in self.off_peak_hours:
                # Charge: increase network demand
                target_charge = mean_load * self.shift_fraction
                charge = min(target_charge, max_rate, (capacity - soc) / dt_hours / eta)
                tes_charge[i] = charge
                soc += charge * dt_hours * eta
                modified_load[i] = load[i] + charge

            # else: neutral hours — no action

            tes_soc[i] = soc

        return {
            "modified_load_mw": modified_load,
            "tes_charge_mw": tes_charge,
            "tes_soc_mwh": tes_soc,
            "cap_mw": None,
        }


class CombinedDSMStrategy:
    """
    Combined peak shaving + load shifting.

    Applies load shifting first, then peak shaving on the already-shifted profile.
    """

    def __init__(
        self,
        peak_shaving: PeakShavingStrategy = None,
        load_shifting: LoadShiftingStrategy = None,
    ):
        self.peak_shaving = peak_shaving or PeakShavingStrategy()
        self.load_shifting = load_shifting or LoadShiftingStrategy()

    def apply(
        self,
        load_mw: np.ndarray,
        hours_of_day: np.ndarray,
        dt_hours: float = 1.0,
    ) -> dict:
        """Apply load shifting first, then peak shaving."""
        # Step 1: Load shifting
        result_ls = self.load_shifting.apply(load_mw, hours_of_day, dt_hours)

        # Step 2: Peak shaving on the shifted profile
        result_ps = self.peak_shaving.apply(result_ls["modified_load_mw"], dt_hours)

        # Combine TES actions
        total_tes_charge = result_ls["tes_charge_mw"] + result_ps["tes_charge_mw"]

        return {
            "modified_load_mw": result_ps["modified_load_mw"],
            "tes_charge_mw": total_tes_charge,
            "tes_soc_mwh_shifting": result_ls["tes_soc_mwh"],
            "tes_soc_mwh_shaving": result_ps["tes_soc_mwh"],
            "cap_mw": result_ps["cap_mw"],
        }


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    hours = 24 * 7  # one week
    t = np.arange(hours)
    hour_of_day = t % 24

    # Synthetic load: base + daily pattern + noise
    base = 100  # MW
    daily = 40 * np.sin(np.pi * (hour_of_day - 6) / 12) * (hour_of_day >= 6) * (hour_of_day <= 22)
    load = base + daily + np.random.normal(0, 5, hours)
    load = np.clip(load, 20, None)

    print("=== Peak Shaving ===")
    ps = PeakShavingStrategy(cap_percentile=75)
    r = ps.apply(load)
    print(f"  Cap: {r['cap_mw']:.1f} MW")
    print(f"  Original peak: {load.max():.1f} MW → Shaved peak: {r['modified_load_mw'].max():.1f} MW")
    print(f"  Peak reduction: {(1 - r['modified_load_mw'].max()/load.max())*100:.1f}%")

    print("\n=== Load Shifting ===")
    ls = LoadShiftingStrategy(shift_fraction=0.15)
    r = ls.apply(load, hour_of_day)
    print(f"  Original peak: {load.max():.1f} MW → Shifted peak: {r['modified_load_mw'].max():.1f} MW")
    print(f"  Energy preserved: original={load.sum():.0f}, modified={r['modified_load_mw'].sum():.0f} MWh")

    print("\n=== Combined ===")
    combined = CombinedDSMStrategy()
    r = combined.apply(load, hour_of_day)
    print(f"  Original peak: {load.max():.1f} MW → Combined: {r['modified_load_mw'].max():.1f} MW")
