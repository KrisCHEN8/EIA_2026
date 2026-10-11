import numpy as np
import pandas as pd

PRIORITY_SOURCES = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]


def daily_schedule(signal, n_charge, n_discharge):
    """24 (or fewer) signal values for one day -> modes. Low signal = charge, high = discharge."""
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    modes = np.array(["direct"] * n, dtype=object)
    n_charge = min(n_charge, n)
    n_discharge = min(n_discharge, n - n_charge)
    order = np.argsort(signal, kind="stable")
    if n_charge > 0:
        modes[order[:n_charge]] = "charge"
    if n_discharge > 0:
        modes[order[n - n_discharge:]] = "discharge"
    return modes


def peak_shaving_signal(network_load_mw):
    """Signal = network load: charge in valleys, discharge at peaks."""
    return np.asarray(network_load_mw, dtype=float)


def production_aware_signal(dates, network_load_mw, daily_dispatch):
    """
    Signal = rank of the estimated marginal production source, with network load
    as tie-breaker (0 = waste heat ... 3 = fossil fuel).

    The marginal source per hour is estimated with a merit-order 'load-slice' view
    of each day: waste heat runs flat (daily total / hours, must-run), and the
    residual above it is filled bottom-up by electricity, biomass, then fossil,
    with each source's hourly slice chosen so its daily total is matched.

    NOTE: this only builds the CONTROL SIGNAL. The energy accounting still uses
    hourly_dispatch.py, which allocates sources proportionally within each day,
    so it may not credit the full effect of this strategy.
    """
    dates = pd.DatetimeIndex(dates)
    load = np.asarray(network_load_mw, dtype=float)
    dd = daily_dispatch.copy()
    dd["day"] = dd["date"].dt.date
    dd = dd.set_index("day")

    rank = np.zeros(len(load))
    days = dates.date
    for day in np.unique(days):
        idx = np.where(days == day)[0]
        if day not in dd.index:
            continue
        row = dd.loc[day]
        n = len(idx)
        L = load[idx]
        residual = np.clip(L - row["Waste heat"] / n, 0.0, None)
        # Thresholds stacked bottom-up: electricity, biomass, fossil on top
        level = np.zeros(n)
        r = np.zeros(n)
        for k, src in enumerate(PRIORITY_SOURCES[1:], start=1):
            target = row[src]
            if target <= 0:
                continue
            # find slice height h so that sum(clip(residual - level, 0, h)) = target
            avail = np.clip(residual - level, 0.0, None)
            if avail.sum() <= target:
                h = avail.max()
            else:
                lo, hi = 0.0, avail.max()
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if np.minimum(avail, mid).sum() < target:
                        lo = mid
                    else:
                        hi = mid
                h = hi
            r[avail > 1e-9] = k            # this source is (so far) the top one in these hours
            level = level + np.minimum(avail, h)
        rank[idx] = r

    # Tie-break by load: scale so rank dominates
    lmax = max(load.max(), 1e-9)
    return rank + 0.5 * load / lmax


def build_schedule(dates, signal, n_charge, n_discharge):
    """Apply daily_schedule day by day. Returns an array of modes aligned with dates."""
    dates = pd.DatetimeIndex(dates)
    signal = np.asarray(signal, dtype=float)
    modes = np.empty(len(signal), dtype=object)
    days = dates.date
    for day in np.unique(days):
        idx = np.where(days == day)[0]
        modes[idx] = daily_schedule(signal[idx], n_charge, n_discharge)
    return modes


def build_production_aware_schedule(dates, signal, n_charge, n_discharge,
                                    charge_max_rank=1, discharge_min_rank=2):
    """
    Within a day the merit-order marginal source rises with load, so ranking hours
    by marginal source alone gives the same hours as peak shaving. What differs is
    WHETHER shifting pays off:
      - discharge only in hours whose marginal source is expensive (rank >= discharge_min_rank,
        default biomass/fossil), at most n_discharge of the highest-signal hours;
      - charge only in hours whose marginal source is cheap (rank <= charge_max_rank,
        default waste heat/electricity), at most n_charge of the lowest-signal hours;
      - no qualifying discharge hour that day -> no charging either (the tank stays idle
        instead of cycling with losses and a higher return temperature for no gain).
    """
    dates = pd.DatetimeIndex(dates)
    signal = np.asarray(signal, dtype=float)
    rank = np.floor(signal)
    modes = np.array(["direct"] * len(signal), dtype=object)
    days = dates.date
    for day in np.unique(days):
        idx = np.where(days == day)[0]
        s, r = signal[idx], rank[idx]
        dis = [i for i in np.argsort(-s, kind="stable") if r[i] >= discharge_min_rank][:n_discharge]
        if not dis:
            continue
        ch = [i for i in np.argsort(s, kind="stable") if r[i] <= charge_max_rank and i not in dis][:n_charge]
        if not ch:
            continue
        modes[idx[dis]] = "discharge"
        modes[idx[ch]] = "charge"
    return modes


def make_schedule(strategy, dates, baseline_network_mw, daily_dispatch, n_charge, n_discharge,
                  charge_max_rank=1, discharge_min_rank=2):
    if strategy == "peak_shaving":
        signal = peak_shaving_signal(baseline_network_mw)
        modes = build_schedule(dates, signal, n_charge, n_discharge)
    elif strategy == "production_aware":
        signal = production_aware_signal(dates, baseline_network_mw, daily_dispatch)
        modes = build_production_aware_schedule(dates, signal, n_charge, n_discharge,
                                                charge_max_rank, discharge_min_rank)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return modes, signal
