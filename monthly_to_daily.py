import pandas as pd
import numpy as np
import calendar
from pathlib import Path


def _dispatch_one_source(month_df, source_name, monthly_source_mwh):
    """
    Distribute *monthly_source_mwh* across days in proportion to remaining
    load.  Implements the iterative priority-dispatch algorithm from
    daily_new_rule.py.
    """
    monthly_source_mwh = float(monthly_source_mwh)
    month_df[source_name] = 0.0

    if monthly_source_mwh <= 0:
        month_df[f"{source_name}_unused_monthly_mwh"] = 0.0
        return month_df

    tolerance = 1e-9
    max_iter = 1_000
    remaining = monthly_source_mwh

    for _ in range(max_iter):
        if remaining <= tolerance:
            break

        active_mask = month_df["remaining_load"] > tolerance
        if not active_mask.any():
            break

        daily_limit = remaining / active_mask.sum()
        add_amount = np.minimum(
            month_df.loc[active_mask, "remaining_load"],
            daily_limit,
        )

        month_df.loc[active_mask, source_name] += add_amount
        month_df.loc[active_mask, "remaining_load"] -= add_amount

        used = add_amount.sum()
        remaining -= used

        if used <= tolerance:
            break

    month_df[f"{source_name}_unused_monthly_mwh"] = remaining

    return month_df


def monthly_to_daily(
    monthly_df,
    weather_file,
    *,
    year=None,
    priority_sources=None,
    base_load_index=6.6,
    balance_temperature_C=17.0,
    weather_date_col="date",
    weather_temp_col="temperature",
    weather_sep=",",
    weather_decimal=".",
):
    """
    Convert a monthly energy-mix DataFrame to a daily dispatch DataFrame.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly energy data.  Must contain ``month_col`` (format "YYYY-MM")
        and one column per energy source listed in ``priority_sources``.
    weather_file : str or Path
        Path to the CSV weather file (same format as used in
        daily_new_rule.py).
    year : int, optional
        Filter weather data to this year.  If None, the year is inferred
        from the ``month`` column of ``monthly_df``.
    priority_sources : list of str, optional
        Ordered list of energy-source column names (high → low priority).
        Defaults to ["Waste heat", "Electricity", "Biomass", "Fossil fuel"].
    base_load_index : float
        Minimum load index applied every day (default 6.6).
    balance_temperature_C : float
        Temperature above which load is at base level (default 17 °C).
    weather_date_col : str
        Raw column name for the date in the weather CSV.
    weather_temp_col : str
        Raw column name for the temperature in the weather CSV.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame with one row per calendar day.
    """
    if priority_sources is None:
        priority_sources = ["Waste heat", "Electricity", "Biomass", "Fossil fuel"]

    # Parse the monthly DataFrame
    monthly_df = monthly_df.copy()
    monthly_df['month'] = pd.to_datetime(
        monthly_df['month'].astype(str).str.strip(), format="%Y-%m"
    )

    if year is None:
        # Infer from monthly_df for backward-compat; not used for filtering
        year = int(monthly_df['month'].dt.year.mode()[0])

    for src in priority_sources:
        if src not in monthly_df.columns:
            raise ValueError(
                f"Column '{src}' not found in monthly_df. "
                f"Available columns: {list(monthly_df.columns)}"
            )
        monthly_df[src] = pd.to_numeric(monthly_df[src], errors="coerce").fillna(0)

    # Build a quick lookup: (year, month_number) -> {source: mwh}
    monthly_lookup: dict[tuple[int, int], dict[str, float]] = {}
    for _, row in monthly_df.iterrows():
        key = (row['month'].year, row['month'].month)
        monthly_lookup[key] = {src: float(row[src]) for src in priority_sources}

    # Load weather data
    weather = pd.read_csv(weather_file, sep=weather_sep, decimal=weather_decimal)
    weather.columns = [str(c).strip() for c in weather.columns]

    weather = weather.rename(columns={
        weather_date_col: "date",
        weather_temp_col: "outdoor_temp_C",
    })

    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather["outdoor_temp_C"] = pd.to_numeric(
        weather["outdoor_temp_C"], errors="coerce"
    )

    weather = weather.dropna(subset=["date", "outdoor_temp_C"]).copy()

    # If the weather data is sub-daily (e.g. hourly), resample to daily mean
    if weather["date"].dt.floor("D").nunique() < len(weather):
        weather = (
            weather
            .set_index("date")
            .resample("D")["outdoor_temp_C"]
            .mean()
            .reset_index()
        )

    weather["month"] = weather["date"].dt.month
    weather["year"] = weather["date"].dt.year

    weather["daily_load_index"] = (
        base_load_index
        + (balance_temperature_C - weather["outdoor_temp_C"]).clip(lower=0)
    )

    # Dispatch month by month
    all_months: list[pd.DataFrame] = []

    for (yr, mo), energy in monthly_lookup.items():
        month_df = weather[
            (weather["year"] == yr) & (weather["month"] == mo)
        ].copy()

        if month_df.empty:
            raise ValueError(
                f"No weather data found for {yr}-{mo:02d}. "
                "Check that the weather file covers the same year."
            )

        monthly_total_mwh = sum(energy.values())

        load_index_sum = month_df["daily_load_index"].sum()
        if load_index_sum <= 0:
            month_df["daily_total_mwh"] = monthly_total_mwh / len(month_df)
        else:
            month_df["daily_total_mwh"] = (
                month_df["daily_load_index"] / load_index_sum * monthly_total_mwh
            )

        month_df["remaining_load"] = month_df["daily_total_mwh"]

        for src in priority_sources:
            month_df = _dispatch_one_source(month_df, src, energy[src])

        # Fossil fuel (last in priority) absorbs any numerical residual
        unmet = month_df["remaining_load"].clip(lower=0)
        month_df[priority_sources[-1]] += unmet
        month_df["remaining_load"] = 0.0

        all_months.append(month_df)

    # Concatenate and add sanity-check columns
    daily = pd.concat(all_months, ignore_index=True).sort_values("date")

    # daily["check_sum_sources_mwh"] = daily[priority_sources].sum(axis=1)
    # daily["check_error_daily_mwh"] = (
    #     daily["check_sum_sources_mwh"] - daily["daily_total_mwh"]
    # )

    return daily


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    mix_file = "./DH production mix/four_category.xlsx"
    sample_monthly = pd.read_excel(mix_file, sheet_name='agg')

    years = range(2021, 2026)

    for yr in years:
        daily_df = monthly_to_daily(
            monthly_df=sample_monthly,
            year=yr,
            weather_file="./DH production mix/weather.csv"
        )

        daily_df.to_csv(f"./DH production mix/daily_{yr}.csv", index=False)

    print("Daily data has been generated.")
