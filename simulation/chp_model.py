"""
CHP Efficiency Model
====================
Models the impact of DH return temperature on CHP plant performance.

Based on the Trondheim DH system (Heimdal waste-to-energy plant with
back-pressure turbine and electric boilers). The "Electricity" source in
the production mix represents electric boilers / heat pumps, while
"Waste heat" includes the waste incineration CHP.

Thermodynamic basis (back-pressure turbine):
    - The turbine exhaust pressure is set by the DH condenser temperature,
      which is determined by the DH return temperature.
    - Lower T_return → lower condensing pressure → larger enthalpy drop
      → more electricity per unit of heat.

Key relationship:
    The power-to-heat ratio σ of a back-pressure turbine can be approximated as:
        σ = η_el / η_th ≈ 1 - T_cond / T_steam
    where T_cond depends on T_return.

    More practically, the electrical efficiency gain per °C reduction in
    return temperature is approximately:
        Δη_el ≈ 0.15% per °C  (typical for waste-to-energy CHP)

    This is a well-documented linearized approximation from Nordic DH
    literature (Frederiksen & Werner, "District Heating and Cooling", 2013).

References:
    - Frederiksen & Werner, "District Heating and Cooling", Studentlitteratur, 2013
    - Dalla Rosa et al., "Low-temperature DH", Energy, 2011
    - Lund et al., "4th Generation District Heating", Energy, 2014
"""

import numpy as np


class CHPModel:
    """
    CHP plant model for a waste-to-energy back-pressure turbine.

    Computes the change in electrical efficiency and power output
    as a function of the DH return temperature relative to a reference.
    """

    def __init__(
        self,
        eta_el_ref: float = 0.22,
        eta_th_ref: float = 0.68,
        T_return_ref: float = 50.0,
        delta_eta_per_degC: float = 0.0015,
        T_steam: float = 400.0,
        P_steam_bar: float = 40.0,
        fuel_cost_nok_per_mwh: float = 250.0,
        co2_factor_waste_kg_per_mwh: float = 230.0,
        co2_factor_fossil_kg_per_mwh: float = 280.0,
        co2_factor_biomass_kg_per_mwh: float = 20.0,
        electricity_price_nok_per_mwh: float = 500.0,
    ):
        """
        Parameters
        ----------
        eta_el_ref : float
            Reference electrical efficiency at T_return_ref.
            Typical WtE CHP: 0.18–0.25.
        eta_th_ref : float
            Reference thermal efficiency at T_return_ref.
            Typical WtE CHP: 0.65–0.75.
        T_return_ref : float
            Reference return temperature for the baseline efficiencies (°C).
        delta_eta_per_degC : float
            Electrical efficiency improvement per 1°C reduction in T_return.
            Literature range: 0.10–0.20% per °C → 0.001–0.002.
        T_steam : float
            Live steam temperature (°C), for Carnot reference only.
        fuel_cost_nok_per_mwh : float
            Fuel cost for waste/biomass (NOK/MWh_fuel).
        co2_factor_waste_kg_per_mwh : float
            CO₂ emission factor for waste heat source (kg CO₂/MWh_thermal).
        co2_factor_fossil_kg_per_mwh : float
            CO₂ emission factor for fossil fuel (kg CO₂/MWh_thermal).
        co2_factor_biomass_kg_per_mwh : float
            CO₂ emission factor for biomass (kg CO₂/MWh_thermal).
        electricity_price_nok_per_mwh : float
            Electricity spot price for valuing additional CHP power output.
        """
        self.eta_el_ref = eta_el_ref
        self.eta_th_ref = eta_th_ref
        self.T_return_ref = T_return_ref
        self.delta_eta = delta_eta_per_degC
        self.T_steam = T_steam
        self.P_steam_bar = P_steam_bar

        # Cost & emission factors
        self.fuel_cost = fuel_cost_nok_per_mwh
        self.co2_waste = co2_factor_waste_kg_per_mwh
        self.co2_fossil = co2_factor_fossil_kg_per_mwh
        self.co2_biomass = co2_factor_biomass_kg_per_mwh
        self.elec_price = electricity_price_nok_per_mwh

    def compute_efficiency(self, T_return: float) -> dict:
        """
        Compute CHP electrical & thermal efficiency at a given return temperature.

        Parameters
        ----------
        T_return : float
            Actual DH return temperature (°C).

        Returns
        -------
        dict with:
            eta_el : float — electrical efficiency
            eta_th : float — thermal efficiency
            eta_total : float — total (first-law) efficiency
            sigma : float — power-to-heat ratio
            delta_T_return : float — change from reference (positive = reduction)
            delta_eta_el : float — change in electrical efficiency
        """
        delta_T = self.T_return_ref - T_return  # positive if T_return decreased
        delta_eta_el = self.delta_eta * delta_T

        eta_el = np.clip(self.eta_el_ref + delta_eta_el, 0.05, 0.40)
        eta_th = self.eta_th_ref  # thermal efficiency largely unchanged
        eta_total = eta_el + eta_th
        sigma = eta_el / eta_th if eta_th > 0 else 0.0

        return {
            "eta_el": eta_el,
            "eta_th": eta_th,
            "eta_total": eta_total,
            "sigma": sigma,
            "delta_T_return": delta_T,
            "delta_eta_el": delta_eta_el,
        }

    def compute_benefits(
        self,
        Q_heat_mwh: float,
        T_return_baseline: float,
        T_return_scenario: float,
        fossil_mwh: float = 0.0,
        waste_mwh: float = 0.0,
        biomass_mwh: float = 0.0,
    ) -> dict:
        """
        Compute the economic and environmental benefits of reducing T_return.

        Parameters
        ----------
        Q_heat_mwh : float
            Total thermal energy delivered (MWh).
        T_return_baseline : float
            Average return temperature in baseline scenario (°C).
        T_return_scenario : float
            Average return temperature in the improved scenario (°C).
        fossil_mwh, waste_mwh, biomass_mwh : float
            Energy from each source (MWh).

        Returns
        -------
        dict with benefit metrics.
        """
        eff_base = self.compute_efficiency(T_return_baseline)
        eff_new = self.compute_efficiency(T_return_scenario)

        # Additional electricity from CHP due to improved efficiency
        # Q_fuel_CHP = Q_heat / eta_th (fuel input to CHP)
        Q_fuel_chp = waste_mwh / eff_base["eta_th"] if eff_base["eta_th"] > 0 else 0
        extra_elec_mwh = Q_fuel_chp * (eff_new["eta_el"] - eff_base["eta_el"])

        # Revenue from additional electricity
        extra_revenue_nok = extra_elec_mwh * self.elec_price

        # CO₂ from production mix
        co2_baseline = (
            waste_mwh * self.co2_waste
            + fossil_mwh * self.co2_fossil
            + biomass_mwh * self.co2_biomass
        )

        return {
            "delta_T_return": T_return_baseline - T_return_scenario,
            "eta_el_baseline": eff_base["eta_el"],
            "eta_el_scenario": eff_new["eta_el"],
            "delta_eta_el": eff_new["eta_el"] - eff_base["eta_el"],
            "extra_electricity_mwh": extra_elec_mwh,
            "extra_revenue_nok": extra_revenue_nok,
            "co2_baseline_tonnes": co2_baseline / 1000.0,
        }


class FlueGasCondensationModel:
    """
    Optional: Flue gas condensation (FGC) heat recovery model.

    FGC allows recovering latent heat from flue gases when the DH return
    temperature is low enough (below the flue gas dew point, ~55–60°C for
    waste incineration).

    Additional heat recovery:
        Q_FGC = ε × m_flue × (h_fg_in - h_fg_out)

    Simplified as a function of return temperature:
        If T_return < T_dew: Q_FGC ∝ (T_dew - T_return)
        If T_return >= T_dew: Q_FGC = 0
    """

    def __init__(
        self,
        T_dew: float = 57.0,
        max_recovery_fraction: float = 0.12,
        reference_delta_T: float = 20.0,
    ):
        """
        Parameters
        ----------
        T_dew : float
            Flue gas dew point temperature (°C).
        max_recovery_fraction : float
            Maximum additional heat recovery as fraction of base thermal output.
        reference_delta_T : float
            ΔT below dew point at which max recovery is achieved.
        """
        self.T_dew = T_dew
        self.max_recovery = max_recovery_fraction
        self.ref_delta_T = reference_delta_T

    def compute_recovery(self, T_return: float, Q_base_kw: float) -> float:
        """
        Compute additional heat recovered from FGC (kW).

        Parameters
        ----------
        T_return : float
            DH return temperature (°C).
        Q_base_kw : float
            Base thermal output without FGC (kW).

        Returns
        -------
        float : additional heat recovery (kW).
        """
        if T_return >= self.T_dew:
            return 0.0

        delta_T = self.T_dew - T_return
        fraction = min(delta_T / self.ref_delta_T, 1.0) * self.max_recovery
        return fraction * Q_base_kw


if __name__ == "__main__":
    chp = CHPModel()
    fgc = FlueGasCondensationModel()

    print("=== CHP Efficiency vs Return Temperature ===")
    print(f"{'T_return':>8} {'η_el':>6} {'η_th':>6} {'η_tot':>6} {'σ':>6} {'Δη_el':>8} {'FGC%':>6}")
    print("-" * 52)
    for t_ret in [60, 55, 50, 45, 40, 35, 30]:
        r = chp.compute_efficiency(t_ret)
        fgc_frac = fgc.compute_recovery(t_ret, 1000.0) / 1000.0 * 100
        print(
            f"{t_ret:8.0f} {r['eta_el']:6.3f} {r['eta_th']:6.3f} "
            f"{r['eta_total']:6.3f} {r['sigma']:6.3f} {r['delta_eta_el']:+8.4f} "
            f"{fgc_frac:6.1f}"
        )

    print("\n=== Benefit Example ===")
    b = chp.compute_benefits(
        Q_heat_mwh=500000,
        T_return_baseline=50,
        T_return_scenario=42,
        waste_mwh=400000,
        fossil_mwh=50000,
        biomass_mwh=50000,
    )
    for k, v in b.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
