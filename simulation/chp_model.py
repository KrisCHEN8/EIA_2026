import numpy as np


class CHPModel:
    """
    CHP plant model for a waste-to-energy back-pressure turbine.

    Computes the change in electrical efficiency and power output
    as a function of the DH return temperature relative to a reference.
    """

    def __init__(
        self,
        eta_el_ref = 0.22,
        eta_th_ref = 0.68,
        T_return_ref = 50.0,
        delta_eta_per_degC = 0.0015,
        T_steam = 400.0,
        P_steam_bar = 40.0,
        fuel_cost_nok_per_mwh = 250.0,
        co2_factor_waste_kg_per_mwh = 230.0,
        co2_factor_fossil_kg_per_mwh = 280.0,
        co2_factor_biomass_kg_per_mwh = 20.0,
        electricity_price_nok_per_mwh = 500.0,
    ):
        """
        eta_el_ref: Reference electrical efficiency at T_return_ref.
            Typical WtE CHP: 0.18-0.25.
        eta_th_ref: Reference thermal efficiency at T_return_ref.
            Typical WtE CHP: 0.65-0.75.
        T_return_ref: Reference return temperature for the baseline efficiencies (°C).
        delta_eta_per_degC: Electrical efficiency improvement per 1°C reduction in T_return.
            Literature range: 0.10-0.20% per °C → 0.001-0.002.
        T_steam: Live steam temperature (°C), for Carnot reference only.
        fuel_cost_nok_per_mwh: Fuel cost for waste/biomass (NOK/MWh_fuel).
        co2_factor_waste_kg_per_mwh: CO2 emission factor for waste heat source (kg CO2/MWh_thermal).
        co2_factor_fossil_kg_per_mwh: CO2 emission factor for fossil fuel (kg CO2/MWh_thermal).
        co2_factor_biomass_kg_per_mwh: CO2 emission factor for biomass (kg CO2/MWh_thermal).
        electricity_price_nok_per_mwh: Electricity spot price for valuing additional CHP power output.
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

    def compute_efficiency(self, T_return):
        """
        Compute CHP electrical & thermal efficiency at a given return temperature.

        Parameters
        T_return: Actual DH return temperature (°C).

        Returns:
            eta_el: electrical efficiency
            eta_th: thermal efficiency
            eta_total: total (first-law) efficiency
            sigma: power-to-heat ratio
            delta_T_return: change from reference (positive = reduction)
            delta_eta_el: change in electrical efficiency
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
    ):
        """
        Compute the economic and environmental benefits of reducing T_return.

        Parameters
        Q_heat_mwh: Total thermal energy delivered (MWh).
        T_return_baseline: Average return temperature in baseline scenario (°C).
        T_return_scenario: Average return temperature in the improved scenario (°C).
        fossil_mwh, waste_mwh, biomass_mwh: Energy from each source (MWh).

        Returns:
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

        # CO2 from production mix
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
        T_dew: Flue gas dew point temperature (°C).
        max_recovery_fraction: Maximum additional heat recovery as fraction of base thermal output.
        reference_delta_T: ΔT below dew point at which max recovery is achieved.
        """
        self.T_dew = T_dew
        self.max_recovery = max_recovery_fraction
        self.ref_delta_T = reference_delta_T

    def compute_recovery(self, T_return: float, Q_base_kw: float) -> float:
        """
        Compute additional heat recovered from FGC (kW).

        Parameters
        T_return: DH return temperature (°C).
        Q_base_kw: Base thermal output without FGC (kW).

        Returns
        float: additional heat recovery (kW).
        """
        if T_return >= self.T_dew:
            return 0.0

        delta_T = self.T_dew - T_return
        fraction = min(delta_T / self.ref_delta_T, 1.0) * self.max_recovery
        return fraction * Q_base_kw
