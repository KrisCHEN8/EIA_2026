import numpy as np


CP_W = 4184.0       # J/(kg K), water
RHO_W = 988.0       # kg/m3, water at ~50 degC


#  Water tank
class WaterTank:
    def __init__(self, water_mass_kg=1000.0, initial_temperature=45.0, num_layers=10,
                 UA_loss=2.0, T_amb=18.0, T_ref=40.0):
        """
        water_mass_kg: total water mass (kg).
        initial_temperature: uniform initial temperature (degC), or array per layer.
        num_layers: number of layers; 10-20 keeps stratification without heavy numerical mixing.
        UA_loss: total heat-loss coefficient to ambient (W/K).
        T_amb: ambient (technical room) temperature (degC).
        T_ref: reference temperature for stored energy / state of charge (degC),
               normally the lowest useful temperature (about the radiator return).
        """
        self.M_water = float(water_mass_kg)
        self.num_layers = int(num_layers)
        self.M_layer = self.M_water / self.num_layers
        self.UA_loss_layer = UA_loss / self.num_layers
        self.T_amb = T_amb
        self.T_ref = T_ref
        self.T_water = np.full(self.num_layers, float(np.mean(initial_temperature)))
        if np.ndim(initial_temperature):
            self.T_water[:] = initial_temperature

        # Energy bookkeeping (J)
        self.E0 = self.stored_energy_J()
        self.E_port = 0.0          # net energy brought in through the ports
        self.E_loss = 0.0          # cumulative standing losses

    @classmethod
    def from_volume(cls, volume_l, **kwargs):
        return cls(water_mass_kg=volume_l / 1000.0 * RHO_W, **kwargs)

    # ---------- state ----------
    @property
    def T_top(self):
        return float(self.T_water[-1])

    @property
    def T_bottom(self):
        return float(self.T_water[0])

    @property
    def mean_temperature(self):
        return float(np.mean(self.T_water))

    def stored_energy_J(self):
        return float(np.sum(self.M_layer * CP_W * (self.T_water - self.T_ref)))

    def capacity_kwh(self, T_low, T_high):
        """Storage capacity between two uniform tank temperatures (kWh)."""
        return self.M_water * CP_W * (T_high - T_low) / 3.6e6

    def state_of_charge(self, T_low, T_high):
        """0 = whole tank at T_low, 1 = whole tank at T_high."""
        E_low = self.M_water * CP_W * (T_low - self.T_ref)
        E_high = self.M_water * CP_W * (T_high - self.T_ref)
        return float(np.clip((self.stored_energy_J() - E_low) / (E_high - E_low), 0.0, 1.0))

    def energy_residual(self):
        """Should stay ~0 (J): change in stored energy minus (port energy - losses)."""
        return (self.stored_energy_J() - self.E0) - (self.E_port - self.E_loss)

    # ---------- dynamics ----------
    def _n_substeps(self, m_tank, dt):
        # Advection (CFL): a layer may be flushed at most ~once per sub-step
        n_adv = abs(m_tank) * dt / (0.9 * self.M_layer)
        # Losses / internal exchange: sub-step well below the shortest time constant
        n_exch = dt / (0.5 * self._min_time_constant())
        return max(1, int(np.ceil(max(n_adv, n_exch))))

    def _min_time_constant(self):
        return self.M_layer * CP_W / max(self.UA_loss_layer, 1e-9)

    def _internal_exchange(self, dt):
        """Hook for the PCM tank (water <-> PCM). Nothing in a plain water tank."""
        return

    def _advect(self, m_tank, T_in, dt):
        """Upwind advection. Returns the outlet temperature of this sub-step."""
        if m_tank == 0.0:
            return np.nan
        T = self.T_water
        frac = abs(m_tank) * dt / self.M_layer            # <= 0.9 by construction
        T_new = T.copy()
        if m_tank > 0:   # charging: enters top, moves down, leaves bottom
            upstream = np.append(T[1:], T_in)
            T_out = T[0]
        else:            # discharging: enters bottom, moves up, leaves top
            upstream = np.insert(T[:-1], 0, T_in)
            T_out = T[-1]
        T_new += frac * (upstream - T)
        self.T_water = T_new
        return T_out

    def _losses(self, dt):
        Q = self.UA_loss_layer * (self.T_water - self.T_amb)      # W per layer
        self.T_water -= Q * dt / (self.M_layer * CP_W)
        self.E_loss += float(np.sum(Q)) * dt

    def _buoyancy_mix(self):
        """Mix out temperature inversions (warmer water below colder), conserving energy.
        Pool-adjacent-violators: equal-mass layers, so energy-weighted mean = plain mean."""
        T = self.T_water
        if np.all(np.diff(T) >= -1e-9):
            return
        blocks = []                      # [sum_T, n_layers]
        for t in T:
            blocks.append([t, 1])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                s, n = blocks.pop()
                blocks[-1][0] += s
                blocks[-1][1] += n
        self.T_water = np.concatenate([np.full(n, s / n) for s, n in blocks])

    def step(self, m_tank, T_in, dt):
        """
        Advance the tank by dt seconds.

        m_tank: kg/s, + = charging (enters top at T_in, leaves bottom),
                      - = discharging (enters bottom at T_in, leaves top), 0 = standby.
        T_in:   inlet temperature (degC); ignored when m_tank == 0.

        Returns dict:
            T_out:  flow-averaged outlet temperature over dt (nan if no flow)
            Q_port: mean power brought in through the ports (W); + charging, - discharging
            E_port: energy through the ports during dt (J)
        """
        m_tank = float(m_tank)
        n = self._n_substeps(m_tank, dt)
        h = dt / n
        T_out_sum = 0.0
        E_port_step = 0.0
        for _ in range(n):
            T_out = self._advect(m_tank, T_in, h)
            if m_tank != 0.0:
                E_port_step += abs(m_tank) * CP_W * (T_in - T_out) * h
                T_out_sum += T_out
            self._internal_exchange(h)
            self._losses(h)
            self._buoyancy_mix()
        self.E_port += E_port_step
        return dict(
            T_out=(T_out_sum / n) if m_tank != 0.0 else np.nan,
            Q_port=E_port_step / dt,
            E_port=E_port_step,
        )


#  PCM tank (water tank with PCM containers in every layer)
class PCMProperties:
    """
    Piecewise-linear enthalpy curve h(T) per kg PCM, h = 0 at T_base.

    Default: Rubitherm RT55 (melting range 51-57 degC).
    NOTE on latent_heat: Rubitherm datasheets give a 'heat storage capacity' that
    combines latent and sensible heat over a wider window (about 15 K). If you use
    that number, set latent_heat = capacity - cp * window to avoid double counting
    the sensible part. Check against your datasheet.
    """

    def __init__(self, T_solid=51.0, T_liquid=57.0, latent_heat=170000.0,
                 cp_solid=2000.0, cp_liquid=2000.0, rho_liquid=770.0, T_base=20.0):
        self.T_s, self.T_l = T_solid, T_liquid
        self.L = latent_heat
        self.cp_s, self.cp_l = cp_solid, cp_liquid
        self.rho_liquid = rho_liquid       # kg/m3, sets container volume (liquid is the larger)
        self.T_base = T_base
        self.h_s = cp_solid * (T_solid - T_base)       # h at onset of melting
        self.h_l = self.h_s + latent_heat              # h at end of melting

    def h(self, T):
        T = np.asarray(T, dtype=float)
        return np.where(
            T < self.T_s, self.cp_s * (T - self.T_base),
            np.where(T <= self.T_l,
                     self.h_s + self.L * (T - self.T_s) / (self.T_l - self.T_s),
                     self.h_l + self.cp_l * (T - self.T_l)))

    def T(self, h):
        h = np.asarray(h, dtype=float)
        return np.where(
            h < self.h_s, self.T_base + h / self.cp_s,
            np.where(h <= self.h_l,
                     self.T_s + (h - self.h_s) / self.L * (self.T_l - self.T_s),
                     self.T_l + (h - self.h_l) / self.cp_l))

    def liquid_fraction(self, h):
        return np.clip((np.asarray(h) - self.h_s) / self.L, 0.0, 1.0)


class PCMStorageTank(WaterTank):
    def __init__(self, water_mass_kg=500.0, pcm_mass_kg=578.0, initial_temperature=45.0,
                 num_layers=10, UA_pcm=2000.0, UA_min_fraction=0.3, pcm=None, **kwargs):
        """
        pcm_mass_kg: total PCM mass, spread evenly over the layers.
        UA_pcm: total water-PCM heat-transfer coefficient when the PCM is liquid (W/K).
        UA_min_fraction: UA multiplier when the PCM is fully solid during discharge
                         (solid crust on the container walls); UA scales linearly with
                         liquid fraction between UA_min_fraction and 1. While melting,
                         full UA is used. Run a sensitivity on UA_pcm and this value.
        pcm: PCMProperties (default RT55).
        """
        self.pcm = pcm or PCMProperties()
        self.M_pcm = float(pcm_mass_kg)
        self.UA_pcm = UA_pcm
        self.UA_min_fraction = UA_min_fraction
        n = int(num_layers)
        self.m_pcm_layer = self.M_pcm / n
        self.UA_pcm_layer = UA_pcm / n
        T0 = np.full(n, float(np.mean(initial_temperature)))
        if np.ndim(initial_temperature):
            T0[:] = initial_temperature
        self.h_pcm = self.pcm.h(T0)                    # J/kg, per layer
        super().__init__(water_mass_kg=water_mass_kg, initial_temperature=T0,
                         num_layers=n, **kwargs)

    @classmethod
    def from_volume(cls, volume_l, pcm_volume_fraction=0.5, pcm=None, **kwargs):
        """Same outer volume as a water tank: PCM containers displace water."""
        pcm = pcm or PCMProperties()
        V = volume_l / 1000.0
        return cls(water_mass_kg=V * (1 - pcm_volume_fraction) * RHO_W,
                   pcm_mass_kg=V * pcm_volume_fraction * pcm.rho_liquid,
                   pcm=pcm, **kwargs)

    # ---------- state ----------
    @property
    def T_pcm(self):
        return self.pcm.T(self.h_pcm)

    @property
    def liquid_fraction(self):
        return self.pcm.liquid_fraction(self.h_pcm)

    def stored_energy_J(self):
        E_w = np.sum(self.M_layer * CP_W * (self.T_water - self.T_ref))
        E_p = np.sum(self.m_pcm_layer * (self.h_pcm - self.pcm.h(self.T_ref)))
        return float(E_w + E_p)

    def capacity_kwh(self, T_low, T_high):
        water = self.M_water * CP_W * (T_high - T_low)
        pcm = self.M_pcm * float(self.pcm.h(T_high) - self.pcm.h(T_low))
        return (water + pcm) / 3.6e6

    def state_of_charge(self, T_low, T_high):
        E_ref = self.pcm.h(self.T_ref)
        E_low = self.M_water * CP_W * (T_low - self.T_ref) + self.M_pcm * float(self.pcm.h(T_low) - E_ref)
        E_high = self.M_water * CP_W * (T_high - self.T_ref) + self.M_pcm * float(self.pcm.h(T_high) - E_ref)
        return float(np.clip((self.stored_energy_J() - E_low) / (E_high - E_low), 0.0, 1.0))

    # ---------- dynamics ----------
    def _min_time_constant(self):
        tau_w = self.M_layer * CP_W / (self.UA_pcm_layer + self.UA_loss_layer)
        tau_p = self.m_pcm_layer * min(self.pcm.cp_s, self.pcm.cp_l) / self.UA_pcm_layer
        return min(tau_w, tau_p)

    def _internal_exchange(self, dt):
        T_p = self.T_pcm
        dT = self.T_water - T_p                         # > 0: water heats PCM (melting)
        f_liq = self.liquid_fraction
        ua_factor = np.where(dT >= 0, 1.0,
                             self.UA_min_fraction + (1 - self.UA_min_fraction) * f_liq)
        Q = self.UA_pcm_layer * ua_factor * dT          # W per layer, water -> PCM
        self.T_water -= Q * dt / (self.M_layer * CP_W)
        self.h_pcm += Q * dt / self.m_pcm_layer


#  Coupling helper: one hour of substation + tank at sub-hourly resolution
def simulate_hour(substation, tank, mode, T_p_in, T_s_set, Q_demand_kw,
                  Q_tank_target_kw=0.0, n_sub=6, dt_hour=3600.0,
                  T_charge_set=65.0, T_bottom_cutoff=None, T_top_cutoff=None):
    """
    Run one hour split into n_sub intervals, so the substation always sees the
    current tank top/bottom temperature (they change noticeably within an hour).

    mode: "direct", "charge" or "discharge" (decided by the control strategy).
    Q_tank_target_kw: charging power (charge) or discharge power (discharge), kW, >= 0.

    Returns hourly results: energies in kWh, temperatures flow-weighted.
    """
    dt = dt_hour / n_sub
    acc = dict(Q_primary_kwh=0.0, Q_delivered_kwh=0.0, Q_tank_sub_kwh=0.0, Q_tank_kwh=0.0,
               unmet_kwh=0.0, mp_T=0.0, mp_sum=0.0, ms_Tr=0.0, ms_sum=0.0, saturated=0,
               n_charge=0, n_discharge=0)

    for _ in range(n_sub):
        if mode == "charge" and tank is not None:
            r = substation.step_charge(T_p_in, T_s_set, Q_demand_kw, tank.T_bottom,
                                       Q_tank_target_kw, T_charge_set, T_bottom_cutoff)
        elif mode == "discharge" and tank is not None:
            r = substation.step_discharge(T_p_in, T_s_set, Q_demand_kw, tank.T_top,
                                          Q_tank_target_kw, T_top_cutoff)
        else:
            r = substation.step_direct(T_p_in, T_s_set, Q_demand_kw)

        if tank is not None:
            t = tank.step(r["m_tank"], r["T_tank_in"], dt)
            acc["Q_tank_kwh"] += t["E_port"] / 3.6e6       # what the tank actually absorbed
        h = dt / 3600.0
        acc["Q_primary_kwh"] += r["Q_primary"] * h
        acc["Q_delivered_kwh"] += r["Q_delivered"] * h
        acc["Q_tank_sub_kwh"] += r["Q_tank"] * h          # what the substation assumed
        acc["unmet_kwh"] += max(Q_demand_kw - r["Q_delivered"], 0.0) * h
        acc["mp_T"] += r["m_p"] * r["T_p_out"]
        acc["mp_sum"] += r["m_p"]
        acc["ms_Tr"] += r["m_s"] * r["T_s_return"]
        acc["ms_sum"] += r["m_s"]
        acc["saturated"] += int(r["saturated"])
        acc["n_charge"] += int(r["mode"] == "charge")
        acc["n_discharge"] += int(r["mode"] == "discharge")

    out = dict(
        Q_primary_kw=acc["Q_primary_kwh"],                 # kWh over 1 h = mean kW
        Q_delivered_kw=acc["Q_delivered_kwh"],
        Q_tank_kw=acc["Q_tank_kwh"],
        Q_tank_substation_kw=acc["Q_tank_sub_kwh"],        # compare with Q_tank_kw: coupling error
        unmet_kwh=acc["unmet_kwh"],
        m_p_mean=acc["mp_sum"] / n_sub,                    # kg/s, for network flow weighting
        T_p_out=acc["mp_T"] / acc["mp_sum"] if acc["mp_sum"] > 0 else np.nan,
        T_s_return=acc["ms_Tr"] / acc["ms_sum"] if acc["ms_sum"] > 0 else np.nan,
        frac_saturated=acc["saturated"] / n_sub,
        frac_charge=acc["n_charge"] / n_sub,
        frac_discharge=acc["n_discharge"] / n_sub,
    )
    if tank is not None:
        out.update(T_top=tank.T_top, T_bottom=tank.T_bottom, T_mean=tank.mean_temperature,
                   T_layers=tank.T_water.copy())
        if hasattr(tank, "h_pcm"):
            out.update(T_pcm_mean=float(np.mean(tank.T_pcm)),
                       liquid_fraction=float(np.mean(tank.liquid_fraction)))
    return out