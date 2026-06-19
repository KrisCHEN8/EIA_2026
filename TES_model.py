class PCMModel:
    def __init__(self, volume_liters=None, mass_kg=None, initial_temperature=40.0):
        # Physical Properties from Rubitherm RT65
        self.cp_standard = 2000.0     # 2 kJ/kg·K baseline specific heat (J/kg·K)
        self.T_solid = 58.0           # Start of melting zone (°C)
        self.T_liquid = 65.0          # End of melting zone (°C)
        self.dT = self.T_liquid - self.T_solid

        # Combined Latent + Sensible heat capacity inside the broader window (J/kg)
        self.total_transition_energy = 150000.0 
        
        # Determine Mass (M_pcm)
        if mass_kg:
            self.mass = mass_kg
        elif volume_liters:
            self.mass = volume_liters * 0.83 
        else:
            self.mass = 60.2   # kg

        # Effective specific heat during phase change
        self.cp_phase_change = self.total_transition_energy / self.dT
        
        # Initial temperature
        self.T = initial_temperature

    def get_effective_cp(self, T):
        """Returns effective specific heat capacity based on the current temperature."""
        if self.T_solid <= T <= self.T_liquid:
            return self.cp_phase_change
        else:
            return self.cp_standard


class PCMStorageTank:
    def __init__(self, water_mass_kg=50.0, pcm_mass_kg=57.8, initial_temperature=40.0):
        self.M_water = water_mass_kg
        self.cp_water = 4184.0   # J/kg·K
        self.T_water = initial_temperature  # Initial water tank temperature (°C)
        
        self.pcm = PCMModel(mass_kg=pcm_mass_kg, initial_temperature=initial_temperature)
        
        # Heat transfer coefficient * surface area coupling water and PCM tube (W/K)
        self.UA_pcm = 50.0

        # Ambient loss coefficient (W/K) and ambient temperature
        self.UA_loss = 2.0
        self.T_amb = 22.0

    def step(self, m_dot, T_in, dt):
        """
        Advances the state of the tank by one time step.
        
        Args:
            m_dot: Mass flow rate through the storage (kg/s)
                    Same flow enters from heat exchanger and exits to building.
            T_in: Temperature of water entering from heat exchanger (°C)
            dt: Time step (seconds)
        
        Returns:
            T_out: Temperature of water leaving to the building (= T_water, well mixed)
        """
        # The temperature of water leaving the tank to the building is the tank temperature (well mixed)
        T_out = self.T_water

        # Current properties
        cp_pcm_eff = self.pcm.get_effective_cp(self.pcm.T)
        
        # Heat exchange between tank water and PCM (W)
        Q_pcm = self.UA_pcm * (self.T_water - self.pcm.T)
        
        # Heat loss to ambient environment (W)
        Q_loss = self.UA_loss * (self.T_water - self.T_amb)

        # Energy carried in by hot water from heat exchanger (W)
        Q_in = m_dot * self.cp_water * T_in

        # Energy carried out by water leaving to building at tank temperature (W)
        Q_out = m_dot * self.cp_water * self.T_water

        # Total energy balance on the tank water (W)
        dT_water_dt = (Q_in - Q_out - Q_pcm - Q_loss) / (self.M_water * self.cp_water)
        dT_pcm_dt = Q_pcm / (self.pcm.mass * cp_pcm_eff)

        # Update states
        self.T_water += dT_water_dt * dt
        self.pcm.T += dT_pcm_dt * dt

        return T_out


class WaterTank:
    def __init__(self, water_mass_kg=50.0, initial_temperature=40.0):
        self.M_water = water_mass_kg
        self.cp_water = 4184.0   # J/kg·K
        self.T_water = initial_temperature  # Initial water tank temperature (°C)

        # Ambient loss coefficient (W/K) and ambient temperature
        self.UA_loss = 2.0
        self.T_amb = 22.0

    def step(self, m_dot, T_in, dt):
        """
        Advances the state of the tank by one time step.

        Args:
            m_dot: Mass flow rate through the storage (kg/s)
                    Same flow enters from heat exchanger and exits to building.
            T_in: Temperature of water entering from heat exchanger (°C)
            dt: Time step (seconds)
        
        Returns:
            T_out: Temperature of water leaving to the building (= T_water, well mixed)
        """
        # The temperature of water leaving the tank to the building is the tank temperature (well mixed)
        T_out = self.T_water
        
        # Heat loss to ambient environment (W)
        Q_loss = self.UA_loss * (self.T_water - self.T_amb)

        # Energy carried in by hot water from heat exchanger (W)
        Q_in = m_dot * self.cp_water * T_in

        # Energy carried out by water leaving to building at tank temperature (W)
        Q_out = m_dot * self.cp_water * self.T_water

        # Total energy balance on the tank water (W)
        dT_water_dt = (Q_in - Q_out - Q_loss) / (self.M_water * self.cp_water)

        # Update states
        self.T_water += dT_water_dt * dt

        return T_out


if __name__ == "__main__":
    pcm_tank = PCMStorageTank(water_mass_kg=60.0, pcm_mass_kg=60.2)
    water_tank = WaterTank(water_mass_kg=60.0)
    
    dt = 60.0          # Time step (seconds)
    hours = 8          # hours simulation
    total_time = hours * 3600     # seconds simulation
    time_steps = int(total_time / dt)

    m_dot_supply = 0.56    # kg/s
    T_supply = 75.0        # °C
    
    print(f"{'Time (mins)':<12}{'PCM Storage (°C)':<18}{'Water Tank (°C)':<20}")
    print("-" * 66)

    for step in range(time_steps):
        T_out_pcm = pcm_tank.step(
            m_dot=m_dot_supply,
            T_in=T_supply,
            dt=dt
        )
        T_out_water = water_tank.step(
            m_dot=m_dot_supply,
            T_in=T_supply,
            dt=dt
        )

        # Print logs every 15 minutes
        if (step * dt) % 900 == 0:
            mins = int((step * dt) / 60)
            print(f"{mins:<12}{pcm_tank.T_water:<18.2f}{water_tank.T_water:<20.2f}")
