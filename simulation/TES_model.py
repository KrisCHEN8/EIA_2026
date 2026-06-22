class PCMModel:
    def __init__(self, volume_liters=None, mass_kg=None, initial_temperature=40.0):
        # Specific heat capacity of solid phase (at 15 °C): 2 kJ/kg·K -> 2000 J/kg·K
        self.cp_standard = 2000.0     

        # Actual phase change temperature boundaries from Rubitherm RT55 datasheet
        self.T_solid = 51.0           # Start of melting area (°C)
        self.T_liquid = 57.0          # End of melting/congealing area (°C)
        self.dT = self.T_liquid - self.T_solid

        # Latent heat storage capacity: 170 kJ/kg -> 170,000 J/kg
        self.latent_heat_energy = 170000.0

        # Effective specific heat during phase change (Latent heat spread over the dT window)
        # Added to the base sensible heat capacity to get total capacity in the transition zone
        self.cp_phase_change = self.cp_standard + (self.latent_heat_energy / self.dT)

        # Determine Mass (M_pcm) based on liquid phase density (0.77 kg/l)
        if mass_kg is not None:
            self.mass = mass_kg
        elif volume_liters is not None:
            self.mass = volume_liters * 0.77  # 0.77 kg/l density at 80 °C
        else:
            self.mass = 60.2   # Default fallback (kg)

        # Initial temperature
        self.T = initial_temperature

    def get_effective_cp(self, T):
        """Returns effective specific heat capacity (J/kg·K) based on the current
        temperature and the RT55 phase change window"""
        if self.T_solid <= T <= self.T_liquid:
            return self.cp_phase_change
        else:
            return self.cp_standard


class PCMStorageTank:
    def __init__(self, water_mass_kg=50.0, pcm_mass_kg=57.8, initial_temperature=50.0):
        self.M_water = water_mass_kg
        self.cp_water = 4184.0   # J/kg·K
        self.T_water = initial_temperature  # Initial water tank temperature (°C)
        
        self.pcm = PCMModel(mass_kg=pcm_mass_kg, initial_temperature=initial_temperature)
        
        # Heat transfer coefficient * surface area coupling water and PCM tube (W/K)
        self.UA_pcm = 2000.0      # W/K

        # Ambient loss coefficient (W/K) and ambient temperature
        self.UA_loss = 2.0
        self.T_amb = 22.0

    def step(self, m_dot, T_in, dt):
        """
        Advances the state of the tank by one time step.
        """
        T_out = self.T_water

        # Current temperature-dependent PCM effective heat capacity
        cp_pcm_eff = self.pcm.get_effective_cp(self.pcm.T)
        
        # Heat exchange between tank water and PCM (W)
        Q_pcm = self.UA_pcm * (self.T_water - self.pcm.T)
        
        # Heat loss to ambient environment (W)
        Q_loss = self.UA_loss * (self.T_water - self.T_amb)

        # Energy carried in by hot water from heat exchanger (W)
        Q_in = m_dot * self.cp_water * T_in

        # Energy carried out by water leaving to building at tank temperature (W)
        Q_out = m_dot * self.cp_water * self.T_water

        # Total energy balance equations
        dT_water_dt = (Q_in - Q_out - Q_pcm - Q_loss) / (self.M_water * self.cp_water)
        dT_pcm_dt = Q_pcm / (self.pcm.mass * cp_pcm_eff)

        # Update states
        self.T_water += dT_water_dt * dt
        self.pcm.T += dT_pcm_dt * dt

        return T_out


class WaterTank:
    def __init__(self, water_mass_kg=50.0, initial_temperature=50.0):
        self.M_water = water_mass_kg
        self.cp_water = 4184.0   # J/kg·K
        self.T_water = initial_temperature  # Initial water tank temperature (°C)

        # Ambient loss coefficient (W/K) and ambient temperature
        self.UA_loss = 2.0
        self.T_amb = 22.0

    def step(self, m_dot, T_in, dt):
        """
        Advances the state of the tank by one time step.
        """
        T_out = self.T_water
        
        # Heat loss to ambient environment (W)
        Q_loss = self.UA_loss * (self.T_water - self.T_amb)

        # Energy carried in by hot water (W)
        Q_in = m_dot * self.cp_water * T_in

        # Energy carried out by water (W)
        Q_out = m_dot * self.cp_water * self.T_water

        # Total energy balance on the tank water (W)
        dT_water_dt = (Q_in - Q_out - Q_loss) / (self.M_water * self.cp_water)

        # Update states
        self.T_water += dT_water_dt * dt

        return T_out


if __name__ == "__main__":
    # Test simulation with updated parameters (starting at 50 °C to witness the 51-57 °C phase transition)
    pcm_tank = PCMStorageTank(water_mass_kg=500.0, pcm_mass_kg=578.0, initial_temperature=50.0)
    water_tank = WaterTank(water_mass_kg=500.0, initial_temperature=50.0)

    dt = 1.0          # Decreased time step to 1 second for numerical stability in explicit Euler
    hours = 1 
    total_time = hours * 3600
    time_steps = int(total_time / dt)

    m_dot_supply = 0.56    # kg/s
    T_supply = 70.0        # °C
    
    print(f"{'Time (mins)':<12}{'PCM water (°C)':<18}{'PCM material (°C)':<22}{'Water Tank (°C)':<20}")
    print("-" * 72)

    for step in range(time_steps):
        T_out_pcm = pcm_tank.step(m_dot=m_dot_supply, T_in=T_supply, dt=dt)
        T_out_water = water_tank.step(m_dot=m_dot_supply, T_in=T_supply, dt=dt)

        # Print logs every 5 minutes (300 seconds)
        if (step * dt) % 300 == 0:
            mins = int((step * dt) / 60)
            print(f"{mins:<12}{pcm_tank.T_water:<18.2f}{pcm_tank.pcm.T:<22.2f}{water_tank.T_water:<20.2f}")
