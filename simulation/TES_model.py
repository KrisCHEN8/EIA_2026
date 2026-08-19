import numpy as np


class PCMModel:
    def __init__(self, volume_liters=None, mass_kg=None, initial_temperature=40.0):
        # Specific heat capacity of solid phase (J/kg·K)
        self.cp_standard = 2000.0     

        # Actual phase change temperature boundaries from Rubitherm RT55 datasheet
        self.T_solid = 51.0           # Start of melting area (degC)
        self.T_liquid = 57.0          # End of melting/congealing area (degC)
        self.dT = self.T_liquid - self.T_solid

        # Latent heat storage capacity (J/kg)
        self.latent_heat_energy = 170000.0

        # Effective specific heat during phase change (Latent heat spread over the dT window)
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
        """Returns effective specific heat capacity (J/kg·K) based on temperature"""
        if self.T_solid <= T <= self.T_liquid:
            return self.cp_phase_change
        else:
            return self.cp_standard


class PCMGroup:
    def __init__(self, pcms):
        self.pcms = pcms   # PCM models for each layer

    @property
    def T(self):
        # Return array of temperatures of the PCM layers
        return np.array([pcm.T for pcm in self.pcms])

    @T.setter
    def T(self, val):
        if isinstance(val, (int, float, np.float64)):
            for pcm in self.pcms:
                pcm.T = float(val)
        else:
            for pcm, t_val in zip(self.pcms, val):
                pcm.T = float(t_val)


class PCMStorageTank:
    def __init__(self, water_mass_kg=50.0, pcm_mass_kg=57.8, initial_temperature=50.0, num_layers=3):
        self.M_water = water_mass_kg
        self.cp_water = 4184.0   # J/kg·K
        self.num_layers = num_layers
        
        # Initialize temperatures for each water layer (from bottom=0 to top=num_layers-1)
        self.T_water = np.full(self.num_layers, float(initial_temperature))
        
        # Initialize PCM models for each layer
        self.pcms = [
            PCMModel(mass_kg=pcm_mass_kg / self.num_layers, initial_temperature=initial_temperature)
            for _ in range(self.num_layers)
        ]
        self.pcm = PCMGroup(self.pcms)
        
        # Heat transfer coefficient * surface area coupling water and PCM tube (W/K)
        self.UA_pcm = 2000.0      # W/K
        self.UA_pcm_layer = self.UA_pcm / self.num_layers

        # Ambient loss coefficient (W/K) and ambient temperature
        self.UA_loss = 2.0
        self.T_amb = 22.0

    @property
    def mean_temperature(self):
        return np.mean(self.T_water)

    def step(self, m_dot, T_in, dt):
        """
        Advances the state of the tank by one time step.
        """
        T_out = self.T_water[-1]  # Output to buildings is the top layer
        
        # Mass of water in each layer
        M_layer = self.M_water / self.num_layers
        UA_loss_layer = self.UA_loss / self.num_layers

        dT_water_dt = np.zeros(self.num_layers)
        dT_pcm_dt = np.zeros(self.num_layers)

        for i in range(self.num_layers):
            pcm = self.pcms[i]
            
            # Current temperature-dependent PCM effective heat capacity
            cp_pcm_eff = pcm.get_effective_cp(pcm.T)
            
            # Heat exchange between tank water and PCM in this layer (W)
            Q_pcm = self.UA_pcm_layer * (self.T_water[i] - pcm.T)
            
            # Heat loss to ambient environment (W)
            Q_loss = UA_loss_layer * (self.T_water[i] - self.T_amb)

            # Convective heat transfer from inlet flow
            if i == 0:
                # Lowest layer is the inlet from HEX
                Q_in = m_dot * self.cp_water * T_in
            else:
                Q_in = m_dot * self.cp_water * self.T_water[i-1]
                
            Q_out = m_dot * self.cp_water * self.T_water[i]

            # Energy balance equations
            dT_water_dt[i] = (Q_in - Q_out - Q_pcm - Q_loss) / (M_layer * self.cp_water)
            dT_pcm_dt[i] = Q_pcm / (pcm.mass * cp_pcm_eff)

        # Update states
        self.T_water += dT_water_dt * dt
        for i in range(self.num_layers):
            self.pcms[i].T += dT_pcm_dt[i] * dt

        return T_out


class WaterTank:
    def __init__(self, water_mass_kg=50.0, initial_temperature=50.0, num_layers=3):
        self.M_water = water_mass_kg
        self.cp_water = 4184.0   # J/kg·K
        self.num_layers = num_layers
        
        # Initialize temperatures for each layer
        self.T_water = np.full(self.num_layers, float(initial_temperature))

        # Ambient loss coefficient (W/K) and ambient temperature
        self.UA_loss = 2.0
        self.T_amb = 22.0

    @property
    def mean_temperature(self):
        return np.mean(self.T_water)

    def step(self, m_dot, T_in, dt):
        """
        Advances the state of the tank by one time step.
        """
        T_out = self.T_water[-1]  # Output to buildings is the top layer
        
        # Mass of water in each layer
        M_layer = self.M_water / self.num_layers
        UA_loss_layer = self.UA_loss / self.num_layers

        # Compute dT/dt for each layer
        dT_water_dt = np.zeros(self.num_layers)
        
        for i in range(self.num_layers):
            # Convective heat transfer (flow)
            if i == 0:
                # Lowest layer is the inlet from HEX
                Q_in = m_dot * self.cp_water * T_in
            else:
                Q_in = m_dot * self.cp_water * self.T_water[i-1]
                
            Q_out = m_dot * self.cp_water * self.T_water[i]
            Q_loss = UA_loss_layer * (self.T_water[i] - self.T_amb)
            
            dT_water_dt[i] = (Q_in - Q_out - Q_loss) / (M_layer * self.cp_water)

        # Update states
        self.T_water += dT_water_dt * dt

        return T_out
