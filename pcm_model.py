class K1PCMModel:
    def __init__(self):
        # Properties from Table 1 for K1 PCM
        self.T_m = 25.0         # Baseline melting temperature (°C)
        self.dT = 2.0           # Melting temperature range (°C)
        self.L = 204000.0       # Latent heat (J/kg)
        self.Q_sensible = 26000.0 # Sensible heat inside the range (J/kg)
        
        # Phase change temperature boundaries (half-range of ±1°C)
        self.T_solid = self.T_m - (self.dT / 2.0)   # 24.0°C
        self.T_liquid = self.T_m + (self.dT / 2.0)  # 26.0°C
        
        # Constant sensible specific heat capacity (J/kg·K) outside phase change
        # (Assuming standard baseline cp ~ 2000 J/kg·K for paraffin when fully solid/liquid)
        self.cp_standard = 2000.0 

    def get_enthalpy(self, T):
        """
        Calculates total enthalpy (J/kg) at temperature T (°C)
        using the simplified linear approximation.
        """
        # 1. Fully Solid Phase
        if T < self.T_solid:
            return self.cp_standard * T
            
        # 2. Phase Change Transition Region (Linear Enthalpy Ramp)
        elif self.T_solid <= T <= self.T_liquid:
            h_solid = self.cp_standard * self.T_solid
            fraction = (T - self.T_solid) / self.dT
            # Total energy injection in this zone includes both latent and phase-change sensible heat
            total_transition_energy = self.L + self.Q_sensible
            return h_solid + (fraction * total_transition_energy)
            
        # 3. Fully Liquid Phase
        else:
            h_liquid = (self.cp_standard * self.T_solid) + self.L + self.Q_sensible
            return h_liquid + self.cp_standard * (T - self.T_liquid)

    def get_effective_cp(self, T):
        """
        Returns the effective specific heat capacity (J/kg·K), 
        which peaks inside the melting range due to latent heat absorption.
        """
        if self.T_solid <= T <= self.T_liquid:
            # Distributed energy capacity across the range
            return (self.L + self.Q_sensible) / self.dT
        return self.cp_standard

# Example Usage:
pcm = K1PCMModel()
print(f"Enthalpy at 23°C (Solid): {pcm.get_enthalpy(23.0):,.1f} J/kg")
print(f"Enthalpy at 25°C (Mid-melt): {pcm.get_enthalpy(25.0):,.1f} J/kg")
print(f"Effective Cp during phase change: {pcm.get_effective_cp(25.0):,.1f} J/kg·K")