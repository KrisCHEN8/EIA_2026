class PCMModel:
    def __init__(self, volume_liters=None, mass_kg=None):
        # Physical Properties from Rubitherm RT65
        self.cp_standard = 2000.0  # 2 kJ/kg·K baseline specific heat
        self.T_solid = 58.0        # Start of melting zone
        self.T_liquid = 65.0       # End of melting zone
        self.dT = self.T_liquid - self.T_solid

        # Combined Latent + Sensible heat capacity inside the broader window
        self.total_transition_energy = 150000.0 
        
        # Determine Mass (M_pcm)
        if mass_kg:
            self.mass = mass_kg
        elif volume_liters:
            self.mass = volume_liters * 0.83 
        else:
            self.mass = 60.2   # kg, from experimental data

    def get_enthalpy(self, T):
        """
        Calculates total enthalpy (J/kg) at temperature T (°C)
        using a continuous simplified linear approximation.
        """
        # 1. Fully Solid Phase
        if T < self.T_solid:
            return self.cp_standard * T

        # 2. Phase Change Transition Region (Linear Enthalpy Ramp)
        elif self.T_solid <= T <= self.T_liquid:
            h_solid = self.cp_standard * self.T_solid
            fraction = (T - self.T_solid) / self.dT
            return h_solid + (fraction * self.total_transition_energy)

        # 3. Fully Liquid Phase
        else:
            h_liquid = (self.cp_standard * self.T_solid) + self.total_transition_energy
            return h_liquid + self.cp_standard * (T - self.T_liquid)

    def get_effective_cp(self, T_pcm):
        """
        Returns dynamic specific heat capacity based on RT65 data.
        """
        if self.T_solid <= T_pcm <= self.T_liquid:
            return self.total_transition_energy / self.dT
        return self.cp_standard

if __name__ == "__main__":
    pcm = PCMModel()
    print(f"Enthalpy at 55°C (Solid): {pcm.get_enthalpy(55.0):,.1f} J/kg")
    print(f"Enthalpy at 63°C (Mid-melt): {pcm.get_enthalpy(63.0):,.1f} J/kg")
    print(f"Effective Cp during phase change: {pcm.get_effective_cp(63.0):,.1f} J/kg·K")
