

import numpy as np


class EvaporationPondModel:


    def __init__(self):
        # Physical constants
        self.rho = 1000.0  # Water density (kg/m³)
        self.cp_base = 4186.0  # Specific heat capacity of pure water (J/kg·K)
        self.Lv = 2.45e6  # Latent heat of vaporization (J/kg)

        # Pond geometry
        self.A = 100.0  # Surface area (m²)

        # Environmental parameters
        self.T_ambient = 20.0  # Ambient temperature (°C)
        self.RH = 0.5  # Relative humidity (0-1)
        self.h_conv = 10.0  # Convective heat transfer coefficient (W/m²·K)
        self.alpha = 0.9  # Solar absorption coefficient

        # Evaporation coefficient (empirical)
        self.k_evap = 2.5e-8  # m/s·Pa

    def solar_irradiance(self, t):

        # Simple day-night cycle (peak at noon)
        hour_of_day = (t % 24)
        if 6 <= hour_of_day <= 18:
            # Sinusoidal during day
            return 800.0 * np.sin(np.pi * (hour_of_day - 6) / 12)
        else:
            return 0.0

    def saturation_vapor_pressure(self, T):

        return 611.2 * np.exp(17.67 * T / (T + 243.5))

    def specific_heat(self, S):

        # Empirical relationship: cp decreases with salinity
        return self.cp_base * (1 - 0.0007 * S)

    def evaporation_rate(self, T, S, V):
        """
        Calculate evaporation rate
        T: temperature (°C)
        S: salinity (g/L)
        V: volume (m³)
        Returns: m³/s
        """
        # Saturation vapor pressure at water temperature
        e_s = self.saturation_vapor_pressure(T)

        # Actual vapor pressure in air
        e_a = self.RH * self.saturation_vapor_pressure(self.T_ambient)

        # Salinity effect (higher salinity reduces evaporation)
        salinity_factor = 1.0 - 0.001 * S

        # Evaporation rate (m/s)
        E_rate = self.k_evap * (e_s - e_a) * salinity_factor

        # Volume rate (m³/s)
        E_volume = E_rate * self.A

        return max(E_volume, 0.0)  # Can't be negative

    def ode_system(self, t, y):
        """
        ODE system: dy/dt = f(t, y)
        y = [T, S, V] where:
            T: temperature (°C)
            S: salinity (g/L)
            V: volume (m³)
        Returns: [dT/dt, dS/dt, dV/dt]
        """
        T, S, V = y

        # Prevent volume from going negative
        if V < 0.1:
            return np.array([0.0, 0.0, 0.0])

        # Evaporation rate
        E = self.evaporation_rate(T, S, V)

        # Heat fluxes
        Q_solar = self.solar_irradiance(t) * self.A * self.alpha
        Q_evap = E * self.rho * self.Lv
        Q_loss = self.h_conv * self.A * (T - self.T_ambient)

        # Specific heat (depends on salinity)
        cp = self.specific_heat(S)

        # Mass of water
        m = self.rho * V

        # ODEs
        dT_dt = (Q_solar - Q_evap - Q_loss) / (m * cp)
        dS_dt = S * E / V if V > 0.1 else 0.0
        dV_dt = -E

        return np.array([dT_dt, dS_dt, dV_dt])

    def get_initial_conditions(self):
        """Return initial conditions [T0, S0, V0]"""
        T0 = 25.0  # Initial temperature (°C)
        S0 = 35.0  # Initial salinity (g/L) - seawater
        V0 = 100.0  # Initial volume (m³) - 1m depth
        return np.array([T0, S0, V0])