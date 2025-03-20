import numpy as np
import xarray as xr
import pandas as pd
import time



class Electrolyser:
    def __init__(self, elec_tech):
        
        
        """
        Initialises the Electrolyser Class, which estimates the hydrogen output based on the electrical input from the wind and solar components.
        
        Input: 
        Electrolyser Tech: Alkaline or PEM. Selects which efficiency curve to generate.
        """
        
        # Specify temperature and pressure
        self.temperature = 100
        self.pressure = 30

        
        # Calculate the dynamic efficiency curve
        if elec_tech == "PEM":
            self.efficiency_curve = self.setup_PEM_efficiency(self.temperature, self.pressure)
            self.efficiency = np.nanmax(self.efficiency_curve['Efficiency'])
            self.minimum_power_input = 0.1
        else:
            self.efficiency_curve = self.setup_alkaline_efficiency_curve(self.temperature, self.pressure)
            self.minimum_power_input = 0.4
        self.efficiency = np.nanmax(self.efficiency_curve['Efficiency'])
        

    def calculate_max_yearly(self, capacity):
        max_H2_output = capacity * self.efficiency * 8760
        max_elec_input = capacity * 8760
        return max_elec_input, max_H2_output

   
    def setup_alkaline_efficiency_curve(self, temperature, pressure):
    
        r1 = 4.45153e-5
        r2 = 6.88874e-9
        d1 = -3.12996e-6
        d2 = 4.47137e-7
        s = 0.33824
        t1 = -0.01539
        t2 = 2.00181
        t3 = 15.24178
        Urev = 1.23

        T = temperature
        p = pressure
        i = np.linspace(50, 6000, 200)
        A = 1
        Vcell = Urev + ((r1+d1) + r2*T + d2*p)*i + s*np.log10((t1+t2/T + t3/(T**2))*i + 1)
        I = A * i

        f11 = 478645.74
        f12 = -2953.15
        f21 = 1.03960
        f22 = -0.00104
        f1 = f11 + f12
        f2 = f21 + f22
        Nf = i**2 / (f1 + i**2) * f2
        F = 96500
        Nh = Nf * I / 2 / F
        LHV = 241000
        P_rate = I.max() * Vcell.max()
        P = I * Vcell
        P_perc = P/P_rate
        efficiency = Nh * LHV / P

        P_interp = np.arange(0.2, 1.01, 0.01)
        efficiency_interp = np.interp(P_interp, P_perc, efficiency)
        efficiency_interp = efficiency_interp.round(2)
        
        efficiency_df = pd.DataFrame({'P_Rated (%)': P_interp, 'Efficiency': efficiency_interp}, columns=['P_Rated (%)', 'Efficiency'])
        return efficiency_df
    
    
    def setup_PEM_efficiency(self, temperature, pressure):
    
        # Set up operating conditions
        V_op = 1.8
        T = temperature
        P = pressure
        
        # Calculate ideal voltage
        F = 96485.3365
        delta_G = 285840 - 163.2*(273+T)
        Vi = delta_G / 2 / F

        # Set up constants for the voltage calculation
        Rio = 0.326
        p0 = 1
        To = 80
        k = 0.0395
        dR = -3.812/1000
        Ri = Rio + k * np.log(P / p0) + dR * (T - To)

        # Set up constants for the reversible voltage
        E_rev0 = 1.476
        R = 8.314
        E_r = E_rev0 + R*(273+T)/2/F * np.log(P/p0)
           
        # Set up voltage curve
        I_max = (V_op - E_r)/Ri
        I = np.linspace(0, I_max, 200)
        V = I * Ri + E_r
        power_rated = I*V 
        efficiency_pem = Vi / V
        power_perc = power_rated / power_rated[-1]

        P_interp = np.arange(0.2, 1.01, 0.01)
        efficiency_interp_pem = np.interp(P_interp, power_perc, efficiency_pem)
        efficiency_interp_pem = efficiency_interp_pem.round(2)
        
        efficiency_df = pd.DataFrame({'P_Rated (%)': P_interp, 'Efficiency': efficiency_interp_pem}, columns=['P_Rated (%)', 'Efficiency'])
        return efficiency_df
    
   
    def get_dynamic_efficiency(self, P_load):
    
        
        # Convert to numpy array
        power_values = P_load.values
        latitudes = P_load.latitude.values
        longitudes = P_load.longitude.values
        time_values = P_load.time.values
        
        # Create a storage array for the efficiency
        efficiency_array = np.zeros_like(power_values)
        
        
        # Get power curve
        efficiency_curve = self.efficiency_curve['Efficiency']
        
        # Round input power values
        power_load_rounded = power_values.round(decimals=2)
        
        # Use numpy where to apply 
        possible_values = np.arange(0.2, 1.01, 0.01)
        possible_values = possible_values.round(2)
        for index, value in enumerate(possible_values):
            efficiency_array = np.where(power_load_rounded == value, efficiency_curve[index], efficiency_array)
        
        dynamic_efficiency = xr.DataArray(efficiency_array, dims=('time', 'latitude', 'longitude'),
                                           coords={'time': time_values,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        
        
        return dynamic_efficiency


    def hydrogen_production(self, renewable_profile, capacity, offshore=None):
        """Calculates the hydrogen produced and renewable energy curtailed for the electrolyser
        
        NEED TO UPDATE TO REFLECT ELECTROLYSER RAMPING"""
        hydrogen_LHV = 33.3  # kWh/kg
        electrolyser_capacity = capacity
        
        
        # Calculate electricity available for hydrogen production, including any curtailed electricity and the shortfall
        electricity_H2 = xr.where(renewable_profile > electrolyser_capacity, electrolyser_capacity, renewable_profile)
        electricity_H2 = xr.where(electricity_H2 < self.minimum_power_input * electrolyser_capacity, 0, electricity_H2)
        if offshore is not None:
            electricity_H2 = electricity_H2 - 9.8 - 1.35
        else:
            electricity_H2 = electricity_H2 - 9.8
        electricity_H2 = xr.where(electricity_H2 < 0, 0, electricity_H2)
        curtailed_electricity = xr.where(renewable_profile > electrolyser_capacity, renewable_profile - electrolyser_capacity, 0)
        electrolyser_shortfall = xr.where(renewable_profile < electrolyser_capacity, electrolyser_capacity - renewable_profile, 0)
        
        # Calculate the partial load
        P_load = electricity_H2 / electrolyser_capacity 
        
        # Look up dynamic efficiency
        dynamic_eff = self.get_dynamic_efficiency(P_load)
        
        hydrogen_production = (electricity_H2-9.8) * dynamic_eff / hydrogen_LHV  # In kg, assumes hourly resolution
        max_elec_input, max_H2_output = self.calculate_max_yearly(electrolyser_capacity)
        electrolyser_shortfall = electrolyser_shortfall / max_elec_input
        return hydrogen_production, curtailed_electricity, electrolyser_shortfall


    def calculate_yearly_output(self, renewable_profile, capacity, offshore=None):
        "Calculates the yearly hydrogen production at each location"
        hydrogen_produced, curtailed_electricity, electrolyser_shortfall = self.hydrogen_production(renewable_profile, capacity, offshore=offshore)
        latitudes = renewable_profile.latitude
        longitudes = renewable_profile.longitude
        
        
        hydrogen_produced_array = xr.DataArray(hydrogen_produced, dims=('time', 'latitude', 'longitude'),
                                           coords={'time': renewable_profile.time,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        curtailed_electricity_array = xr.DataArray(curtailed_electricity, dims=('time', 'latitude', 'longitude'),
                                           coords={'time': renewable_profile.time,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        electrolyser_shortfall_array = xr.DataArray(electrolyser_shortfall, dims=('time', 'latitude', 'longitude'),
                                           coords={'time': renewable_profile.time,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        hydrogen_produced_yearly = hydrogen_produced_array.groupby('time.year').sum(dim='time')
        curtailed_electricity_yearly = curtailed_electricity_array.groupby('time.year').sum(dim='time')
        electrolyser_shortfall_yearly = electrolyser_shortfall_array.groupby('time.year').sum(dim='time')
        
        # Accounting for CAPEX only in year 0
        years = hydrogen_produced_yearly.year
        lat_len = len(latitudes)
        lon_len = len(longitudes)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))
        zero_array = np.zeros((1, int(lat_len), int(lon_len)))
        
        zeroth_year_array = xr.DataArray(zero_array, dims=('year', 'latitude', 'longitude'),
                                           coords={'year': new_year,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        

        hydrogen_produced_yearly = xr.concat([zeroth_year_array * 0, hydrogen_produced_yearly], dim = 'year')
        curtailed_electricity_yearly = xr.concat([zeroth_year_array * 0, curtailed_electricity_yearly], dim = 'year')
        electrolyser_shortfall_yearly = xr.concat([zeroth_year_array * 0, electrolyser_shortfall_yearly], dim = 'year')
        
        # Create a dataset with all the arrays
        data_vars = {'hydrogen_produced': hydrogen_produced_yearly,
                     'curtailed_electricity': curtailed_electricity_yearly, 'electrolyser_shortfall': electrolyser_shortfall_yearly}
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        electrolyser_output = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return electrolyser_output
    
    
    
pem_model = Electrolyser("PEM")
alk_model = Electrolyser("ALK")
    
    
    
    
    
    
    
    
