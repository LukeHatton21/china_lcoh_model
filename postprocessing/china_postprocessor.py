import numpy as np
import xarray as xr
import time
import csv
import math
import cartopy.crs as ccrs
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.colors as colors
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
import pandas as pd
import warnings
from geodata_v3 import Global_Data

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

class ResultsProcessor:
    def __init__(self, pem_folder, alk_folder, output_folder, data_path, combined_country_grids, country_index, land_cover, land_mapping, colormap, colors, country_codes, capex_mapping):
        """ Class to process results from the Hydrogen Model
        
        INPUTS
        Results folder: Folder in which the collated results are stored"""    
        
        
        # Read in the files
        self.pem_folder = pem_folder
        self.alk_folder = alk_folder
        self.output_folder = output_folder
        self.raw_country_data = xr.open_dataset(combined_country_grids)
        
        # Establish constants
        self.lifetime = 20
        
        # Collate all costs
        self.read_in_hybrid_results("PEM", 2000)
        self.read_in_hybrid_results("ALK", 1700)
        
        # Set PEM and Alkaline Costs
        self.PEM_cost = 2000
        self.ALK_cost = 1700
        
        # Get country data
        self.country_wacc_mapping = pd.read_csv((data_path + "new_country_waccs.csv"))
        self.country_data = xr.open_dataset((data_path + "country_grids.nc"))
        
        # Get data required to calculate the global supply

        self.country_grid = self.get_countries_in_required_resolution(self.raw_country_data, self.PEM_0['levelised_cost'])
        self.country_index = pd.read_csv(country_index)
        self.land_cover = xr.open_dataset(land_cover)
        self.land_mapping = pd.read_csv(land_mapping)
        self.color_df = pd.read_csv(colormap)
        self.country_codes = pd.read_csv(country_codes)
        self.capex_mapping = pd.read_csv(capex_mapping)
        
        # Read colours
        self.solar = colors[0]
        self.onshore = colors[1]
        self.offshore = colors[2]
        self.offshore_elec = colors[3]
        
    def get_countries_in_required_resolution(self, country_data, target_data):

        # Reindex using the xarray reindex function
        new_coords = {'latitude': target_data.latitude, 'longitude': target_data.longitude}
        countries_resampled = country_data.reindex(new_coords, method='nearest')

        
        # Create new dataset with countries
        data_vars = {'country': countries_resampled['country']}
        coords = {'latitude': target_data.latitude,
                  'longitude': target_data.longitude}
        countries_output = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return countries_output    
        
        
    def read_in_hybrid_results(self, technology, cost):
        
        # Get correct folder
        if technology == "PEM":
            results_folder = self.pem_folder
        else:
            results_folder = self.alk_folder
        
        
        # Get wind and solar results
        setattr(self, technology + "_wind_results", xr.open_dataset(results_folder + technology + "_" + str(cost) + "USD_0%.nc"))
        setattr(self, technology + "_solar_results", xr.open_dataset(results_folder + "SOLAR_" + technology + "_" + str(cost) + "USD.nc"))
        
        # Read in hybrid files and set up empty combined dataset
        hybrid_files = glob.glob(results_folder + "*" + technology + "*.nc")
        combined_ds = None
        
        # Sort the files based on the solar fraction
        hybrid_files = sorted(hybrid_files)
        
        
        for i, file in enumerate(hybrid_files):
            

            # Get the variable name and dataset
            variable_name = technology + f"_{(i)*10}"
            ds = xr.open_dataset(file)
            
            if i == 0:
                country_grid = self.get_countries_in_required_resolution(self.raw_country_data, ds)
            
            # Set Greenland to zero
            ds['levelised_cost'] = xr.where(country_grid['country']==83, np.nan, ds['levelised_cost'])
            ds['hydrogen_production'] = xr.where(country_grid['country']==83, np.nan, ds['hydrogen_production'])
            
            # Expand the dimensions
            ds = ds.expand_dims({"solar_fraction":[i*10]})
            
            # Combine the dataset
            if combined_ds is None:
                combined_ds = ds
            else:
                combined_ds = combined_ds.combine_first(ds)
                
            ds = ds.sel(solar_fraction = i*10)
            ds = ds.drop_vars('solar_fraction')
            # Save the dataset onto the ResultsProcessor object
            setattr(self, variable_name, ds)
        
        # Save the combined results onto the ResultsProcessor object
        setattr(self,technology + "_results", combined_ds)
        
            
    def get_cheapest_lcoh(self, technology, specified_results=None):
        
        # Extract collated results
        if technology == "PEM":
            collated_results = self.PEM_results
        else:
            collated_results = self.ALK_results
        if specified_results is not None:
            collated_results = specified_results
            
        # Extract the LCOH for the specified technology
        if specified_results is None:
            lcoh = collated_results['levelised_cost']
        else:
            lcoh = collated_results['Regional_LCOH']
        
        # Calculate the cheapest lcoh across the dimensions
        min_lcoh = lcoh.min(dim="solar_fraction")
        
        # Get the minimum index
        min_index = lcoh.idxmin(dim="solar_fraction")
        
        # Extract the cheapest solar fraction at each location
        indexed_results = collated_results.sel(solar_fraction=min_index, method='nearest')
        
        # Add solar fraction to indexed_results
        indexed_results['Optimal_SF'] = min_index
        # Add to the ResultsProcessor class
        if specified_results is None:
            setattr(self,technology + "_cheapest_sf", indexed_results)
        
        return indexed_results
    
    
    def calculate_uniform_results(self, data, wacc):
    
        # Extract key figures from the data
        latitudes = data.latitude.values
        longitudes = data.longitude.values
        electrolyser_cost = data['electrolyser_costs'].values
        wind_costs = data['wind_costs'].values
        solar_costs = data['solar_costs'].values

        # Calculate old and new discount factors
        discount_factor = self.calculate_discount_factor(wacc)

        # Extract annual costs and electricity production
        annual_hydrogen_production = data['hydrogen_production'].values * 1000
        annual_costs = 0.03 * electrolyser_cost + 0.023 * solar_costs + 0.026 * wind_costs 
        capital_costs = data['total_capital_costs'].values

        # Calculate new LCOE based on discount factor
        new_lcoh = xr.where(np.isnan(annual_hydrogen_production) == True, np.nan, (capital_costs + discount_factor * annual_costs ) / (discount_factor * annual_hydrogen_production))

        # Rename 
        data = data.rename({"levelised_cost":"Initial_LCOH", "levelised_cost_ren":"Initial_LCOH_REN", "levelised_cost_elec": "Initial_LCOH_ELEC"})
        data['Uniform_LCOH'] = xr.DataArray(new_lcoh, dims={"latitude":latitudes, "longitude":longitudes})

        return data
    
    
    def calculate_discount_factor(self, discount_rate):
        
        discount_factor = 0
        for i in np.arange(1, self.lifetime + 1, 1):
            discount_factor = discount_factor + 1 / ((1 + discount_rate) ** i)
            
        return discount_factor
    
    def calculate_new_lcoh(self, data, technology, selected_countries, electrolyser_cost=None):
        
        def get_index_number(selected_countries):
            
            index_numbers = []
            for i in selected_countries:
                country_index = self.country_index[self.country_index['country'].isin(selected_countries)]['index'].values[0]
                index_numbers = index_numbers.append(country_index)
        
            return index_numbers
        
        if not isinstance(selected_countries[0], int):
            selected_countries = get_index_number(selected_countries)
        
        # Read in country waccs
        country_waccs = self.country_wacc_mapping
        land_mapping = self.country_grid['country']

        # Expand dimensions to the new axis
        land_mapping = land_mapping.expand_dims(dim={"solar_fraction": np.arange(0, 110, 10)})  

        # Read in initial LCOH
        initial_lcoh = data['levelised_cost']

        # Read in parameters from the data
        hydrogen_production = data['hydrogen_production'] * 1000
        solar_costs = data['solar_costs']
        wind_costs = data['wind_costs']
        electrolyser_costs = data['electrolyser_costs']
        total_capital_costs = data['total_capital_costs']

        # Calculate renewables annual costs
        wind_annual_costs = xr.where(np.isnan(solar_costs)==True, wind_costs* 0.03, wind_costs * 0.026)
        solar_annual_costs = solar_costs * 0.023
        electrolyser_annual_costs = electrolyser_costs * 0.03
        annual_costs = wind_annual_costs + solar_annual_costs + electrolyser_annual_costs

        # Get CAPEX correction factor
        wind_capex_costs = self.capex_mapping[['index', 'Wind_Percentage']].rename(columns={"index":"country"}) 
        solar_capex_costs = self.capex_mapping[['index', 'Solar_Percentage']].rename(columns={"index":"country"}) 
        if technology == "PEM":
            elec_capex_costs = self.capex_mapping[['index', 'PEM_Percentage']].rename(columns={"index":"country", "PEM_Percentage":"Elec_Percentage"})
        else:
            elec_capex_costs = self.capex_mapping[['index', 'ALK_Percentage']].rename(columns={"index":"country", "ALK_Percentage":"Elec_Percentage"})

        # Set countries to loop over
        set_countries = np.arange(1, 251, 1)

        data['Regional_LCOH'] = xr.full_like(data['levelised_cost'], np.nan)
        data['Regional_CAPEX'] = xr.full_like(data['levelised_cost'], np.nan)
        data['Regional_OPEX'] = xr.full_like(data['levelised_cost'], np.nan)
        data['Regional_WACC'] = xr.full_like(data['levelised_cost'], np.nan)
        data['adjusted_wind_costs'] = xr.full_like(data['levelised_cost'], np.nan)  
        data['adjusted_solar_costs'] = xr.full_like(data['levelised_cost'], np.nan)  
        data['adjusted_elec_costs'] = xr.full_like(data['levelised_cost'], np.nan)  

        if selected_countries is not None:
            countries_set = selected_countries
        else:
            countries_set = np.arange(1, 251, 1)

        # Loop over every country
        for index in countries_set:   

            # Get solar and wind CAPEX correction
            solar_capex_correction = float(solar_capex_costs[solar_capex_costs['country']==index]['Solar_Percentage'].values)
            wind_capex_correction = float(wind_capex_costs[wind_capex_costs['country']==index]['Wind_Percentage'].values)
            elec_capex_correction = float(elec_capex_costs[elec_capex_costs['country']==index]['Elec_Percentage'].values)
            
            if electrolyser_cost is not None:
                if technology == "PEM":
                    elec_capex_correction = electrolyser_cost / self.PEM_cost
                else:
                    elec_capex_correction = electrolyser_cost / self.ALK_cost

            # Get WACC value
            row = country_waccs.loc[country_waccs['index'] == index]
            wind_discount_rate = row['onshore wacc'].values
            solar_discount_rate = row['solar pv wacc'].values

            # Calculate new wind costs
            adjusted_wind_costs = xr.where(land_mapping == index, wind_costs * wind_capex_correction, wind_costs)
            data['adjusted_wind_costs'] = xr.where(land_mapping == index, wind_costs * wind_capex_correction, data['adjusted_wind_costs'])

            # Calculate new solar costs
            adjusted_solar_costs = xr.where(land_mapping == index, solar_costs * solar_capex_correction, solar_costs)
            data['adjusted_solar_costs'] = xr.where(land_mapping == index, wind_costs * wind_capex_correction, data['adjusted_solar_costs'])

            # Calculate new electrolyser costs
            adjusted_elec_costs = xr.where(land_mapping == index, electrolyser_costs * elec_capex_correction, electrolyser_costs)
            data['adjusted_elec_costs'] = xr.where(land_mapping == index, wind_costs * wind_capex_correction, data['adjusted_elec_costs'])

            # Calculate new operating costs
            adjusted_wind_opex = xr.where(np.isnan(solar_costs)==True, np.nan, adjusted_wind_costs * 0.026)
            adjusted_solar_opex = adjusted_solar_costs * 0.023
            adjusted_elec_opex = adjusted_elec_costs * 0.03

            # Sum total operating and capital costs
            adjusted_capital_costs = xr.where(land_mapping == index, total_capital_costs - solar_costs - wind_costs - electrolyser_costs + adjusted_solar_costs + adjusted_wind_costs + adjusted_elec_costs, total_capital_costs)
            adjusted_operating_costs = xr.where(land_mapping == index, adjusted_wind_opex + adjusted_solar_opex + adjusted_elec_opex, annual_costs)


            # Calculate new discount factor
            weighted_discount_rate = (wind_discount_rate * wind_costs + solar_discount_rate * solar_costs) / (wind_costs + solar_costs)
            weighted_discount_rate = xr.where((np.isnan(weighted_discount_rate) == True), 0.097, weighted_discount_rate)
            discount_factor = self.calculate_discount_factor(weighted_discount_rate)

            # Get hydrogen production
            specified_hydrogen_production = xr.where(land_mapping == index, hydrogen_production, np.nan)

            # Calculate new LCOH, for onshore locations
            data['Regional_LCOH'] = xr.where(land_mapping == index, (adjusted_capital_costs + (adjusted_operating_costs * discount_factor)) / (discount_factor * hydrogen_production), data['Regional_LCOH'])
            data['Regional_CAPEX'] = xr.where(land_mapping == index, adjusted_capital_costs, data['Regional_CAPEX'])
            data['Regional_OPEX'] = xr.where(land_mapping == index, adjusted_operating_costs, data['Regional_OPEX'])
            data['Regional_WACC'] = xr.where(land_mapping == index, weighted_discount_rate, data['Regional_WACC'])

        # Extract only values for the selected countries, if specified
        if selected_countries is not None:
            data['Regional_LCOH'] = xr.where(np.isin(land_mapping, selected_countries), data['Regional_LCOH'], np.nan)
        calculated_data = data.drop_vars(names=["levelised_cost", "levelised_cost_ren", "levelised_cost_elec", "total_capital_costs","solar_costs", "wind_costs"], errors="ignore")  
        calculated_data = calculated_data.drop_vars(["dims"], errors="ignore")

        return calculated_data
    
    
    
    def calculate_required_capex(self, data, lcoh, elec_capex, technology):

        # Establish the dataset values
        lcoh_initial = data['levelised_cost']
        lcoh_elec = data['levelised_cost_elec']

        # Perform the calculation
        capex_required = xr.where(np.isnan(lcoh_initial) == True, np.nan, (1 - (lcoh_initial-lcoh)/lcoh_elec) * elec_capex)
        capex_required = xr.where(capex_required > 2000, np.nan, capex_required)
        capex_required = xr.where(capex_required < 200, 200, capex_required)

        # Add the required CAPEX to the dataset as an additional variable
        data['required_capex_' + str(float(lcoh))] = capex_required

        # Return the dataset

        return data
    
    def calculate_required_waccs(self, data, lcoh, technology):


        # Extract key figures from the data
        latitudes = data.latitude.values
        longitudes = data.longitude.values

        # Store costs
        electrolyser_cost = data['electrolyser_costs'].values
        solar_costs = data['solar_costs'].values
        wind_costs = data['wind_costs'].values

        # Get other parameters
        annual_hydrogen_production = data['hydrogen_production'] * 1000
        initial_lcoh = data['levelised_cost']

        # Calculate reductions to 2030 
        electrolyser_reduction = electrolyser_cost * 0.6
        solar_reduction = solar_costs * 0.385
        wind_reduction = xr.where(np.isnan(solar_costs)==True, data['wind_costs'] * 0.337, data['wind_costs']* 0.054)

        # Calculate wind annual costs
        wind_annual_costs = xr.where(np.isnan(solar_costs)==True, (data['wind_costs'] - wind_reduction)* 0.03, (data['wind_costs'] - wind_reduction) * 0.026)


        # Calculate annual costs
        annual_costs = 0.03 * (electrolyser_cost - electrolyser_reduction) + 0.023 * (solar_costs - solar_reduction)  + wind_annual_costs
        capital_costs = data['total_capital_costs'] - electrolyser_reduction - solar_reduction  - wind_reduction

        # Calculate discount factor at each location
        discount_factor = capital_costs / ( (annual_hydrogen_production * lcoh ) - annual_costs)

        # Create array of discount factor to WACC values and round discount factor
        discount_rates = np.linspace(0, 0.15, 1001)
        discount_factors_array = self.calculate_discount_factor(discount_rates)
        xdata = discount_rates
        ydata = discount_factors_array

        # Calculate curve fit
        ylog_data = np.log(ydata)
        curve_fit = np.polyfit(xdata, ylog_data, 2)
        y = np.exp(curve_fit[2]) * np.exp(curve_fit[1]*xdata) * np.exp(curve_fit[0]*xdata**2)


        # Create interpolator
        interpolator = interp1d(ydata, xdata, kind='nearest', bounds_error=False, fill_value=(np.nan, 9.99))

        # Use rounded discount factors to calculate WACC values
        estimated_waccs = interpolator(discount_factor)*100
        estimated_waccs = xr.where(discount_factor < 0, 999, estimated_waccs)
        wacc_da = xr.DataArray(estimated_waccs, coords={"latitude": latitudes, "longitude":longitudes})
        wacc_da =  xr.where(np.isnan(initial_lcoh)==True, np.nan, wacc_da)
        data['Required_WACC'] = wacc_da


        return data
    
    
    
    def calculate_total_capex_reduction(self, data, lcoh):

        # Establish the dataset values
        lcoh_initial = data['levelised_cost']
        lcoh_elec = data['levelised_cost_elec']
        lcoh_ren = data['levelised_cost_ren']

        # Perform the calculation
        capex_reduction = xr.where(np.isnan(lcoh_initial) == True, np.nan, ((lcoh_initial - lcoh)/(lcoh_elec + lcoh_ren))*100)
        capex_reduction = xr.where(capex_reduction > 100, 100, capex_reduction)
        capex_reduction = xr.where(capex_reduction < 0, 999, capex_reduction)

        # Add the required CAPEX to the dataset as an additional variable
        data['capex_reduction_' + str(int(lcoh))] = capex_reduction

        # Return the dataset

        return data
        
        
        
    def plot_data(self, values, latitudes, longitudes, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend=None):
        
        # create the heatmap using pcolormesh
        fig = plt.figure(figsize=(30, 15), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.LogNorm(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
        cb = fig.colorbar(heatmap, ax=ax, shrink=0.5, ticks=tick_values, format="%0.0f", extend=extend, anchor=(0, 0.32))


        cb.ax.tick_params(labelsize=20)
        if title is not None:
            cb.ax.set_title(title, fontsize=25)

        # set the extent and aspect ratio of the plot
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        ax.set_aspect(1)

        # add axis labels and a title
        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
        ax.add_feature(borders, edgecolor='gray', linestyle=':')
        ax.coastlines()
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        ax.coastlines()

        plt.show()
        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")
            
            
            
    def produce_G7_supply_curves(self, upper_bound, noplot=None):
        
        def cutoff_value(lcoh, production, upper_bound, solar_fraction=None):
            bounded_production = xr.where(lcoh > upper_bound, np.nan, production)
            bounded_costs = xr.where(lcoh > upper_bound, np.nan, lcoh)
            
            if solar_fraction is not None:
                bounded_solar_fraction = xr.where(lcoh > upper_bound, np.nan, solar_fraction/100)
                
                return bounded_production, bounded_costs, bounded_solar_fraction
            
            return bounded_production, bounded_costs

        
        # Apply the same cutoff for alkaline and PEM
        solar_alkaline_prod, solar_alkaline_costs = cutoff_value(self.ALK_100['levelised_cost'], self.ALK_100['hydrogen_production'], upper_bound)
        wind_alkaline_prod, wind_alkaline_costs = cutoff_value(self.ALK_0['levelised_cost'], self.ALK_0['hydrogen_production'], upper_bound)
        solar_pem_prod, solar_pem_costs = cutoff_value(self.PEM_100['levelised_cost'], self.PEM_100['hydrogen_production'], upper_bound)
        wind_pem_prod, wind_pem_costs = cutoff_value(self.PEM_0['levelised_cost'], self.PEM_0['hydrogen_production'], upper_bound)
        cheapest_pem_prod, cheapest_pem_costs, cheapest_pem_sf = cutoff_value(self.PEM_cheapest_sf['levelised_cost'].drop_vars('solar_fraction'), self.PEM_cheapest_sf['hydrogen_production'].drop_vars('solar_fraction'), upper_bound, self.PEM_cheapest_sf['Optimal_SF'].drop_vars('solar_fraction'))
        self.bounded_PEM = cheapest_pem_prod
        self.bounded_PEM_cost = cheapest_pem_costs
        self.bounded_PEM_SF = cheapest_pem_sf
        cheapest_alk_prod, cheapest_alk_costs, cheapest_alk_sf = cutoff_value(self.ALK_cheapest_sf['levelised_cost'].drop_vars('solar_fraction'), self.ALK_cheapest_sf['hydrogen_production'].drop_vars('solar_fraction'), upper_bound, self.ALK_cheapest_sf['Optimal_SF'].drop_vars('solar_fraction'))
        self.bounded_ALK = cheapest_alk_prod
        self.bounded_ALK_cost = cheapest_alk_costs
        self.bounded_ALK_SF = cheapest_alk_sf
        
        # PEM
        onshore_pem_costs = xr.where(np.isnan(solar_pem_costs ), np.nan, wind_pem_costs)
        onshore_pem_production = xr.where(np.isnan(solar_pem_costs ), np.nan, wind_pem_prod)
        offshore_pem_costs = xr.where(np.isnan(solar_pem_costs ), wind_pem_costs, np.nan)
        offshore_pem_production = xr.where(np.isnan(solar_pem_costs ),wind_pem_prod, np.nan)
        
        # Alkaline
        onshore_alk_costs = xr.where(np.isnan(solar_pem_costs ), np.nan, wind_alkaline_costs)
        onshore_alk_production = xr.where(np.isnan(solar_pem_costs ), np.nan, wind_alkaline_prod)
        offshore_alk_costs = xr.where(np.isnan(solar_pem_costs ), wind_alkaline_costs, np.nan)
        offshore_alk_production = xr.where(np.isnan(solar_pem_costs ), wind_alkaline_prod, np.nan)
        
        
        
        # Get Supply Curves
        solar_alkaline_ds = self.get_supply_curves(solar_alkaline_costs,  solar_alkaline_prod, "Solar PV")
        wind_alkaline_ds = self.get_supply_curves(wind_alkaline_costs,  wind_alkaline_prod, "Wind")
        optimal_alkaline_ds = self.get_supply_curves(cheapest_alk_costs, cheapest_alk_prod, "Hybrid", solar_fractions=cheapest_alk_sf)
        solar_pem_ds = self.get_supply_curves(solar_pem_costs,  solar_pem_prod, "Solar PV")
        wind_pem_ds = self.get_supply_curves(wind_pem_costs, wind_pem_prod, "Wind")
        optimal_pem_ds = self.get_supply_curves(cheapest_pem_costs, cheapest_pem_prod, "Hybrid", solar_fractions=cheapest_pem_sf)
        
        # Store relevant supply curve
        self.solar_pem_ds = solar_pem_ds
        self.wind_pem_ds = wind_pem_ds
        self.optimal_pem_ds = optimal_pem_ds
        
        if noplot is not None:
            return
        
        # Produce supply curves
        self.plot_supply_curve(solar_pem_ds, "Solar + PEM", "a")
        self.plot_supply_curve(wind_pem_ds, "Wind + PEM", "b")
        self.plot_supply_curve(optimal_pem_ds, "Hybrid + PEM", "c")
        self.plot_supply_curve(solar_alkaline_ds, "Solar + Alkaline", "a")
        self.plot_supply_curve(wind_alkaline_ds, "Wind + Alkaline", "b")
        self.plot_supply_curve(optimal_alkaline_ds, "Hybrid + Alkaline", "c")
        
        
    def get_utilisations(self, annual_production, technology, solar_fractions=None):
        
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values
        global_cover = self.land_cover.reindex_like(annual_production, method="nearest")
        mapping = self.land_mapping
        
        utilisation = xr.zeros_like(global_cover['cover'])
        for i in np.arange(0, 21, 1):
            # Use xarray's where and isin functions to map land use categories to values
            if technology == "Solar PV":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
            elif technology =="Wind":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)
                utilisation = xr.where(np.isnan(self.PEM_100['levelised_cost'].sel(latitude=latitudes, longitude=longitudes)) & ~np.isnan(self.PEM_0['levelised_cost'].sel(latitude=latitudes, longitude=longitudes)), 1, utilisation)      
            elif technology =="Hybrid":
                wind_utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)
                wind_utilisation = xr.where(np.isnan(self.PEM_100['levelised_cost'].sel(latitude=latitudes, longitude=longitudes)) & ~np.isnan(self.PEM_0['levelised_cost'].sel(latitude=latitudes, longitude=longitudes)), 1, wind_utilisation)
                solar_utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
                utilisation = solar_fractions * solar_utilisation + wind_utilisation * (1 - solar_fractions)
                
                
        return utilisation    
        
    def get_supply_curves(self, levelised_costs, annual_production, technology, plot_global=None, solar_fractions=None):
    
        
        # Calculate area of each grid point in kms 
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values
        grid_areas = self.get_areas(annual_production)
        utilisation_factors = self.get_utilisations(annual_production, technology, solar_fractions)
        
        # Set out constants
        if technology == "Wind":
            power_density = 6520 # kW/km2
        elif technology == "Solar PV":
            power_density = 32950  # kW/km2
        elif technology == "Hybrid":
            power_density = 6520 + solar_fractions * (32950 - 6520)
        installed_capacity = 1000
        
        # Scale annual hydrogen production by turbine density
        max_installed_capacity = power_density * grid_areas['area'] * utilisation_factors
        ratios = max_installed_capacity / installed_capacity
        technical_hydrogen_potential = annual_production * ratios
        
        # Create new dataset with cost and production volume
        if solar_fractions is not None:
            data_vars = {'Optimal_SF': solar_fractions, 'hydrogen_technical_potential': technical_hydrogen_potential,
                     'LCOH': levelised_costs, 'country': self.country_grid['country']}
        else:
            data_vars = {'hydrogen_technical_potential': technical_hydrogen_potential,
                     'LCOH': levelised_costs, 'country': self.country_grid['country']}
        coords = {'latitude': latitudes,
                  'longitude': longitudes}
        supply_curve_ds = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return supply_curve_ds
    
    


    def plot_global_supply_curves(self, supply_curve_ds, technology=None, graphmarking=None, filename=None):


        def clean_results(dataframe, cutoff):
            # Sort by levelised cost
            sorted_supply = dataframe.sort_values(by=['levelised_cost'])

            # Remove rows that are empty
            cleaned_df = sorted_supply.dropna(axis='index')
            final_df = cleaned_df.copy()

            # Apply a threshold for cost (if applicable)
            index_names = final_df[ final_df['levelised_cost'] >= cutoff ].index
            final_df.drop(index_names, inplace = True)

            # Apply a threshold for locations with zero utilisation (if applicable)
            util_index_names = final_df[final_df['hydrogen_technical_potential'] == 0 ].index
            final_df.drop(util_index_names, inplace = True)

            return final_df


        # Select relevant dataset and convert to a dataframe
        technology_df = supply_curve_ds.to_dataframe()

        # Select grid cells corresponding to countries
        combined_df = pd.merge(self.country_codes, technology_df, on='country', how='outer')

        # Plot the results
        cleaned_results_df = clean_results(combined_df , 10)
        sorted_lc = cleaned_results_df.sort_values(by=['levelised_cost'])
        sorted_lc.loc[:, 'total_cumulative_potential'] = sorted_lc['hydrogen_technical_potential'].cumsum()

        # Plot the results, highlighting key countries          
        rounded_df = sorted_lc.round({'levelised_cost': 3})
        fig, ax = plt.subplots(figsize=(20, 8))
        color_labels = {}
        cmap = mpl.colormaps['RdBu_r']
        norm = mpl.colors.Normalize(vmin=0, vmax=100)  # Normalize to the range of solar fractions

        # Iterate through each data point and create a bar with the specified width
        for index, row in rounded_df.iterrows():
            width = row['hydrogen_technical_potential'] / 1e+06  # Bar width
            height = row['levelised_cost']  # Bar height
            solar_fraction = row['Optimal_SF']
            solar_frac_label = str(int(solar_fraction))
            cumulative_production = row['total_cumulative_potential'] / 1e+06  # Cumulative production
            color = cmap(solar_fraction)

            # Create a dummy bar element with the color and label
            dummy_bar = plt.bar([], [], color=color, label=solar_frac_label)

            # Add the color and label to the dictionary
            color_labels[color] = solar_frac_label

            # Plot a bar with the specified width, height, x-position, and color
            ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)

        # Set labels
        ax.set_xlim(0, cumulative_production)
        ax.set_ylabel('LCOH(US$/kg)', fontsize=25)
        ax.set_xlabel('Annual Hydrogen Production (Mt/yr)', fontsize=25)
        ax.set_ylim([0, 10])
        ax.plot(rounded_df['total_cumulative_potential'], rounded_df['levelised_cost'], linewidth=10, color='black')

        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed

        # Create a legend based on the color_labels dictionary
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{color_labels[color]}',
                                  markerfacecolor=color, markersize=20) for color in color_labels]

        #plt.legend(handles=legend_elements, title='Solar Fraction', loc='upper left', fontsize=15, title_fontsize=20, ncol=4)

        # Add color bar
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Solar Fraction (%)', fontsize=20)
        cbar.ax.tick_params(labelsize=15)

        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
        plt.savefig(filename + ".png")
        plt.show()
    
    
    def get_max_electricity_price(self, data, efficiency):

        lcoh = data['levelised_cost']

        # Get electricity price
        data['electricity_price_max'] = lcoh / 33.33 * 1 / efficiency

        return data
    
    def get_max_ci(self, data, efficiency, benchmark):

        ci = data['carbon_intensity']

        # Get electricity price
        ci_electricity = (benchmark - ci) / 33.33 * 1 / efficiency
        data['electricity_ci_max'] = xr.where(ci_electricity < 0, 999, ci_electricity)


        return data

    
    def get_areas(self, annual_production):
        
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values

        # Add an extra value to latitude and longitude coordinates
        latitudes_extended = np.append(latitudes, latitudes[-1] + np.diff(latitudes)[-1])
        longitudes_extended = np.append(longitudes, longitudes[-1] + np.diff(longitudes)[-1])

        # Calculate the differences between consecutive latitude and longitude points
        dlat_extended = np.diff(latitudes_extended)
        dlon_extended = np.diff(longitudes_extended)
        
        # Calculate the Earth's radius in kms
        radius = 6371

        # Compute the mean latitude value for each grid cell
        mean_latitudes_extended = (latitudes_extended[:-1] + latitudes_extended[1:]) / 2
        mean_latitudes_2d = mean_latitudes_extended[:, np.newaxis]

        # Convert the latitude differences and longitude differences from degrees to radians
        dlat_rad_extended = np.radians(dlat_extended)
        dlon_rad_extended = np.radians(dlon_extended)

        # Compute the area of each grid cell using the Haversine formula
        areas_extended = np.outer(dlat_rad_extended, dlon_rad_extended) * (radius ** 2) * np.cos(np.radians(mean_latitudes_2d))

        # Create a dataset with the three possible capital expenditures 
        area_dataset = xr.Dataset()
        area_dataset['latitude'] = latitudes
        area_dataset['longitude'] = longitudes
        area_dataset['area'] = (['latitude', 'longitude'], areas_extended, {'latitude': latitudes, 'longitude': longitudes})
        
        return area_dataset
    
    def plot_supply_curve(self, supply_dataset, technology, graphmarking=None, nolegend=None):
        
        # Convert the supply curve dataset to a dataframe
        supply_df = supply_dataset.to_dataframe()
        
        color_df = self.color_df  # Adjust the file path as needed

        # Merge the color DataFrame with your final_df based on 'country'
        plotting_df = supply_df.merge(color_df, on='country', how='left')

        rounded_df = plotting_df.round({'levelised_cost': 2})
        sum_df = rounded_df.groupby(['levelised_cost','country']).agg({'hydrogen_technical_potential': 'sum', 'Name': 'first', 'Region': 'first', 'Color': 'first'}).reset_index()

        sorted_df = sum_df.sort_values(['levelised_cost', 'country'])
        
        italy_data = pd.DataFrame({'Color': "lime", 'hydrogen_technical_potential': [], "levelised_cost": [], "country": 109, "Region": "Europe", "Name":"Italy"})
        italy_cumulative_sum = italy_data['hydrogen_technical_potential'].cumsum()
        sorted_df = pd.concat([sorted_df, italy_data])
        
        # Initialize your color_labels dictionary
        color_labels = {}

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
        plt.tight_layout(pad=1.1)
        # Group the data by color
        grouped = sorted_df.groupby('Color')
           
           
            
            
        # Iterate through each color group and plot the data as a line
        for color, group in grouped:
            cumulative_sum = group['hydrogen_technical_potential'].cumsum()
            
        # Add the color and label to the dictionary
            if color in {'navy', 'seagreen', 'gold', 'orange', 'red'}:
                if color == 'navy':
                    color_labels[color] = "Europe (aggregated)"
                else:
                    color_labels[color] = group['Name'].iloc[0]
                #ax.plot(group['levelised_cost'].values, group['hydrogen_technical_potential'].cumsum().values / 1e+06, label=color_labels[color])
                ax.plot(group['hydrogen_technical_potential'].cumsum().values / 1e+06, group['levelised_cost'].values, label=color_labels[color], color=color)

        
        # Set labels
        ax.set_ylabel('LCOH (US$/kg)', va='center', fontsize=20)
        ax.set_xlabel('Cumulative Hydrogen \n Potential (Mt/yr)', fontsize=20)
        ax.set_xlim(xmin=0, xmax=2000)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6,  7, 8, 9, 10])
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.text(0.99, 0.94, technology, horizontalalignment='right', transform=ax.transAxes, fontsize=20, fontweight='bold')

        
        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the label size as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the label size as needed
        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
        # Create a legend based on the color_labels dictionary
        if nolegend is None:
            plt.legend(title='Countries', loc='lower right', fontsize=18, title_fontsize=20, ncol=2)
        plt.savefig(self.output_folder + "G7SupplyCurve" + technology + ".png", bbox_inches="tight", pad_inches=0.5)
            
        # Show the plot
        plt.show()
        
        
    def calculate_embodied_carbon(self, data, technology, elec_tech):
        
        # Specify lifetime
        lifetime = 20

        # Specify constants
        CI_wind = 509
        CI_solar = 615
        CI_pem = 0.1485 
        CI_alk = 0.137 

        # Calculate the embodied carbon for wind
        if technology == "Wind":
            embodied_carbon = 1000 * CI_wind
        elif technology == "Solar":
            embodied_carbon = 1000 * CI_solar
        elif technology == "Hybrid":
            embodied_carbon = 1000 * CI_wind * (1 - data['Optimal_SF']/100) + 1000 * CI_solar * data['Optimal_SF']/100

        if elec_tech == "PEM":
            CI_elec = CI_pem
        else:
            CI_elec = CI_alk

        # Calculate the annual hydrogen production for wind
        total_production = data['hydrogen_production']*lifetime*1000 

        # Calculate the embodied carbon per kg H2 for each location
        carbon_intensity = embodied_carbon / total_production + CI_elec
        data['carbon_intensity'] = carbon_intensity

        return data
    
    
    def calculate_marginal_cost(self, supply_curve_ds, production_amount, technology=None):

        def clean_results(dataframe, cutoff):
            # Sort by levelised cost
            sorted_supply = dataframe.sort_values(by=['levelised_cost'])

            # Remove rows that are empty
            cleaned_df = sorted_supply.dropna(axis='index')
            final_df = cleaned_df.copy()

            # Apply a threshold for cost (if applicable)
            index_names = final_df[ final_df['levelised_cost'] >= cutoff ].index
            final_df.drop(index_names, inplace = True)

            # Apply a threshold for locations with zero utilisation (if applicable)
            util_index_names = final_df[final_df['hydrogen_technical_potential'] == 0 ].index
            final_df.drop(util_index_names, inplace = True)

            return final_df


        # Select relevant dataset and convert to a dataframe
        technology_df = supply_curve_ds.to_dataframe()

        # Clean and sort the results
        cleaned_results_df = clean_results(technology_df , 10)
        sorted_lc = cleaned_results_df.sort_values(by=['levelised_cost'])
        sorted_lc.loc[:, 'total_cumulative_potential'] = sorted_lc['hydrogen_technical_potential'].cumsum()

        # Get closest value
        df_closest = sorted_lc[sorted_lc['total_cumulative_potential'] > production_amount].iloc[0]
        marginal_cost = df_closest['levelised_cost']
        cost = np.round(marginal_cost, 2)
        print(f"The Marginal Cost of Producing {production_amount / 1e+6}Mt per year is US${cost}/kg using " + technology)
        
        
        
        
    def produce_country_supply_curves(self, upper_bound, largescale=None):
    
        # Get Lowest Cost and Corresponding Production
        solar_costs = self.PEM_100['levelised_cost']
        wind_costs = self.PEM_0['levelised_cost']
        hybrid_costs = self.PEM_cheapest_sf['levelised_cost']
        
        # Get corresponding production
        solar_production = self.PEM_100['hydrogen_production']
        wind_production = self.PEM_0['hydrogen_production']
        hybrid_production = self.PEM_cheapest_sf['hydrogen_production']
        solar_fractions=self.PEM_cheapest_sf['Optimal_SF'].drop_vars('solar_fraction')/100
        
        def cutoff_value(lcoh, production, upper_bound, solar_fractions=None):
            bounded_production = xr.where(lcoh > upper_bound, np.nan, production)
            bounded_costs = xr.where(lcoh > upper_bound, np.nan, lcoh)
            if solar_fractions is not None:
                bounded_sf = xr.where(lcoh > upper_bound, np.nan, solar_fractions)
                
                return bounded_production, bounded_costs, bounded_sf
            else:
                return bounded_production, bounded_costs
        
        # Apply cutoff value
        solar_bounded_production, solar_bounded_costs = cutoff_value(solar_costs, solar_production, upper_bound)
        wind_bounded_production, wind_bounded_costs = cutoff_value(wind_costs, wind_production, upper_bound)
        hybrid_bounded_production, hybrid_bounded_costs, hybrid_bounded_sf = cutoff_value(hybrid_costs.drop_vars('solar_fraction'), hybrid_production.drop_vars('solar_fraction'), upper_bound, solar_fractions=solar_fractions)
        print(hybrid_bounded_costs)
        print(hybrid_bounded_sf)
        # Get Supply Curves
        solar_supply_ds = self.get_supply_curves(solar_bounded_costs, solar_bounded_production, "Solar PV")
        wind_supply_ds = self.get_supply_curves(wind_bounded_costs, wind_bounded_production, "Wind")
        hybrid_supply_ds = self.get_supply_curves(hybrid_bounded_costs, hybrid_bounded_production, "Hybrid", solar_fractions=hybrid_bounded_sf)
        
        # Rename supply curve
        solar_supply_ds = self.get_supply_curves(solar_bounded_costs, solar_bounded_production, "Solar PV")
        wind_supply_ds = self.get_supply_curves(wind_bounded_costs, wind_bounded_production, "Wind")
        hybrid_supply_ds = self.get_supply_curves(hybrid_bounded_costs, hybrid_bounded_production, "Hybrid", solar_fractions=solar_fractions)
        
        # Rename solar
        rename_solar = {var: f"{var}_SOLAR" for var in solar_supply_ds.data_vars}
        solar_supply_ds = solar_supply_ds.rename(rename_solar)
        
        # Rename wind
        rename_wind = {var: f"{var}_WIND" for var in wind_supply_ds.data_vars}
        wind_supply_ds = wind_supply_ds.rename(rename_wind)
        
        # Rename solar
        rename_hybrid = {var: f"{var}_HYBRID" for var in hybrid_supply_ds.data_vars}
        hybrid_supply_ds = hybrid_supply_ds.rename(rename_hybrid)
        
        combined_supply_ds = xr.merge([solar_supply_ds, wind_supply_ds , hybrid_supply_ds])   
        
        # Convert to a dataframe
        combined_supply_ds = combined_supply_ds.drop_vars({"country_SOLAR", "country_WIND"}).rename({"country_HYBRID": "country"})

        # Plot country curves
        for i in np.arange(1, 252, 1):
            self.plot_country_curve(combined_supply_ds, i, largescale)
        
        return combined_supply_ds
        
        
        
        
    def plot_country_curve(self, supply_dataset, country, largescale=None):
        
        # Convert the supply curve dataset to a dataframe
        supply_df = supply_dataset.to_dataframe()
        
        # Get the colormap / country mapping
        color_df = self.color_df 

        # Merge the color DataFrame with supply_df based on 'country'
        plotting_df = pd.merge(supply_df, color_df, on='country', how='outer')
        selected_df = plotting_df.loc[plotting_df['country'] == country].copy()
        
        # Get the corresponding technologies
        solar_df = selected_df[['hydrogen_technical_potential_SOLAR', 'levelised_cost_SOLAR']].rename(columns={"levelised_cost_SOLAR":"levelised_cost", "hydrogen_technical_potential_SOLAR": "tech_potential"}).sort_values(['levelised_cost']).dropna(axis=0, how="any")
        wind_df = selected_df[['hydrogen_technical_potential_WIND', 'levelised_cost_WIND']].rename(columns={"levelised_cost_WIND":"levelised_cost", "hydrogen_technical_potential_WIND": "tech_potential"}).sort_values(['levelised_cost']).dropna(axis=0, how="any")
        hybrid_df = selected_df[['hydrogen_technical_potential_HYBRID', 'levelised_cost_HYBRID']].rename(columns={"levelised_cost_HYBRID":"levelised_cost", "hydrogen_technical_potential_HYBRID": "tech_potential"}).sort_values(['levelised_cost']).dropna(axis=0, how="any")
        
        
        # Get required parameters
        region_name = selected_df['Region'].iloc[0]
        country_iso = selected_df['ISO Code'].iloc[0]
        country_name = selected_df['Name'].iloc[0]
        subregion_name = selected_df['Final Region'].iloc[0]    
        
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")

        # Iterate through each color group and plot the data as a line
        for dataframe in solar_df, wind_df, hybrid_df:
            if dataframe.empty:
                cumulative_sum = np.array([np.nan, np.nan])
                levelised_cost = np.array([np.nan, np.nan])
            else:
                cumulative_sum = dataframe['tech_potential'].cumsum().values
                levelised_cost = dataframe['levelised_cost'].values
            if dataframe is solar_df:
                color = "red"
                technology = "Solar"
            elif dataframe is wind_df:
                color = "blue"
                technology = "Wind"
            else:
                color = "purple"
                technology = "Hybrid"
            
            # Add the color and label to the dictionary and plot
            ax.plot(cumulative_sum / 1e+6, levelised_cost, label=technology, color=color, linewidth=5)

        # Set labels
        ax.set_ylabel('LCOH (US$/kg)', fontsize=40)
        ax.set_xlabel('Annual Hydrogen \nPotential (Mt/yr)', fontsize=40)
        ax.set_ylim([0, 10])
        ax.set_xlim([0, 1000])
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the label size as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the label size as needed
        ax.text(0.81, 0.88, str(country_iso), transform=ax.transAxes, fontsize=60, fontweight='bold')
        # Create a legend based on the color_labels dictionary
        plt.legend(title='Technology', loc='lower right', fontsize=25, title_fontsize=30)
        plt.savefig(self.output_folder + "/FINAL/REGION/"+ str(subregion_name) + str(country_iso) +"SupplyCurve.png", bbox_inches='tight')
            
        # Show the plot
        plt.show()
        

    
    def get_relative_costs(self, data, countries_1, countries_2):

        geodata = self.land_mapping
        geodata = geodata.reindex({"latitude":data.latitude, "longitude":data.longitude})
        countries_1 = xr.where(np.isin(geodata, countries_1), data['Calculated_LCOE'], np.nan)
        countries_2 = xr.where(np.isin(geodata, countries_2), data['Calculated_LCOE'], np.nan)

        mean_1 = np.nanmean(countries_1)
        mean_2 = np.nanmean(countries_2)
        increase = mean_1 / mean_2
        print(f"Countries in group 1 are {(increase-1)*100:0.2f}% more expensive than in group 2") 

    def plot_data_shading(self, values, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend_set=None, graphmarking=None, special_value=None, hatch_label=None, hatch_color=None, hatch_label_2=None, special_value_2=None, hatch_color_2=None, hatch_label_3=None, special_value_3=None, hatch_color_3=None, center_norm=None, log_norm=None, formatting=None):      
    
        # Get latitudes and longitudes
        latitudes = values.latitude
        longitudes = values.longitude

        # create the heatmap using pcolormesh
        if anchor is None:
            anchor = 0.355
        fig = plt.figure(figsize=(30, 15), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        if center_norm is None:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.Normalize(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
        else:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.CenteredNorm(vcenter= center_norm, halfrange = center_norm - tick_values[0]), transform=ccrs.PlateCarree(), cmap=cmap)
        if log_norm is not None:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.LogNorm(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)


        # Check if there is a need for extension
        values_min = np.nanmin(values)
        values_max = np.nanmax(values)
        if values_min < tick_values[0]:
            extend = "min"
        elif values_max > tick_values[-1]:
            extend = "max"
        elif (values_max > tick_values[-1]) & (values_min < tick_values[0]):
            extend = "both"
        elif extend_set is not None:
            extend = extend_set
        else:
            extend = "neither"


        axins = inset_axes(
        ax,
        width="1.5%",  
        height="80%",  
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
        if formatting is None:
            formatting = "%0.0f"
        cb = fig.colorbar(heatmap, cax=axins, shrink=0.5, ticks=tick_values, format=formatting, extend=extend, anchor=(0.15, anchor))


        cb.ax.tick_params(labelsize=20)
        if title is not None:
            cb.ax.set_title(title, fontsize=25)

        if hatch_color is None:
            hatch_color = "silver"
            hatch_color_2 = "mistyrose"
            hatch_color_3 = "lavender"

        # Add the special shading
        if special_value is not None:
            special_overlay = np.where(values == special_value, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['//'], colors=hatch_color, linewidth=0.05, transform=ccrs.PlateCarree(), alpha=1)

        if special_value_2 is not None:
            special_overlay = np.where(values == special_value_2, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['\\'], colors=hatch_color_2, linewidth=0.05, transform=ccrs.PlateCarree(), alpha=1)

        if special_value_3 is not None:
            special_overlay = np.where(values == special_value_3, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['x'], colors=hatch_color_3, linewidth=0.05, transform=ccrs.PlateCarree(), alpha=1)

        # set the extent and aspect ratio of the plot
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
        ax.set_aspect(1)

        # add axis labels and a title
        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
        ax.add_feature(borders, edgecolor='black', linestyle=':')
        ax.coastlines()
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        ax.coastlines()
        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        hatch_patches=[]
        if special_value is not None and hatch_label is not None:
            hatch_patch_1 = Patch(facecolor=hatch_color, edgecolor='black', hatch="//", label=hatch_label)
            hatch_patches.append(hatch_patch_1)

        if hatch_label_2 is not None:
            hatch_patch_2 = Patch(facecolor=hatch_color_2, edgecolor='black', hatch="\\", label=hatch_label_2)
            hatch_patches.append(hatch_patch_2)


        if hatch_label_3 is not None:
            hatch_patch_3 = Patch(facecolor=hatch_color_3, edgecolor='black', hatch="x", label=hatch_label_3)
            hatch_patches.append(hatch_patch_3)

        if hatch_patches:
            ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")


            return 

    
# Inputs to the Results Processor
land_cover = r"/Users/china_lcoh_model//CHINA_DATA/DATA/GlobalLandCover.nc"
colormap = r"/Users/china_lcoh_model//CHINA_DATA/DATA/colormap.csv"
land_mapping = "/Users/china_lcoh_model/DATA/LandUseCSV.csv"       
pem_results_folder = "/Users/china_lcoh_model//V3_PEM_COLLATED_OUTPUTS/"
alkaline_results_folder = "/Users/china_lcoh_model//V4_ALK_COLLATED_OUTPUTS/"
output_folder = "/Users/china_lcoh_model//PLOTS/"
input_data_path = "/Users/china_lcoh_model//DATA/"
country_codes = r"/Users/china_lcoh_model//CHINA_DATA/DATA/countrycodecolormap.csv"
colors = ["#FFA109", "#0BADD9", "#0E60BA", "#BE0017"]
combined_country_grids = "/Users/china_lcoh_model/DATA/country_grid_combined.nc"
country_index = "/Users/china_lcoh_model/DATA/country_mapping.csv"
capex_mapping = "/Users/china_lcoh_model/DATA/Adjusted_Country_Costs.csv"
file_output_folder = "/Users/china_lcoh_model/PLOTS/"

# Call the Results Processor Model
results_model = ResultsProcessor(pem_results_folder, alkaline_results_folder, output_folder, input_data_path,combined_country_grids, country_index, land_cover, land_mapping, colormap, colors, country_codes, capex_mapping)

# Set latitude  and longitude limits for China
lat_lon = [15, 60, 75, 140]

# Calculate the cheapest PEM and Alkaline results
cheapest_PEM = results_model.get_cheapest_lcoh("PEM")
cheapest_ALK = results_model.get_cheapest_lcoh("ALK")

# Calculate the regional costs (for China)
PEM_China = results_model.calculate_new_lcoh(results_model.PEM_results, "PEM", selected_countries=[47]).sel(latitude=slice(lat_lon[0], lat_lon[1]), longitude=slice(lat_lon[2], lat_lon[3]))
ALK_China = results_model.calculate_new_lcoh(results_model.ALK_results, "PEM", selected_countries=[47]).sel(latitude=slice(lat_lon[0], lat_lon[1]), longitude=slice(lat_lon[2], lat_lon[3]))

# Calculate the regional costs (for China, based on the lowest electrolyser cost)
PEM_China_cheapest = results_model.calculate_new_lcoh(results_model.PEM_results, "PEM", selected_countries=[47], electrolyser_cost = 484.7).sel(latitude=slice(lat_lon[0], lat_lon[1]), longitude=slice(lat_lon[2], lat_lon[3]))
ALK_China_cheapest = results_model.calculate_new_lcoh(results_model.ALK_results, "PEM", selected_countries=[47], electrolyser_cost = 192).sel(latitude=slice(lat_lon[0], lat_lon[1]), longitude=slice(lat_lon[2], lat_lon[3]))

# Calculate the regional costs (for China, based on the highest electrolyser cost)
PEM_China_highest = results_model.calculate_new_lcoh(results_model.PEM_results, "PEM", selected_countries=[47], electrolyser_cost = 750.4).sel(latitude=slice(lat_lon[0], lat_lon[1]), longitude=slice(lat_lon[2], lat_lon[3]))
ALK_China_highest = results_model.calculate_new_lcoh(results_model.ALK_results, "PEM", selected_countries=[47], electrolyser_cost = 222).sel(latitude=slice(lat_lon[0], lat_lon[1]), longitude=slice(lat_lon[2], lat_lon[3]))

import matplotlib.colors as colors

def produce_lcoh_plots(results_model, data, name, technology, lcoh_cmap, sf_cmap, output_folder, additional=None, sf_additional=None):
    
    # Produce plot for solar
    results_model.plot_data_shading(data['Regional_LCOH'].sel(solar_fraction=100), tick_values=[0, 1, 2, 3, 4, 5], cmap=lcoh_cmap, title="Levelised\n Cost of\n Hydrogen\n(USD/kg)\n", filename = output_folder + name + "_" + technology + "_Solar_LCOH", graphmarking="a")  
    print(f"Minimum for solar is{np.nanmin(data['Regional_LCOH'].sel(solar_fraction=100))}")
    
    # Produce plot for wind
    results_model.plot_data_shading(data['Regional_LCOH'].sel(solar_fraction=0), tick_values=[0, 1, 2, 3, 4, 5], cmap=lcoh_cmap, title="Levelised\n Cost of\n Hydrogen\n(USD/kg)\n", filename = output_folder + name + "_" + technology + "_Wind_LCOH", graphmarking="b")  
    print(f"Minimum for wind is{np.nanmin(data['Regional_LCOH'].sel(solar_fraction=0))}")
    
    # Calculate the cheapest solar fraction
    optimal_sf = results_model.get_cheapest_lcoh(technology, specified_results=data)
    
    # Produce plot of cheapest price
    results_model.plot_data_shading(optimal_sf['Regional_LCOH'], tick_values=[0, 1, 2, 3, 4, 5], cmap=lcoh_cmap, title="Levelised\n Cost of\n Hydrogen\n(USD/kg)\n", filename = output_folder + name + "_" + technology + "_Hybrid_LCOH", graphmarking="c") 
    print(f"Minimum for optimal is {optimal_sf['Regional_LCOH'].values}")
    
    if additional is not None:
        results_model.plot_data_shading(optimal_sf['Regional_LCOH'], tick_values=[0, 1, 2, 3, 4, 5], cmap=lcoh_cmap, title="Levelised\n Cost of\n Hydrogen\n(USD/kg)\n", filename = output_folder + name + "_" + technology + "_Hybrid_LCOH_MAIN", graphmarking=additional) 
        sf_additional = "b"
    else:
        sf_additional = sf_additional
    
    
    # Produce plot of cheapest solar fraction
    results_model.plot_data_shading(optimal_sf['Optimal_SF'], tick_values=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], cmap=sf_cmap, title="Lowest\nCost\nHybrid\nConfig\n(%, Solar)\n", filename = output_folder + name + "_" + technology + "_Hybrid_SF", graphmarking=sf_additional)
    
    # Plot Equivalent Price in USD/MBTu
    price_btu = optimal_sf['Regional_LCOH']/33.33*1000*0.29307107
    print(f"Minimum gas price is {np.nanmin(price_btu)}")
    results_model.plot_data_shading(price_btu, tick_values=[0, 10, 20, 30, 40, 50], cmap=lcoh_cmap, title="\nLevelised\n Cost of\n Hydrogen\n(USD/\nMBtu)\n", filename=output_folder + name + "_" + technology + "_Hybrid_BTU_NS", graphmarking="a")
    price_btu = xr.where(price_btu < 9.4, 111, price_btu)
    price_btu = xr.where((price_btu < 13.2) & (price_btu > 9.4), 222, price_btu)
    price_btu = xr.where((price_btu < 33.4) & (price_btu > 13.2), 999, price_btu)
    print(f"Minimum gas price is {np.nanmin(price_btu)}")
    results_model.plot_data_shading(price_btu, tick_values=[0, 10, 20, 30, 40, 50], cmap=lcoh_cmap, title="\nLevelised\n Cost of\n Hydrogen\n(USD/\nMBtu)\n", filename=output_folder + name + "_" + technology + "_Hybrid_BTU", special_value=111, hatch_label="Below Q1&2 2024 JKM Average (USD 9.4/MBTu)", special_value_2=222, hatch_label_2="Below Q1-4 2023 JKM Average (USD 13.2/MBTu)", special_value_3=999, hatch_label_3="Below Q1-4 2022 JKM Average (USD 33.4/MBTu)", graphmarking="b")
    
    return optimal_sf, price_btu
    
# Produce PEM LCOH Heatmaps  
#optimal_PEM,optimal_PEM_btu  = produce_lcoh_plots(results_model, PEM_China, "AVG", "PEM", "RdPu", "plasma", "/Users/china_lcoh_model/PLOTS/")
optimal_PEM_cheapest, cheapest_PEM_btu = produce_lcoh_plots(results_model, PEM_China_cheapest, "LOW", "PEM", "RdPu", "plasma", "/Users/china_lcoh_model/PLOTS/", sf_additional="d")
#optimal_PEM_highest, highest_PEM_btu = produce_lcoh_plots(results_model, PEM_China_highest, "HIGH", "PEM", "RdPu", "plasma", "/Users/china_lcoh_model/PLOTS/")

# Produce ALK LCOH Heatmaps  
#optimal_ALK, optimal_ALK_btu = produce_lcoh_plots(results_model, ALK_China, "AVG", "ALK", "GnBu", "plasma", "/Users/china_lcoh_model/PLOTS/")
optimal_ALK_cheapest, cheapest_ALK_btu = produce_lcoh_plots(results_model, ALK_China_cheapest, "LOW", "ALK", "GnBu", "plasma", "/Users/china_lcoh_model/PLOTS/", "d")
#optimal_ALK_highest, highest_ALK_btu = produce_lcoh_plots(results_model, ALK_China_highest, "HIGH", "ALK", "GnBu", "plasma", "/Users/china_lcoh_model/PLOTS/")




def china_production_scale(results_model, pem_data, alk_data, upper_bound, technology):
    
    def cutoff_value(lcoh, production, upper_bound, solar_fraction=None):
        bounded_production = xr.where(lcoh > upper_bound, np.nan, production)
        bounded_costs = xr.where(lcoh > upper_bound, np.nan, lcoh)

        if solar_fraction is not None:
            bounded_solar_fraction = xr.where(lcoh > upper_bound, np.nan, solar_fraction/100)

            return bounded_production, bounded_costs, bounded_solar_fraction

        return bounded_production, bounded_costs, None
    
    if technology == "Hybrid":
        alk_sf = alk_data['Optimal_SF']
        pem_sf = pem_data['Optimal_SF']
    else:
        alk_sf = None
        pem_sf = None


    # Apply the cutoff
    pem_prod, pem_costs, pem_sf_bounded = cutoff_value(pem_data['Regional_LCOH'], pem_data['hydrogen_production'], upper_bound, solar_fraction = pem_sf)
    alk_prod, alk_costs, alk_sf_bounded = cutoff_value(alk_data['Regional_LCOH'], alk_data['hydrogen_production'], upper_bound, solar_fraction = alk_sf)
    

    # Get Supply Curves
    alkaline_ds = results_model.get_supply_curves(alk_costs,  alk_prod, technology, solar_fractions = alk_sf_bounded)
    pem_ds = results_model.get_supply_curves(pem_costs,  pem_prod, technology, solar_fractions = pem_sf_bounded)
    
    # Get Supply Curves for Wind
    
    return alkaline_ds, pem_ds

alkaline_ds, pem_ds = china_production_scale(results_model, optimal_PEM_cheapest, optimal_ALK_cheapest, 10, "Hybrid")
alkaline_wind, pem_wind = china_production_scale(results_model, optimal_PEM_cheapest, optimal_ALK_cheapest, 10, "Wind")
alkaline_solar, pem_solar = china_production_scale(results_model, optimal_PEM_cheapest, optimal_ALK_cheapest, 10, "Solar PV")


tech_potential = input("Plot Chinese technical potential? Y/N")
if tech_potential == "Y":
    results_model.plot_data_shading(alkaline_ds['hydrogen_technical_potential']/1e+06, tick_values=[0, 0.5, 1, 1.5, 2], cmap="GnBu", title="\nHydrogen\nTechnical\nPotential\n(Mt/yr)\n", filename="Tech_Potential_Alk", formatting="%0.1f", graphmarking="b")
    results_model.plot_data_shading(pem_ds['hydrogen_technical_potential']/1e+06, tick_values=[0, 0.5, 1, 1.5, 2], cmap="PuRd", title="\nHydrogen\nTechnical\nPotential\n(Mt/yr)\n", filename=file_output_folder + "Tech_Potential_PEM", formatting="%0.1f", graphmarking="a")


    #results_model.plot_data_shading(alkaline_solar['hydrogen_technical_potential']/1e+06, tick_values=[0, 0.5, 1, 1.5, 2], cmap="Oranges", title="\nHydrogen\nTechnical\nPotential\n(Mt p.a.)\n", filename=file_output_folder + "Solar_Potential_Alk", formatting="%0.1f", graphmarking="a")
    #results_model.plot_data_shading(alkaline_wind['hydrogen_technical_potential']/1e+06, tick_values=[0, 0.5, 1, 1.5, 2], cmap="Greens", title="\nHydrogen\nTechnical\nPotential\n(Mt p.a.)\n", filename=file_output_folder + "Wind_Potential_Alk", formatting="%0.1f", graphmarking="b")

    #results_model.plot_data_shading(pem_solar['hydrogen_technical_potential']/1e+06, tick_values=[0, 0.5, 1, 1.5, 2], cmap="Oranges", title= "\nHydrogen\nTechnical\nPotential\n(Mt p.a.)\n", filename=file_output_folder + "Solar_Potential_PEM", formatting="%0.1f", graphmarking="a")
    #results_model.plot_data_shading(pem_wind['hydrogen_technical_potential']/1e+06, tick_values=[0, 0.5, 1, 1.5, 2], cmap="Greens", title="\nHydrogen\nTechnical\nPotential\n(Mt p.a.)\n", filename=file_output_folder + "Wind_Potential_PEM", formatting="%0.1f", graphmarking="b")


### Produce CSV files for plots
output_folder = "/Users/china_lcoh_model/CSV_FILES/"

## Figure 2
optimal_PEM_cheapest.to_dataframe()[['Regional_LCOH', 'hydrogen_production', 'Optimal_SF']].rename({"Regional_LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_production": "Hydrogen Production (TPA for 1MW REN)", "Optimal_SF": "Lowest Cost Solar Fraction (%)"}).to_csv(output_folder + "Figure2a.csv")
optimal_ALK_cheapest.to_dataframe()[['Regional_LCOH', 'hydrogen_production', 'Optimal_SF']].rename({"Regional_LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_production": "Hydrogen Production (TPA for 1MW REN)", "Optimal_SF": "Lowest Cost Solar Fraction (%)"}).to_csv(output_folder + "Figure2b.csv")

## Figure 3
optimal_PEM_cheapest.to_dataframe()[['Regional_LCOH', 'hydrogen_production', 'Optimal_SF']].rename({"Regional_LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_production": "Hydrogen Production (TPA for 1MW REN)", "Optimal_SF": "Lowest Cost Solar Fraction (%)"}).to_csv(output_folder + "Figure3.csv")

## Figure 4
cheapest_PEM_btu.to_dataframe().reset_index().rename({"Regional_LCOH":"Levelised Cost of Hydrogen (USD/MBTu)"}).to_csv(output_folder + "Figure4.csv")

## Figure 5
pem_ds.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)", "Optimal_SF":"Least Cost Solar Fraction"}).to_csv(output_folder + "Figure5a.csv")
alkaline_ds.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)", "Optimal_SF":"Least Cost Solar Fraction"}).to_csv(output_folder + "Figure5b.csv")


# Solar Potentials
pem_solar.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)"}).to_csv(output_folder + "PEM_Solar_Potential.csv")
alkaline_solar.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)"}).to_csv(output_folder + "ALK_Solar_Potential.csv")

# Wind Potentials
pem_wind.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)"}).to_csv(output_folder + "PEM_Wind_Potential.csv")
alkaline_wind.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)"}).to_csv(output_folder + "ALK_Wind_Potential.csv")

global_perspective = input("Plot GH2 countries? Y/N")
if global_perspective == "Y":
    # Get results for exporting countries
    PEM_export = results_model.calculate_new_lcoh(results_model.PEM_results, "PEM", selected_countries=[45, 12, 232, 136, 11, 19, 21, 97, 54, 55, 58, 63, 69, 74, 56, 88, 99, 101, 109, 134, 132, 133, 152, 165, 178, 183, 188, 201, 199, 67, 196, 8, 57, 64, 68, 114, 150, 159, 163, 248, 104, 100, 171, 9, 30, 48, 76])
    PEM_export = results_model.get_cheapest_lcoh("PEM", specified_results=PEM_export)

    # Merge results
    PEM_merged = xr.merge([PEM_export['Regional_LCOH'].drop_vars('solar_fraction'), optimal_PEM_cheapest['Regional_LCOH'].drop_vars('solar_fraction')])

    # Colour EU-27 blue
    #PEM_merged['Regional_LCOH'] = xr.where(results_model.country_grid['country'].isin([11, 19, 21, 97, 54, 55, 58, 63, 69, 74, 56, 88, 99, 101, 109, 134, 132, 133, 152, 165, 178, 183, 188, 201, 199, 67, 196]), 999, PEM_merged['Regional_LCOH'])

    # Remove offshore locations
    PEM_merged = xr.where(np.isnan(results_model.PEM_results['levelised_cost'].sel(solar_fraction=100)), np.nan, PEM_merged)


    results_model.plot_data_shading(PEM_merged['Regional_LCOH'].sel(latitude=slice(-65, 90)), tick_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap="Greens", title="Levelised\nCost\nof Hydrogen\n(USD/kg)", filename="Global_GH2", special_value=999)

    # Copy for PEM subsidies
    PEM_subsidies = PEM_merged

    # Account for EU and US subsidies - EU is an average of 0.435 whilst the US is 1 (if wage & apprenticeship conditions are met)
    PEM_subsidies['Regional_LCOH'] = xr.where(results_model.country_grid['country'].isin([11, 19, 21, 97, 54, 55, 58, 63, 69, 74, 56, 88, 99, 101, 109, 134, 132, 133, 152, 165, 178, 183, 188, 201, 199, 67, 196]), PEM_merged['Regional_LCOH'] - 0.470474, PEM_merged['Regional_LCOH'])
    PEM_subsidies['Regional_LCOH'] = xr.where(results_model.country_grid['country'] == 232, PEM_subsidies['Regional_LCOH'] - 1, PEM_subsidies['Regional_LCOH'])

    results_model.plot_data_shading(PEM_subsidies['Regional_LCOH'].sel(latitude=slice(-65, 90)), tick_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap="Greens", title="Levelised\nCost\nof Hydrogen\n(USD/kg)", special_value=999)
    
## Figure 2
optimal_PEM_cheapest.to_dataframe()[['Regional_LCOH', 'hydrogen_production', 'Optimal_SF']].rename({"Regional_LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_production": "Hydrogen Production (TPA for 1MW REN)", "Optimal_SF": "Lowest Cost Solar Fraction (%)"}).to_csv(output_folder + "Figure2cd.csv")
pem_solar.to_dataframe()[['LCOH', 'hydrogen_technical_potential']].rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential": "Hydrogen Production (TPA)"}).to_csv(output_folder + "Figure2a.csv")
pem_wind.to_dataframe()[['LCOH', 'hydrogen_technical_potential']].rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential": "Hydrogen Production (TPA)"}).to_csv(output_folder + "Figure2b.csv")


## Figure 4
cheapest_PEM_btu.to_dataframe().reset_index().rename({"Regional_LCOH":"Levelised Cost of Hydrogen (USD/MBTu)"}).to_csv(output_folder + "Figure3.csv")


# Solar Potentials
pem_solar.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)"}).to_csv(output_folder + "PEM_Solar_Potential.csv")
alkaline_solar.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)"}).to_csv(output_folder + "ALK_Solar_Potential.csv")

# Wind Potentials
pem_wind.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)"}).to_csv(output_folder + "PEM_Wind_Potential.csv")
alkaline_wind.to_dataframe().reset_index().rename({"LCOH":"Levelised Cost of Hydrogen (USD/kg)", "hydrogen_technical_potential":"Hydrogen Technical Potential (TPA)"}).to_csv(output_folder + "ALK_Wind_Potential.csv")
