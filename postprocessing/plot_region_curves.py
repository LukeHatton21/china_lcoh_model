import geopandas
import rioxarray
import xarray as xr
from shapely.geometry import mapping
import regionmask


def produce_supply_curve(solar_data, wind_data, hybrid_data, name, graphmarking=None, filename=None):
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")

    
    # Technology
    technologies = ["Solar", "Wind", "Hybrid"]
    color = ["red", "green", "purple"]
    
    for i, data in enumerate([solar_data, wind_data, hybrid_data]):
    
        # Convert to dataframe
        data_df = data.to_dataframe()
        sorted_df = data_df.sort_values(['LCOH'])

        # Create cumulative potential
        sorted_df['cumulative_sum'] = sorted_df['hydrogen_technical_potential'].cumsum()



        # Plot the data
        ax.plot(sorted_df['cumulative_sum'].values  / 1e+06, sorted_df['LCOH'].values, label=technologies[i], color=color[i])
    
    # Set labels
    ax.set_ylabel('LCOH (US$/kg)', va='center', fontsize=20)
    ax.set_xlabel('Cumulative Hydrogen \n Potential (Mt/a)', fontsize=20)
    ax.set_xlim(xmin=0, xmax=100)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6,  7, 8, 9, 10])
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.legend(loc="upper right", fontsize=20)
    ax.text(0.5, 0.94, name, horizontalalignment='center', transform=ax.transAxes, fontsize=20, fontweight='bold')

    # Set the size of x and y-axis tick labels
    ax.tick_params(axis='x', labelsize=20)  # Adjust the label size as needed
    ax.tick_params(axis='y', labelsize=20)  # Adjust the label size as needed
    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

    
    # Plot    
    if filename is not None:
        plt.savefig("/Users/lukehatton/Green Hydrogen 2024/CHINA_DATA/CURVES/"+filename + ".png")
    
    plt.show()
    

# Import China shapefile
China_Regions = geopandas.read_file('/Users/lukehatton/Green Hydrogen 2024/CHINA_DATA/SHP/China_Regions_English.shp', crs="epsg:4326")

# Create regionmask
lon = optimal_PEM_cheapest.longitude.values
lat = optimal_PEM_cheapest.latitude.values
mask = regionmask.mask_geopandas(China_Regions, lon, lat).rename({"lat":"latitude", "lon":"longitude"})

# Get index numbers
index_numbers = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 25.0, 27.0, 28.0, 52.0, 63.0, 66.0, 121.0, 177.0, 186.0, 202.0, 204.0, 205.0, 206.0, 207.0, 208.0, 217.0, 222.0, 277.0, 281.0, 284.0, 325.0, 327.0, 356.0, 464.0, 538.0, 818.0, 897.0, 898.0]
index_mapping = {0.0:"Heilongjiang", 1.0: "Inner Mongolia", 2.0:"Xinjiang", 3.0:"Jilin", 4.0:"Liaoning", 5.0:"Gansu", 6.0:"Hebei", 7.0:"Beijing", 25.0:"Shanxi", 27.0:"Tianjin", 28.0: "Beijing", 52.0:"Shaanxi", 63.0:"Ningxia", 66.0:"Qinghai", 121.0:"Shandong", 177.0:"Tibet", 186.0:"Heinan", 202.0:"Jiangsu", 204.0:"Anhui", 205.0:"Sichuan", 206.0:"Hubei", 207.0:"Chongqing", 208.0:"Shanghai", 217.0:"Shanghai", 222.0:"Zhejiang", 277.0:"Hunan", 281.0: "?", 284.0: "Jiangxi", 325.0:"Yunnan", 327.0:"Guizhou", 356.0:"Fujian", 464.0:"Guangxi", 538.0:"Taiwan", 818.0:"Hainan", 897.0:"Guangdong", 898.0:"Hong Kong"}

# Loop over the index values of the regions
for i in index_numbers[0:-1]:
    
    # Get mask
    selected_mask_solar = pem_solar.where(mask == i)
    selected_mask_wind = pem_wind.where(mask == i)
    selected_mask_hybrid = pem_ds.where(mask == i)
    
    # Get name
    name = index_mapping.get(i)
    
    # Call supply function
    produce_supply_curve(selected_mask_solar, selected_mask_wind, selected_mask_hybrid, name, filename=name+"CostCurve")
    

def produce_supply_curve(solar_data, wind_data, hybrid_data, name, graphmarking=None, filename=None):
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")

    
    # Technology
    technologies = ["Solar", "Wind", "Hybrid"]
    color = ["red", "green", "purple"]
    
    for i, data in enumerate([solar_data, wind_data, hybrid_data]):
    
        # Convert to dataframe
        data_df = data.to_dataframe()
        sorted_df = data_df.sort_values(['LCOH'])

        # Create cumulative potential
        sorted_df['cumulative_sum'] = sorted_df['hydrogen_technical_potential'].cumsum()



        # Plot the data
        ax.plot(sorted_df['cumulative_sum'].values  / 1e+06, sorted_df['LCOH'].values, label=technologies[i], color=color[i])
    
    # Set labels
    ax.set_ylabel('LCOH (US$/kg)', va='center', fontsize=20)
    ax.set_xlabel('Cumulative Hydrogen \n Potential (Mt/a)', fontsize=20)
    ax.set_xlim(xmin=0, xmax=2500)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6,  7, 8, 9, 10])
    ax.legend(loc="lower right", fontsize=20)
    ax.text(0.25, 0.9, name, horizontalalignment='center', transform=ax.transAxes, fontsize=20, fontweight='bold')

    # Set the size of x and y-axis tick labels
    ax.tick_params(axis='x', labelsize=20)  # Adjust the label size as needed
    ax.tick_params(axis='y', labelsize=20)  # Adjust the label size as needed
    if graphmarking is not None:
        ax.text(0.02, 0.95, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
        
    if filename is not None:
        plt.savefig("/Users/lukehatton/Green Hydrogen 2024/CHINA_DATA/CURVES/"+filename + ".png")

    
    # Plot
    plt.show()
    
produce_supply_curve(pem_solar, pem_wind, pem_ds, "China\nPEM", filename="PEM_Curve_China", graphmarking="a")
produce_supply_curve(alkaline_solar, alkaline_wind, alkaline_ds, "China\nALK", filename="ALK_Curve_China", graphmarking="b")

    