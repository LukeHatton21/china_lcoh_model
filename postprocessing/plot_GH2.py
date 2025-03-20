def plot_data_shading(values, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend_set=None, graphmarking=None, special_value=None, hatch_label=None, hatch_color=None, hatch_label_2=None, special_value_2=None, hatch_color_2=None, hatch_label_3=None, special_value_3=None, hatch_color_3=None, center_norm=None, log_norm=None, formatting=None):      
    
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
    if extend_set is not None:
        extend = extend_set
    else:
        extend = extend


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
        cb.ax.set_title(title, fontsize=20)
        
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


# Get results for exporting countries
PEM_export = results_model.calculate_new_lcoh(results_model.PEM_results, "PEM", selected_countries=[45, 12, 232, 136, 11, 19, 21, 97, 54, 55, 58, 63, 69, 74, 56, 88, 99, 101, 109, 134, 132, 133, 152, 165, 178, 183, 188, 201, 199, 67, 196, 8, 57, 64, 68, 114, 150, 159, 163, 248, 104, 100, 171, 9, 30, 48])
PEM_export = results_model.get_cheapest_lcoh("PEM", specified_results=PEM_export)

# Merge results
PEM_merged = xr.merge([PEM_export['Regional_LCOH'].drop_vars('solar_fraction'), optimal_PEM_cheapest['Regional_LCOH'].drop_vars('solar_fraction')])


# Copy for PEM subsidies
PEM_subsidies = PEM_merged


# Remove offshore locations
PEM_merged = xr.where(np.isnan(PEM_merged['Regional_LCOH']), 999, PEM_merged)
PEM_merged = xr.where(np.isnan(results_model.PEM_results['levelised_cost'].sel(solar_fraction=100)), np.nan, PEM_merged)

# Plot
plot_data_shading(PEM_merged['Regional_LCOH'].sel(latitude=slice(-65, 90)), tick_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap="PuBu", title="Levelised\nCost\nof Hydrogen\n(USD/kg)\n", special_value=999, hatch_color="grey", hatch_label="Countries not\nexamined", filename="Global_GH2", graphmarking="a")

# Account for EU and US subsidies - EU is an average of 0.435 whilst the US is 1 (if wage & apprenticeship conditions are met)
PEM_subsidies['Regional_LCOH'] = xr.where(results_model.country_grid['country'].isin([11, 19, 21, 97, 54, 55, 58, 63, 69, 74, 56, 88, 99, 101, 109, 134, 132, 133, 152, 165, 178, 183, 188, 201, 199, 67, 196]), PEM_merged['Regional_LCOH'] - 0.470474, PEM_merged['Regional_LCOH'])
PEM_subsidies['Regional_LCOH'] = xr.where(results_model.country_grid['country'] == 232, PEM_subsidies['Regional_LCOH'] - 1, PEM_subsidies['Regional_LCOH'])

# Remove offshore locations
PEM_subsidies = xr.where(np.isnan(PEM_merged['Regional_LCOH']), 999, PEM_subsidies)
PEM_subsidies = xr.where(np.isnan(results_model.PEM_results['levelised_cost'].sel(solar_fraction=100)), np.nan, PEM_subsidies)

# Plot Subsidies
plot_data_shading(PEM_subsidies['Regional_LCOH'].sel(latitude=slice(-65, 90)), tick_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap="PuBu", title="Levelised\nCost\nof Hydrogen\n(USD/kg, with\nsubsidies)\n", special_value=999, hatch_color="grey", hatch_label="Countries not\nexamined", filename="Global_GH2_subsidies", graphmarking="b")


# Calculate CAPEX reduction
PEM_subsidies['Regional_LCOH'] = xr.where(PEM_subsidies['Regional_LCOH']==999, np.nan, PEM_subsidies['Regional_LCOH'])
PEM_subsidies['CAPEX_Reduction'] = (PEM_subsidies['Regional_LCOH'] - 2) / PEM_subsidies['Regional_LCOH'] * 100
PEM_subsidies['CAPEX_Reduction'] = xr.where(PEM_subsidies['Regional_LCOH'] < 2, 111, PEM_subsidies['CAPEX_Reduction'])
PEM_subsidies['CAPEX_Reduction'] = xr.where(np.isnan(PEM_subsidies['Regional_LCOH']), 999, PEM_subsidies['CAPEX_Reduction'])
PEM_subsidies = xr.where(np.isnan(results_model.PEM_results['levelised_cost'].sel(solar_fraction=100)), np.nan, PEM_subsidies)
plot_data_shading(PEM_subsidies['CAPEX_Reduction'].sel(latitude=slice(-65, 90)), tick_values = [0, 20, 40, 60, 80, 100], cmap="PuBu", title="CAPEX\nReductions\nto reach\nUS$2/kg (%)\n", special_value=999, hatch_color="grey", special_value_2 = 111, hatch_color_2="red", hatch_label_2="Locations already\nbelow US$/kg", hatch_label="Countries not\nexamined", filename="Global_GH2_CAPEX", graphmarking="c", extend_set="neither")