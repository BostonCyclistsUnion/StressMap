'''
Plotting Level of Traffic Stress

This notebook plots the Level of Traffic Stress map calculated in `LTS_OSM'.
'''
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import plotly.express as px
import shapely.geometry

city = "Cambridge"

# %% Load and Prep Data
all_lts_df = pd.read_csv(f"data/{city}_all_lts.csv")

# convert to a geodataframe for plotting
all_lts = gpd.GeoDataFrame(
    all_lts_df.loc[:, [c for c in all_lts_df.columns if c != "geometry"]],
    geometry=gpd.GeoSeries.from_wkt(all_lts_df["geometry"]), 
    crs='wgs84') # projection from graph

gdf_nodes = pd.read_csv(f"data/{city}_gdf_nodes.csv", index_col=0)

# define lts colours for plotting
conditions = [
    (all_lts['lts'] == 1),
    (all_lts['lts'] == 2),
    (all_lts['lts'] == 3),
    (all_lts['lts'] == 4),
    ]

# create a list of the values we want to assign for each condition
values = ['g', 'b', 'y', 'r']

# create a new column and use np.select to assign values to it using our lists as arguments
all_lts['color'] = np.select(conditions, values)

# %% Plot LTS
geo_df = all_lts
lats = []
lons = []
names = []

for feature, name in zip(geo_df.geometry, geo_df.name):
    if isinstance(feature, shapely.geometry.linestring.LineString):
        linestrings = [feature]
    elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
        linestrings = feature.geoms
    else:
        continue
    for linestring in linestrings:
        x, y = linestring.xy
        lats = np.append(lats, y)
        lons = np.append(lons, x)
        names = np.append(names, [name]*len(y))
        lats = np.append(lats, None)
        lons = np.append(lons, None)
        names = np.append(names, None)
center = {
        'lon': round((max([v for v in lons if v is not None]) +
                       min([v for v in lons if v is not None])) / 2, 6),
        'lat': round((max([v for v in lats if v is not None]) +
                       min([v for v in lats if v is not None])) / 2, 6)
        }
fig = px.line_geo(lat=lats, lon=lons, hover_name=names, 
                  scope='usa', center=center, fitbounds="locations")
fig.show()

all_lts[all_lts['lts'] > 0].plot(
    linewidth = 0.1, color = all_lts[all_lts['lts'] > 0]['color'])

fig, ax = plt.subplots()
all_lts[all_lts['lts'] > 0].plot(
    ax = ax, linewidth = 0.1, color = all_lts[all_lts['lts'] > 0]['color'])
# plt.savefig(f"plots/{city}_lts.pdf")
plt.savefig(f"plots/{city}_lts.png", dpi = 300)

# %% Plot segments that aren't missing speed and lane info

has_speed_lanes = all_lts[(~all_lts['maxspeed'].isna())
       & (~all_lts['lanes'].isna())]

fig, ax = plt.subplots(figsize = (8,8))
has_speed_lanes.plot(ax = ax, linewidth = 0.5, color = has_speed_lanes['color'])

ax.set_xlim(-79.43, -79.37)
ax.set_ylim(43.645, 43.675)

ax.set_yticks([])
ax.set_xticks([])

# plt.savefig(f"plots/LTS_{city}_has_speed_has_lanes.pdf")
plt.savefig(f"plots/LTS_{city}_has_speed_has_lanes.png", dpi = 300)
