# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Plotting Level of Traffic Stress
#
# This notebook plots the Level of Traffic Stress map calculated in `LTS_OSM'.

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt

city = "Victoria"


all_lts_df = pd.read_csv("data/all_lts_%s.csv" %city)

# convert to a geodataframe for plotting
all_lts = gpd.GeoDataFrame(
    all_lts_df.loc[:, [c for c in all_lts_df.columns if c != "geometry"]],
    geometry=gpd.GeoSeries.from_wkt(all_lts_df["geometry"]), 
    crs='wgs84') # projection from graph

gdf_nodes = pd.read_csv("data/gdf_nodes_%s.csv" %city, index_col=0)

# +
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
# -

fig, ax = plt.subplots()
all_lts[all_lts['lts'] > 0].plot(ax = ax, linewidth = 0.1, color = all_lts[all_lts['lts'] > 0]['color'])
plt.savefig("lts_%s.pdf" %city)
plt.savefig("lts_%s.png" %city, dpi = 300)

# ## Plot segments that aren't missing speed and lane info

has_speed_lanes = all_lts[(~all_lts['maxspeed'].isna())
       & (~all_lts['lanes'].isna())]

# +
fig, ax = plt.subplots(figsize = (8,8))
has_speed_lanes.plot(ax = ax, linewidth = 0.5, color = has_speed_lanes['color'])

ax.set_xlim(-79.43, -79.37)
ax.set_ylim(43.645, 43.675)

ax.set_yticks([])
ax.set_xticks([])

plt.savefig("LTS_%s_has_speed_has_lanes.pdf" %city)
plt.savefig("LTS_%s_has_speed_has_lanes.png" %city, dpi = 300)
# -


