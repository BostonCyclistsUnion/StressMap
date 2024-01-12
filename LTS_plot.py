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
    (all_lts['lts'] == 0),
    ]

# create a list of the values we want to assign for each condition
# values = ['g', 'b', 'y', 'r']
values = ['green', 'blue', 'yellow', 'red', 'grey']

# create a new column and use np.select to assign values to it using our lists as arguments
all_lts['color'] = np.select(conditions, values)

# %% Plot LTS
# geo_df = all_lts
lats = []
lons = []
names = []
colors = []
linegroup = []
lts = []
notes = []

for index, row in all_lts.iterrows():
    feature = row.geometry
    if index < 1000000:
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            if row.lts > 0:
                x, y = linestring.xy
                # Big speed improvement appending lists
                lats.append(list(y))
                lons.append(list(x))
                names.append([row['name']]*len(y))
                colors.append([row.color]*len(y))
                linegroup.append([index]*len(y))
                lts.append([f'LTS {row.lts}']*len(y))
                notes.append([row.short_message]*len(y))

    else:
        break

lats = np.array(lats).flatten()
lons = np.array(lons).flatten()
names = list(np.array(names).flatten())
colors = list(np.array(colors).flatten())
linegroup = list(np.array(linegroup).flatten())
lts = list(np.array(lts).flatten())
notes = list(np.array(notes).flatten())

mapData = pd.DataFrame()
mapData['lats'] = lats
mapData['lons'] = lons
mapData['names'] = names
mapData['colors'] = colors
mapData['linegroup'] = linegroup
mapData['lts'] = lts
mapData['notes'] = notes
mapData.to_csv(f'data/{city}_mapData.csv')

center = {
        'lon': round((lons.max() + lons.min()) / 2, 6),
        'lat': round((lats.max() + lats.min()) / 2, 6)
        }

fig = px.line_geo(lat=lats, lon=lons,
                  hover_name=names, #hover_data=notes,
                  color=colors, color_discrete_map="identity",
                  line_group=linegroup,
                #   labels=lts,
                  scope='usa', center=center, fitbounds="locations",
                  title=f'Level of Biking Traffic Stress Map for {city}')
fig.show()
fig.write_html(f'plots/{city}_stressmap.html')

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
