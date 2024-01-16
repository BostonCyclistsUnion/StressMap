'''
Plotting Level of Traffic Stress

This notebook plots the Level of Traffic Stress map calculated in `LTS_OSM'.
'''
import os
import numpy as np
import pandas as pd

import geopandas as gpd
import shapely.geometry

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import plotly.express as px

dataFolder = 'data'
queryFolder = 'query'
plotFolder = 'plots'

# create a list of the values we want to assign for each condition
ltsColors = ['grey', 'green', 'blue', 'orange', 'red']

# %% Load and Prep Data
def load_data(region):
    all_lts_df = pd.read_csv(f"{dataFolder}/{region}_4_all_lts.csv", low_memory=False)

    # convert to a geodataframe for plotting
    all_lts = gpd.GeoDataFrame(
        all_lts_df.loc[:, [c for c in all_lts_df.columns if c != "geometry"]],
        geometry=gpd.GeoSeries.from_wkt(all_lts_df["geometry"]),
        crs='wgs84') # projection from graph

    # gdf_nodes = pd.read_csv(f"{dataFolder}/{region}_6_gdf_nodes.csv", index_col=0)

    # define lts colours for plotting
    conditions = [
        (all_lts['lts'] == 0),
        (all_lts['lts'] == 1),
        (all_lts['lts'] == 2),
        (all_lts['lts'] == 3),
        (all_lts['lts'] == 4),
        ]

    # create a new column and use np.select to assign values to it using our lists as arguments
    all_lts['color'] = np.select(conditions, ltsColors)

    return all_lts#, gdf_nodes

# %% Plot LTS
def plot_lts_plotly(region, all_lts):
    mapDataPath = f'data/{region}_plotly_data.csv'

    if os.path.exists(mapDataPath):
        mapData = pd.read_csv(mapDataPath)
        lats = mapData['lats']
        lons = mapData['lons']
        names = mapData['names']
        colors = mapData['colors']
        linegroup = mapData['linegroup']
        lts = mapData['lts']
        notes = mapData['notes']
    else:
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
        mapData.to_csv(mapDataPath)

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
                    title=f'Level of Biking Traffic Stress Map for {region}')
    fig.show()
    fig.write_html(f'{plotFolder}/{region}_stressmap.html')
    print(f'Saved {region}_stressmap.html')

def plot_lts_static(region, all_lts):
    all_lts[all_lts['lts'] > 0].plot(
        linewidth = 0.1, color = all_lts[all_lts['lts'] > 0]['color'])

    fig, ax = plt.subplots(figsize=(15,15))
    all_lts[all_lts['lts'] > 0].plot(
        ax = ax, linewidth = 0.1, color = all_lts[all_lts['lts'] > 0]['color'])

    ax.title.set_text(f'Level of Biking Traffic Stress Map for {region}')

    legendLines = [Line2D([0], [0], color=ltsColors[1], lw=4),
                   Line2D([0], [0], color=ltsColors[2], lw=4),
                   Line2D([0], [0], color=ltsColors[3], lw=4),
                   Line2D([0], [0], color=ltsColors[4], lw=4),
                   ]
    ax.legend(legendLines, ['LTS 1', 'LTS 2', 'LTS 3', 'LTS 4'])

    ax.set_axis_off()
    fig.tight_layout()

    # plt.savefig(f"{plotFolder}/{region}_lts.pdf")
    plt.savefig(f"{plotFolder}/{region}_lts.png", dpi = 600)
    fig.show()
    print(f'Saved {region}_lts.png')

def plot_not_missing_data(region, all_lts):
    # Plot segments that aren't missing speed and lane info
    has_speed_lanes = all_lts[(~all_lts['maxspeed'].isna())
        & (~all_lts['lanes'].isna())]

    fig, ax = plt.subplots(figsize = (8,8))
    has_speed_lanes.plot(ax = ax, linewidth = 0.5, color = has_speed_lanes['color'])

    ax.set_xlim(-79.43, -79.37)
    ax.set_ylim(43.645, 43.675)

    ax.set_yticks([])
    ax.set_xticks([])

    # plt.savefig(f"{plotFolder}/LTS_{region}_has_speed_has_lanes.pdf")
    plt.savefig(f"{plotFolder}/LTS_{region}_has_speed_has_lanes.png", dpi = 300)
    fig.show()
    print(f'Saved LTS_{region}_has_speed_has_lanes.png')

# %% Script
def main(region):
    all_lts = load_data(region)

    plot_lts_static(region, all_lts)
    # plot_lts_plotly(all_lts)

if __name__ == '__main__':
    city = 'Cambridge'
    # city = 'Boston'

    main(city)
