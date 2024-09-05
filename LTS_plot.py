"""
Plotting Level of Traffic Stress

This notebook plots the Level of Traffic Stress map calculated in `LTS_OSM'.
"""
import os
import glob
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import geopandas as gpd
import shapely.geometry
import contextily as cx

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import plotly.express as px
import lonboard
from lonboard.colormap import apply_categorical_cmap

dataFolder = 'data'
queryFolder = 'query'
plotFolder = 'plots'

# create a list of the values we want to assign for each condition
ltsColors = ['grey', 'green', 'deepskyblue', 'orange', 'red']


def load_data(region):
    if type(region) == list:
        df = pd.DataFrame()
        for r in region:
            dfr = pd.read_csv(f'{dataFolder}/{r}_4_all_lts.csv', low_memory=False)
            df = pd.concat([df, dfr])
            print(f'{dfr.shape=} | {df.shape=} | {r}')
    else:
        df = pd.read_csv(f'{dataFolder}/{region}_4_all_lts.csv', low_memory=False)

    # convert to a geodataframe for plotting
    geodf = gpd.GeoDataFrame(
        df.loc[:, [c for c in df.columns if c != "geometry"]],
        geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
        crs='wgs84') # projection from graph

    # gdf_nodes = pd.read_csv(f"{dataFolder}/{region}_6_gdf_nodes.csv", index_col=0)

    # define lts colours for plotting
    conditions = [
        (geodf['LTS'] == 0),
        (geodf['LTS'] == 1),
        (geodf['LTS'] == 2),
        (geodf['LTS'] == 3),
        (geodf['LTS'] == 4),
        ]

    # create a new column and use np.select to assign values to it using our lists as arguments
    geodf['color'] = np.select(conditions, ltsColors, default='grey')

    return geodf#, gdf_nodes


def plot_lts_plotly(region, all_lts):
    mapDataPath = f'data/{region}_plotly_data.csv'

    if os.path.exists(mapDataPath):
        mapData = pd.read_csv(mapDataPath)
        lats = mapData['lats']
        lons = mapData['lons']
        names = mapData['names']
        colors = mapData['colors']
        linegroup = mapData['linegroup']
        lts = mapData['LTS']
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
                    if row.LTS > 0:
                        x, y = linestring.xy
                        # Big speed improvement appending lists
                        lats.append(list(y))
                        lons.append(list(x))
                        names.append([row['name']]*len(y))
                        colors.append([row.color]*len(y))
                        linegroup.append([index]*len(y))
                        lts.append([f'LTS {row.LTS}']*len(y))
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
    linewidth = 0.4

    fig, ax = plt.subplots(figsize=(15,11))
    all_lts[all_lts['lts'] > 0].plot(
        ax = ax, linewidth = linewidth, color = all_lts[all_lts['lts'] > 0]['color'])

    ax.title.set_text(f'Level of Biking Traffic Stress Map for {region}')

    lw = 5
    legendLines = [Line2D([0], [0], color=ltsColors[1], lw=lw),
                   Line2D([0], [0], color=ltsColors[2], lw=lw),
                   Line2D([0], [0], color=ltsColors[3], lw=lw),
                   Line2D([0], [0], color=ltsColors[4], lw=lw),
                   ]
    ax.legend(legendLines, ['LTS 1', 'LTS 2', 'LTS 3', 'LTS 4'])

    cx.add_basemap(ax, crs=all_lts.crs, zoom=16,
                   source=cx.providers.CartoDB.DarkMatter)
    ax.set_axis_off()
    fig.tight_layout()

    # plt.savefig(f"{plotFolder}/{region}_lts.pdf")
    plt.savefig(f"{plotFolder}/{region}_lts.png", dpi = 300)
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


def plot_lts_lonboard_geojson(region, all_lts):
    lts = all_lts[all_lts['LTS'] > 0]
    geo_json = lts[['geometry', 'LTS', 'osmid', 'name', 'highway',
                    'biking_permitted', 'biking_permitted_rule',
                    'bike_lane_exist', 'bike_lane_exist_rule',
                    'bike_lane_separation', 'bike_lane_separation_rule',
                    'parking', 'parking_rule',
                    'speed', 'speed_rule',
                    'centerline', 'centerline_rule',
                    'ADT', 'ADT_rule',
                    'lane_count', 'oneway', 
                    'street_narrow_wide', 
                    'width_street', 'width_street_rule',
                    'width_parking', 'width_parking_rule', 
                    'width_bikelanebuffer', 'width_bikelanebuffer_rule',
                    'width_bikelane', 'width_bikelane_rule', 'bikelane_reach', 'cycleway',
                    'LTS_biking_permitted', 'LTS_bike_lane_separation', 
                    'LTS_mixed', 'LTS_bikelane_noparking', 'LTS_bikelane_yesparking',
                    ]].to_json()

    # Save GeoJson
    json_plot_file = f'{plotFolder}/{region}_LTS_lonboard.json'
    with open(json_plot_file, 'w') as f:
        f.write(geo_json + '\n')

    shutil.copy(json_plot_file, f'{plotFolder}/LTS.json')
    return


def plot_lts_lonboard(region, all_lts):
    lts = all_lts[all_lts['LTS'] > 0]

    layer = lonboard.PathLayer.from_geopandas(
        gdf=lts[["geometry", "LTS", "name"]], width_scale=2
    )
    layer.get_color = apply_categorical_cmap(
        values=lts["LTS"],
        cmap={
            0: [0, 0, 0],  # black
            1: [0, 128, 0],  # green
            2: [0, 191, 255],  # blue
            3: [255, 165, 0],  # orange
            4: [255, 0, 0],  # red
        },
    )

    # Save map to HTML
    plotFile = f'{plotFolder}/{region}_LTS_lonboard.html'
    lonboard.Map(layers=[layer],
                #  basemap_style=lonboard.basemap.CartoBasemap.Positron,
                 basemap_style=lonboard.basemap.CartoBasemap.DarkMatter,
                 _height=800,
                 ).to_html(plotFile)

    print(f'{plotFile} saved.')


def plot_all_regions():
    regionFiles = glob.glob(f'{dataFolder}/*_4_all_lts.csv')
    regions = [os.path.basename(f).split('_')[0] for f in regionFiles]

    all_lts = load_data(regions)

    # plot_lts_static('GreaterBoston', all_lts) # This is slow
    plot_lts_lonboard('GreaterBoston', all_lts)


def main(region, format="json"):
    Path(plotFolder).mkdir(exist_ok=True)

    all_lts = load_data(region)

    # plot_lts_static(region, all_lts)
    if format == "html":
        plot_lts_lonboard(region, all_lts)
    if format == "json":
        plot_lts_lonboard_geojson(region, all_lts)


if __name__ == '__main__':
    # city = 'Cambridge'
    city = 'Boston'
    # city = 'Somerville'

    main(city)

    # plot_all_regions()
