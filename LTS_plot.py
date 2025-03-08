"""
Plotting Level of Traffic Stress

This notebook plots the Level of Traffic Stress map calculated in `LTS_OSM'.
"""
# import os
# import glob
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import geopandas as gpd
# import shapely.geometry
# import contextily as cx

# from matplotlib import pyplot as plt
# from matplotlib.lines import Line2D
# import plotly.express as px
# import lonboard
# from lonboard.colormap import apply_categorical_cmap

dataFolder = 'data'
queryFolder = 'query'
plotFolder = 'plots'

# create a list of the values we want to assign for each condition
ltsColors = ['grey', 'green', 'deepskyblue', 'orange', 'red']


def load_data(region):
    if type(region) is list:
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


def plot_lts_geojson(region, all_lts):
    lts = all_lts[all_lts['LTS'] > 0]

    # Worth about 1.5% of json size (don't know if more effective when getting to mapbox)
    # lts = lts.replace({'yes': 1, 'no': 0}).infer_objects(copy=False)
    # Worth about 4% of json size
    # lts = lts.replace({'Assumed': 'A'})

    fields_general = [
                    'geometry', 'LTS', 'osmid', 'name', 'highway',
                    'speed', 'speed_rule',
                    'centerline', 'centerline_rule',
                    'ADT', 'ADT_rule',
                    'lane_count', 'oneway', 
                    'street_narrow_wide', 
                    'width_street', 'width_street_rule',
                    # 'cycleway', 
                    'parse',                  
                    ]
    fields_dirs = [ 
                    'bike_allowed', 'bike_lane', 'separation',
                    'parking', 
                    # 'parking_rule',
                    'parking_width',
                    # 'parking_width_rule', 
                    'buffer', 'buffer_rule', 'bike_width', 'bike_width_rule', 'bike_reach',                      
                    'LTS_mixed', 'LTS_bikelane_noparking', 'LTS_bikelane_yesparking',
                    'LTS_bike_access', 'LTS',
                    ]
    
    fields_dirs = [field + '_fwd' for field in fields_dirs] + [field + '_rev' for field in fields_dirs]
    geo_json = lts[fields_general + fields_dirs].to_json()

    # Save GeoJson
    json_plot_file = f'{plotFolder}/{region}_LTS.json'
    with open(json_plot_file, 'w') as f:
        f.write(geo_json + '\n')

    shutil.copy(json_plot_file, f'{plotFolder}/LTS.json')
    return

def main(region, format="json"):
    Path(plotFolder).mkdir(exist_ok=True)

    all_lts = load_data(region)

    # plot_lts_static(region, all_lts)
    if format == "json":
        plot_lts_geojson(region, all_lts)


if __name__ == '__main__':
    # city = 'Cambridge'
    city = 'Boston'
    # city = 'Somerville'

    main(city)