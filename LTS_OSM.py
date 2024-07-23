'''
Level of Traffic Stress maps with Open Street Map

This calculates Level of Traffic Stress from Open Street Map data. It uses 
the [osmnx](https://osmnx.readthedocs.io/en/stable/) Python package to download a street network.

Each function that saves a file will check if it already exists. If a file does not exist, 
subsequent files will also be overwritten. This means to rerun from a given point, you can
just delete the file that is created at that stage. Files are numbered in the folder in order
of generation.
'''

import json, yaml
import os
from pathlib import Path
from collections import defaultdict
import datetime

import requests

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


import geopandas as gpd
import osmnx as ox
import networkx as nx

# import matplotlib
# from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm

import lts_functions as lts

dataFolder = 'data'
queryFolder = 'query'

overpass_url = "http://overpass-api.de/api/interpreter"

OVERWRITE = False

# %% Functions
def check_files(region):
    # WIP: plan to better check what files exist and skip steps without unessesary file loads
    
    fileList = {
        'queryFile': Path('query') / (region + '.query'),
        'osmjsonFile': '',
        'waytagsFile': '',
        'graphFile': '',
        'allLtsFile': '',
        'allLtsSmallFile': '',
        'gdfNodeFile': '',
        'ltsGraphFile': '',
    }

    for file in fileList:
        pass
    


def build_query(region, key, value):
    global OVERWRITE
    filepath = Path('query') / (region + '.query')
    filepath.parent.mkdir(exist_ok=True)
    if filepath.exists():
        print(f"{region} query already exists")
    else:
        OVERWRITE = True
        with filepath.open(mode='w') as f:
            f.write('[timeout:600][out:json][maxsize:2000000000];\n')
            f.write(f'area["{key}"="{value}"]->.search_area;\n')
            f.write('.search_area out body;\n')
            f.write("""
(
    way[highway][footway!=sidewalk][service!=parking_aisle](area.search_area);
    way[footway=sidewalk][bicycle][bicycle!=no][bicycle!=dismount](area.search_area);
);
out;
            """)
        print(f'{filepath} created')

def download_osm(region):
    '''

    https://towardsdatascience.com/loading-data-from-openstreetmap-with-python-and-the-overpass-api-513882a27fd0
    '''
    global OVERWRITE
    queryFilepath = os.path.join(queryFolder, f'{region}.query')
    dataFilepath = os.path.join(dataFolder, f'{region}_1.json')

    if os.path.exists(dataFilepath) and (OVERWRITE is False):
        print(f'OSM data already downloaded for {region}')
    else:
        OVERWRITE = True
        with open(queryFilepath, 'r') as f:
            lines = f.readlines()
        overpass_query = ''.join(lines)
        # print(overpass_query)

        print(f'Downloaing OSM map data for {region}...')
        response = requests.get(overpass_url,
                                params={'data': overpass_query},
                                timeout=60*5)
        data = response.json()

        print(f'\tDownloaded OSM map data for {region}')

        with open(dataFilepath, 'w') as f:
            json.dump(data, f)
            print(f'Saved {region} map data')

def extract_tags(region):
    '''
    Extract OSM tags to use in download
    '''
    global OVERWRITE
    # load the data
    wayTagsCSV = os.path.join(dataFolder, f'{region}_2_way_tags.csv')

    if os.path.exists(wayTagsCSV) and (OVERWRITE is False):
        way_tags_series = pd.read_csv(wayTagsCSV, index_col=0)['tag']
        print(f'Read {wayTagsCSV}')
    else:
        OVERWRITE = True
        print(f'Finding way tags for {region}...')
        with open(os.path.join(dataFolder, f'{region}_1.json'), 'r') as f:
            data = json.load(f)

        # make a dataframe of tags
        dfs = []

        for element in data['elements']:
            if element['type'] != 'way':
                continue
            df = pd.DataFrame.from_dict(element['tags'], orient = 'index')
            dfs.append(df)

        tags_df = pd.concat(dfs).reset_index()
        tags_df.columns = ["tag", "tagvalue"]

        # count all the unique tag and value combinations
        # tag_value_counts = tags_df.value_counts().reset_index()
        # count all the unique tags
        tag_counts = tags_df['tag'].value_counts().reset_index()

        # explore the tags that start with 'cycleway'
        print(f"Cycleway tags:\n{tag_counts[tag_counts['tag'].str.contains('cycleway')]}")

        way_tags_series = tag_counts['tag'] # all unique tags from the OSM download
        way_tags_series.to_csv(wayTagsCSV)
        print(f'\t{wayTagsCSV} saved.')

    way_tags = list(way_tags_series)

    # add the above list to the global osmnx settings
    ox.settings.useful_tags_way += way_tags
    ox.settings.osm_xml_way_tags = way_tags
    print('Way tags added to osmnx settings.')


def download_data(region):
    '''
    Download data for a given region
    '''
    global OVERWRITE
    # create a filter to download selected data
    # this filter is based on osmfilter = ox.downloader._get_osm_filter("bike")
    # keeping the footway and construction tags
    osmfilter = ('["highway"]["area"!~"yes"]["access"!~"private"]'
                '["highway"!~"abandoned|bus_guideway|corridor|elevator|escalator|motor|'
                'planned|platform|proposed|raceway|steps"]'
                '["bicycle"!~"no"]["service"!~"private"]'
                '["indoor"!~"yes"]'
                '["service"!="parking_aisle"]')

    # check if data has already been downloaded; if not, download
    filepath = f"{dataFolder}/{region}_3.graphml"
    if os.path.exists(filepath) and (OVERWRITE is False):
        # load graph
        print(f"Loading saved graph for {region}")
        G = ox.load_graphml(filepath)
    else:
        OVERWRITE = True
        print(f"Downloading {region} data (this may take some time)...")
        G = ox.graph_from_place(
            f"{region}, Massachusetts",
            retain_all=True,
            truncate_by_edge=True,
            simplify=False,
            custom_filter=osmfilter,
        )
        print(f"Saving {region} graph")
        ox.save_graphml(G, filepath)

        # plot downloaded graph - this is slow for a large area
        # fig, ax = ox.plot_graph(G, node_size=0, edge_color="w", edge_linewidth=0.2)
        # ox.plot_graph(G, node_size=0, edge_color="w", edge_linewidth=0.2)

    # convert graph to node and edge GeoPandas GeoDataFrames
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    print(f'{gdf_edges.shape=}')
    print(f'{gdf_nodes.shape=}')

    return gdf_nodes, gdf_edges

def read_lts_csv(filepath):

    loadCols = ['u','v','key', 'osmid', 'geometry', 'access_aisle', 'access:conditional', 
            'access:disabled', 'access', 'aeroway', 'alt_name', 'area:highway', 
            'area', 'barrier', 'bicycle', 'bridge:movable', 'bridge:name', 
            'bridge', 'bus:conditional', 'bus:lanes:conditional', 'bus', 
            'busway:left', 'busway:right', 'busway', 'change:lanes:forward', 
            'change', 'class:bicycle', 'construction', 'covered', 'crossing_ref', 
            'crossing:island', 'crossing:markings', 'crossing:signals', 'crossing', 
            'cycleway:both:buffer', 'cycleway:both:lane', 'cycleway:both', 
            'cycleway:buffer', 'cycleway:lane', 'cycleway:left:buffer', 
            'cycleway:left:lane', 'cycleway:left:oneway', 'cycleway:left', 
            'cycleway:right:buffer', 'cycleway:right:lane', 'cycleway:right:oneway', 
            'cycleway:right', 'cycleway:surface', 'cycleway', 'description', 
            'designated_direction', 'designation', 'direction', 'disused', 
            'embedded_rails', 'emergency', 'entrance', 'exit', 'expressway', 
            'fee', 'flashing_lights', 'floating', 'foot', 'footway:surface', 
            'footway', 'highway:conditional', 'highway', 'incline', 'indoor', 
            'informal', 'junction', 'kerb', 'landing', 'lane_markings', 
            'lanes:backward', 'lanes:bus:backward', 'lanes:bus:forward',
            'lanes:conditional', 'lanes:forward', 'lanes', 'layer', 'level', 
            'light_rail', 'location', 'man_made', 'material', 'maxlength', 
            'maxspeed:advisory', 'maxspeed:bus', 'maxspeed:hgv', 'maxspeed:type', 
            'maxspeed:variable', 'maxspeed', 'motor_vehicle:conditional', 
            'motor_vehicle', 'motorcar', 'mtb:scale', 'name:en', 'name', 
            'natural', 'noexit', 'noname', 'official_name', 'oneway:bicycle', 
            'oneway:bus', 'oneway:conditional', 'oneway', 'opening_date', 
            'parking:both:orientation', 'parking:both', 'parking:condition:both:customers', 
            'parking:condition:both:maxstay', 'parking:condition:both:time_interval', 
            'parking:condition:both', 'parking:condition:left:maxstay', 
            'parking:condition:left:time_interval', 'parking:condition:left', 
            'parking:condition:right:maxstay', 'parking:condition:right:time_interval', 
            'parking:condition:right', 'parking:lane:both_1', 'parking:lane:both:parallel', 
            'parking:lane:both', 'parking:lane:left:parallel', 'parking:lane:left', 
            'parking:lane:right:parallel', 'parking:lane:right', 'parking:lane', 
            'parking:left:orientation', 'parking:left', 'parking:right:both', 
            'parking:right:orientation', 'parking:right', 'place', 'placement', 
            'protected', 'psv', 'public_transport', 'railway', 'ramp:bicycle', 
            'ramp:wheelchair', 'ramp', 'ruined', 'sac_scale', 'segregated', 
            'service', 'short_name', 'shoulder:right', 'shoulder', 'sidewalk:both:surface', 
            'sidewalk:both', 'sidewalk:left', 'sidewalk:right:surface', 
            'sidewalk:right', 'sidewalk', 'signal', 'stairs', 'start_date', 
            'step_count', 'subway', 'surface', 'tracktype', 'traffic_calming', 
            'traffic_island', 'traffic_signals:countdown', 'traffic_signals:sound', 
            'traffic_signals:vibration', 'traffic_signals', 'trail_visibility', 
            'trolley_wire', 'trolleybus', 'tunnel', 'turn:lanes:backward', 
            'turn:lanes:conditional', 'turn:lanes:forward', 'turn:lanes', 
            'turn', 'vehicle', 'was:bridge:movable', 'width:feet', 'width',
            ]
    
    dtypeDict = {'u': 'Int64',
                 'v': 'Int64',
                 'key': 'Int32',
                #  'level': 'float32',
                 'level': 'object',
                 'osmid': 'Int64',
                #  'lanes': 'Int32',
                #  'lanes:forward': 'Int32',
                #  'lanes:backward': 'Int32',
                 'lanes': 'object',
                 'lanes:forward': 'object',
                 'lanes:backward': 'object',
                 'layer': 'Int32',
                 'oneway': 'bool',
                 'geometry': 'object',
                }
    
    dtypes = defaultdict(CategoricalDtype, dtypeDict)
    df = pd.read_csv(filepath, usecols=lambda x: x in loadCols, 
                    dtype=dtypes, 
                    keep_default_na=True, na_values="''",
                    low_memory=False)

    # convert to a geodataframe for plotting
    geodf = gpd.GeoDataFrame(
        df.loc[:, [c for c in df.columns if c != "geometry"]],
        geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
        crs='wgs84') # projection from graph

    # Make some geo dataframes have the right index
    geoIndex = ['u','v','key']
    if set(geoIndex).issubset(geodf.columns):
        geodf.set_index(geoIndex, inplace=True)

    return geodf

def read_gdf_nodes_csv(filepath):

    dtypeDict = {'x': 'float64',
                 'y': 'float64',
                 'osmid': 'Int64',
                 'street_count': 'Int32',
                 'highway': 'category',
                 'ref': 'category',
                 'geometry': 'object',
                 'lts': 'Int32',
                 'message': 'category',                 
                }
    
    df = pd.read_csv(filepath, dtype=dtypeDict, 
                     keep_default_na=True, na_values="''",
                     low_memory=False)

    # convert to a geodataframe for plotting
    geodf = gpd.GeoDataFrame(
        df.loc[:, [c for c in df.columns if c != "geometry"]],
        geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
        crs='wgs84') # projection from graph

    return geodf

def lts_edges(region, gdf_edges):
    '''
    Calculate the LTS for all edges
    '''
    global OVERWRITE
    filepathAll = f"{dataFolder}/{region}_4_all_lts.csv"
    filepathSmall = f"{dataFolder}/{region}_5_all_lts_small.csv"

    if os.path.exists(filepathAll) and (OVERWRITE is False):
        # load graph
        print(f"Loading LTS for {region}")
        all_lts = read_lts_csv(filepathAll)
    else:
        OVERWRITE = True

        # Load the configuration files to caluclate ratings
        rating_dict = lts.read_rating()
        tables = lts.read_tables()

        # Start with is biking allowed, get edges where biking is not *not* allowed.
        gdf_edges = lts.biking_permitted(gdf_edges, rating_dict)
        gdf_edges = lts.is_separated_path(gdf_edges, rating_dict)
        gdf_edges = lts.is_bike_lane(gdf_edges, rating_dict)
        gdf_edges = lts.parking_present(gdf_edges, rating_dict)
        gdf_edges = lts.get_prevailing_speed(gdf_edges, rating_dict)
        gdf_edges = lts.get_lanes(gdf_edges, default_lanes = 2)
        gdf_edges = lts.get_centerlines(gdf_edges, rating_dict)
        gdf_edges = lts.width_ft(gdf_edges)
        gdf_edges = lts.define_narrow_wide(gdf_edges)
        gdf_edges = lts.define_adt(gdf_edges, rating_dict)

        all_lts = lts.calculate_lts(gdf_edges, tables)
        
        # print(f'Saving LTS for {region}')
        all_lts.to_csv(filepathAll)
        # https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_file.html

    if os.path.exists(filepathSmall) & os.path.exists(filepathAll) & (OVERWRITE is False):
        # load graph
        print(f"Loading LTS_small for {region}")
        all_lts_small = read_lts_csv(filepathSmall)
    else:
        OVERWRITE = True
        all_lts_small = all_lts[['osmid', 'lanes', 'name', 'highway', 'maxspeed', 'geometry',
                                'length', 'rule', 'lts', 'lanes_assumed', 'maxspeed_assumed',
                                'message', 'short_message']]
        print(f'Saving LTS_small for {region}')
        all_lts_small.to_csv(filepathSmall)

    return all_lts, all_lts_small


def lts_nodes(region, gdf_nodes, all_lts):
    '''
    Calculate node LTS.

    - An intersection without either was assigned the highest LTS of its intersecting roads.
    - Stop signs reduced an otherwise LTS2 intersection to LTS1.
    - A signalized intersection of two lowstress links was assigned LTS1.
    - Assigned LTS2 to signalized intersections where a low-stress (LTS1/ 2) link crosses a 
        high-stress (LTS3/4) link.
    '''
    global OVERWRITE
    filepath = f"{dataFolder}/{region}_6_gdf_nodes.csv"

    if os.path.exists(filepath) & (OVERWRITE is False):
        print(f'Loading {filepath}')
        gdf_nodes = read_gdf_nodes_csv(filepath)
        gdf_nodes.set_index('osmid', inplace=True)

    else:
        OVERWRITE = True
        gdf_nodes['highway'].value_counts()

        gdf_nodes['lts'] = np.nan # make lts column
        gdf_nodes['message'] = '' # make message column

        for node in tqdm(gdf_nodes.index):
            # pylint: disable=bare-except
            try:
                edges = all_lts.loc[node]
            except:
                #print("Node not found in edges: %s" %node)
                gdf_nodes.loc[node, 'message'] = "Node not found in edges"
                continue
            # pylint: enable=bare-except
            control = gdf_nodes.loc[node,'highway'] # if there is a traffic control
            max_lts = edges['lts'].max()
            node_lts = int(max_lts) # set to max of intersecting roads
            message = "Node LTS is max intersecting LTS"
            if node_lts > 2:
                if control == 'traffic_signals':
                    node_lts = 2
                    message = "LTS 3-4 with traffic signals"
            elif node_lts <= 2:
                if control == 'traffic_signals' or control == 'stop':
                    node_lts = 1
                    message = "LTS 1-2 with traffic signals or stop"

            gdf_nodes.loc[node,'message'] = message
            gdf_nodes.loc[node,'lts'] = node_lts # assign node lts

        gdf_nodes.to_csv(filepath)
        print(f'Saved LTS nodes for {region}')

    return gdf_nodes


def save_LTS_graph(region, all_lts_small, gdf_nodes):
    '''
    Save LTS graph for plotting
    '''
    global OVERWRITE
    filepath = f'{dataFolder}/{region}_7_lts.graphml'

    if os.path.exists(filepath) & (OVERWRITE is False):
        print(f'{region} LTS graph already exists')
    else:
        OVERWRITE = True
        # make graph with LTS information
        G_lts = ox.graph_from_gdfs(gdf_nodes, all_lts_small)

        # save LTS graph
        print(f'Saving {region} LTS graph')
        ox.save_graphml(G_lts, filepath)

def combine_data(fullRegion, regionList):

    def combine_all_lts(fullRegion, regionList):
        print('All LTS - 4')
        allLTSpathCombined = f'{dataFolder}/{fullRegion}_4_all_lts.csv'
        allLTS = pd.DataFrame()
        for region in regionList:
            print(f'\t{region}')
            allLTSpath = f'{dataFolder}/{region}_4_all_lts.csv'
            allLTS = pd.concat([allLTS, read_lts_csv(allLTSpath)])
        allLTS.to_csv(allLTSpathCombined)

    def combine_gdf_nodes(fullRegion, regionList):
        print('GDF Nodes - 6')
        gdfNodesPathCombined = f'{dataFolder}/{fullRegion}_6_gdf_nodes.csv'
        gdfNodes = pd.DataFrame()
        for region in regionList:
            print(f'\t{region}')
            gdfNodesPath = f'{dataFolder}/{region}_6_gdf_nodes.csv'
            gdfNodes = pd.concat([gdfNodes, pd.read_csv(gdfNodesPath, index_col=0)])
        gdfNodes.to_csv(gdfNodesPathCombined)

    def combine_lts_graph(fullRegion, regionList):
        print('LTS Graph - 7')
        print(datetime.datetime.now())
        graphPathCombined = f'{dataFolder}/{fullRegion}_7_lts.graphml'
        G_lts = []
        for region in regionList:
            print(f'\t{region}')
            graphPath = f'{dataFolder}/{region}_7_lts.graphml'
            G_lts.append(ox.load_graphml(graphPath))
        G_lts_all = nx.compose_all(G_lts)
        ox.save_graphml(G_lts_all, graphPathCombined)
        print(datetime.datetime.now())

    combine_all_lts(fullRegion, regionList)
    combine_gdf_nodes(fullRegion, regionList)
    combine_lts_graph(fullRegion, regionList)


# %% Run as Script
def main(region, key, value, rebuild=False):
    global OVERWRITE
    OVERWRITE = rebuild
    Path(dataFolder).mkdir(exist_ok=True)

    build_query(region, key, value)
    download_osm(region)
    extract_tags(region)
    gdfNodes, gdfEdges = download_data(region)
    all_lts, all_lts_small = lts_edges(region, gdfEdges)
    gdf_nodes = lts_nodes(region, gdfNodes, all_lts)
    save_LTS_graph(region, all_lts_small, gdf_nodes)

if __name__ == '__main__':
    # city = ['Cambridge', 'wikipedia', 'en:Cambridge, Massachusetts']
    city = ['Boston', 'wikipedia', 'en:Boston,']

    main(*city, True)
