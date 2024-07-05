'''
Level of Traffic Stress maps with Open Street Map

This calculates Level of Traffic Stress from Open Street Map data. It uses 
the [osmnx](https://osmnx.readthedocs.io/en/stable/) Python package to download a street network.

Each function that saves a file will check if it already exists. If a file does not exist, 
subsequent files will also be overwritten. This means to rerun from a given point, you can
just delete the file that is created at that stage. Files are numbered in the folder in order
of generation.
'''

import json
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
        # Start with is biking allowed, get edges where biking is not *not* allowed.
        gdf_allowed, gdf_not_allowed = lts.biking_permitted(gdf_edges)
        print(f'{gdf_allowed.shape=}')
        print(f'{gdf_not_allowed.shape=}')

        # check for separated path
        separated_edges, unseparated_edges = lts.is_separated_path(gdf_allowed)
        # assign separated ways lts = 1
        separated_edges['lts'] = 1
        print(f'{separated_edges.shape=}')
        print(f'{unseparated_edges.shape=}')

        to_analyze, no_lane = lts.is_bike_lane(unseparated_edges)
        print(f'{to_analyze.shape=}')
        print(f'{no_lane.shape=}')

        parking_detected, parking_not_detected = lts.parking_present(to_analyze)
        print(f'{parking_detected.shape=}')
        print(f'{parking_not_detected.shape=}')

        if parking_detected.shape[0] > 0:
            parking_lts = lts.bike_lane_analysis_with_parking(parking_detected)
        else:
            parking_lts = parking_detected

        if parking_not_detected.shape[0] > 0:
            no_parking_lts = lts.bike_lane_analysis_no_parking(parking_not_detected)
        else:
            no_parking_lts = parking_not_detected

        # Next, go to the last step - mixed traffic

        lts_no_lane = lts.mixed_traffic(no_lane)

        # final components: lts_no_lane, parking_lts, no_parking_lts, separated_edges
        # these should all add up to gdf_allowed
        components = lts_no_lane.shape[0] + parking_lts.shape[0] +\
                     no_parking_lts.shape[0] + separated_edges.shape[0]
        compareStr = (f'gdf_allowed = {gdf_allowed.shape[0]}\nComponents  = {components}\n'
                      f'\t{lts_no_lane.shape[0]=}\n\t{parking_lts.shape[0]=}\n'
                      f'\t{no_parking_lts.shape[0]=}\n\t{separated_edges.shape[0]=}'
                      )
        print(compareStr)

        gdf_not_allowed['lts'] = 0

        all_lts = pd.concat([separated_edges, parking_lts, no_parking_lts,
                            lts_no_lane, gdf_not_allowed])

        # decision rule glossary
        # these are from Bike Ottawa's stressmodel code
        # pylint: disable=line-too-long
        rule_message_dict = {'p1':'Cycling not permitted due to bicycle=\'dismount\' tag.',
                            'p2':'Cycling not permitted due to bicycle=\'no\' tag.',
                            'p6':'Cycling not permitted due to access=\'no\' tag.', 
                            'p3':'Cycling not permitted due to highway=\'motorway\' tag.',
                            'p4':'Cycling not permitted due to highway=\'motorway_link\' tag.', 
                            'p7':'Cycling not permitted due to highway=\'proposed\' tag.', 
                            'p5':'Cycling not permitted. When footway="sidewalk" is present, there must be a bicycle="yes" when the highway is "footway" or "path".', 
                            's3':'This way is a separated path because highway=\'cycleway\'.',
                            's1':'This way is a separated path because highway=\'path\'.', 
                            's2':'This way is a separated path because highway=\'footway\' but it is not a crossing.', 
                            's7':'This way is a separated path because cycleway* is defined as \'track\'.', 
                            's8':'This way is a separated path because cycleway* is defined as \'opposite_track\'.', 
                            'b1':'LTS is 1 because there is parking present, the maxspeed is less than or equal to 40, highway="residential", and there are 2 lanes or less.',
                            'b2':'Increasing LTS to 3 because there are 3 or more lanes and parking present.', 
                            'b3':'Increasing LTS to 3 because the bike lane width is less than 4.1m and parking present.', 
                            'b4':'Increasing LTS to 2 because the bike lane width is less than 4.25m and parking present.', 
                            'b5':'Increasing LTS to 2 because the bike lane width is less than 4.5m, maxspeed is less than 40 on a residential street and parking present.',
                            'b6':'Increasing LTS to 2 because the maxspeed is between 41-50 km/h and parking present.', 
                            'b7':'Increasing LTS to 3 because the maxspeed is between 51-54 km/h and parking present.', 
                            'b8':'Increasing LTS to 4 because the maxspeed is over 55 km/h and parking present.', 
                            'b9':'Increasing LTS to 3 because highway is not \'residential\'.', 
                            'c1':'LTS is 1 because there is no parking, maxspeed is less than or equal to 50, highway=\'residential\', and there are 2 lanes or less.',
                            'c3':'Increasing LTS to 3 because there are 3 or more lanes and no parking.',
                            'c4':'Increasing LTS to 2 because the bike lane width is less than 1.7 metres and no parking.', 
                            'c5':'Increasing LTS to 3 because the maxspeed is between 51-64 km/h and no parking.', 
                            'c6':'Increasing LTS to 4 because the maxspeed is over 65 km/h and no parking.', 
                            'c7':'Increasing LTS to 3 because highway with bike lane is not \'residential\' and no parking.', 
                            'm17':'Setting LTS to 1 because motor_vehicle=\'no\'.', 
                            'm13':'Setting LTS to 1 because highway=\'pedestrian\'.', 
                            'm14':'Setting LTS to 2 because highway=\'footway\' and footway=\'crossing\'.', 
                            'm2':'Setting LTS to 2 because highway=\'service\' and service=\'alley\'.', 
                            'm15':'Setting LTS to 2 because highway=\'track\'.', 
                            'm3':'Setting LTS to 2 because maxspeed is 50 km/h or less and service is \'parking_aisle\'.', 
                            'm4':'Setting LTS to 2 because maxspeed is 50 km/h or less and service is \'driveway\'.', 
                            'm16':'Setting LTS to 2 because maxspeed is less than 35 km/h and highway=\'service\'.', 
                            'm5':'Setting LTS to 2 because maxspeed is up to 40 km/h, 3 or fewer lanes and highway=\'residential\'.', 
                            'm6':'Setting LTS to 3 because maxspeed is up to 40 km/h and 3 or fewer lanes on non-residential highway.', 
                            'm7':'Setting LTS to 3 because maxspeed is up to 40 km/h and 4 or 5 lanes.', 
                            'm8':'Setting LTS to 4 because maxspeed is up to 40 km/h and the number of lanes is greater than 5.', 
                            'm9':'Setting LTS to 2 because maxspeed is up to 50 km/h and lanes are 2 or less and highway=\'residential\'.', 
                            'm10':'Setting LTS to 3 because maxspeed is up to 50 km/h and lanes are 3 or less on non-residential highway.', 
                            'm11':'Setting LTS to 4 because the number of lanes is greater than 3.', 
                            'm12':'Setting LTS to 4 because maxspeed is greater than 50 km/h.'}

        simplified_message_dict = {'p1':r'bicycle $=$ "dismount"',
                            'p2':r'bicycle $=$ "no"',
                            'p6':r'access $=$ "no"', 
                            'p3':r'highway $=$ "motorway"',
                            'p4':r'highway $=$ "motorway_link"', 
                            'p7':r'highway $=$ "proposed"', 
                            'p5':r'footway $=$ "sidewalk", bicycle$\neq$"yes"', 
                            's3':r'highway $=$ "cycleway"',
                            's1':r'highway $=$" path"', 
                            's2':r'separated, highway $=$" footway", not a crossing', 
                            's7':r'cycleway* $=$ "track"', 
                            's8':r'cycleway* $=$ "opposite_track"', 
                            'b1':r'bike lane w/ parking, $\leq$ 40 km/h, highway $=$ "residential", $\leq$ 2 lanes',
                            'b2':r'bike lane w/ parking, 3 or more lanes', 
                            'b3':r'bike lane width $<$ 4.1m, parking', 
                            'b4':r'bike lane width $<$ 4.25m, parking', 
                            'b5':r'bike lane width $<$ 4.5m, $\leq$ 40 km/h, residential, parking',
                            'b6':r'bike lane w/ parking, speed 41-50 km/h', 
                            'b7':r'bike lane w/ parking, speed 51-54 km/h', 
                            'b8':r'bike lane w/ parking, speed $>$ 55 km/h', 
                            'b9':r'bike lane w/ parking, highway $\neq$ "residential"', 
                            'c1':r'bike lane no parking, $\leq$ 50 km/h, highway $=$ "residential", $\leq$ 2 lanes',
                            'c3':r'bike lane no parking, $\leq$ 65 km/h, $\geq$ 3 lanes',
                            'c4':r'bike lane width $<$ 1.7m, no parking', 
                            'c5':r'bike lane no parking, speed 51-64 km/h', 
                            'c6':r'bike lane no parking, speed $>$ 65 km/h', 
                            'c7':r'bike lane no parking, highway $\neq$ "residential"', 
                            'm17':r'mixed traffic, motor_vehicle $=$ "no"', 
                            'm13':r'mixed traffic, highway $=$ "pedestrian"', 
                            'm14':r'mixed traffic, highway $=$ "footway", footway $=$ "crossing"', 
                            'm2':r'mixed traffic, highway $=$ "service", service $=$ "alley"', 
                            'm15':r'mixed traffic, highway $=$ "track"', 
                            'm3':r'mixed traffic, speed $\leq$ 50 km/h, service $=$ "parking_aisle"', 
                            'm4':r'mixed traffic, speed $\leq$ 50 km/h, service $=$ "driveway"', 
                            'm16':r'mixed traffic, speed $\leq$ 35 km/h, highway $=$ "service"', 
                            'm5':r'mixed traffic, speed $\leq$ 40 km/h, highway $=$ "residential", $\leq$ 3 lanes', 
                            'm6':r'mixed traffic, speed $\leq$ 40 km/h, highway $\neq$ "residential", $\leq$ 3 lanes', 
                            'm7':r'mixed traffic, speed $\leq$ 40 km/h, 4 or 5 lanes', 
                            'm8':r'mixed traffic, speed $\leq$ 40 km/h, lanes $>$ 5', 
                            'm9':r'mixed traffic, speed $\leq$ 50 km/h, highway $=$ "residential",$\leq$ 2 lanes', 
                            'm10':r'mixed traffic, speed $\leq$ 50 km/h, highway $\neq$ "residential", $\leq$ 3 lanes', 
                            'm11':r'mixed traffic, speed $\leq$ 50 km/h, lanes $>$ 3', 
                            'm12':r'mixed traffic, speed $>$ 50 km/h'}
        
        rating_dict = {
            # Biking Permitted Rules - biking_permitted()
            'p1':{
                'bicycle': 'dismount',
                'rule_message': 'Cycling not permitted due to bicycle=\'dismount\' tag.',
                'simple_message': r'bicycle $=$ "dismount"',
                },
            'p2':{
                'bicycle': 'no',
                'rule_message': 'Cycling not permitted due to bicycle=\'no\' tag.',
                'simple_message': r'bicycle $=$ "no"',
                },
            'p6':{
                'access': 'no',
                'rule_message': 'Cycling not permitted due to access=\'no\' tag.',
                'simple_message': r'access $=$ "no"',
                },
            'p3':{
                'highway': 'motorway',
                'rule_message': 'Cycling not permitted due to highway=\'motorway\' tag.',
                'simple_message': r'highway $=$ "motorway"',
                },
            'p4':{
                'highway': 'motorway_link',
                'rule_message': 'Cycling not permitted due to highway=\'motorway_link\' tag.',
                'simple_message': r'highway $=$ "motorway_link"',
                },
            'p7':{
                'highway': 'proposed',
                'rule_message': 'Cycling not permitted due to highway=\'proposed\' tag.',
                'simple_message': r'highway $=$ "proposed"',
                },
            'p5':{ # Multiple combos, should split?
                'rule_message': 'Cycling not permitted. When footway="sidewalk" is present, there must be a bicycle="yes" when the highway is "footway" or "path".',
                'simple_message': r'footway $=$ "sidewalk", bicycle$\neq$"yes"',
                },
            # Separated Paths - is_separated_path()
            's3':{
                'highway': 'cycleway',
                'rule_message': 'This way is a separated path because highway=\'cycleway\'.',
                'simple_message': r'highway $=$ "cycleway"',
                },
            's1':{
                'highway': 'path',
                'rule_message': 'This way is a separated path because highway=\'path\'.',
                'simple_message': r'highway $=$" path"',
                },
            's2':{ # 
                'rule_message': 'This way is a separated path because highway=\'footway\' but it is not a crossing.',
                'simple_message': r'separated, highway $=$" footway", not a crossing',
                },
            's7':{
                'rule_message': 'This way is a separated path because cycleway* is defined as \'track\'.',
                'simple_message': r'cycleway* $=$ "track"',
                },
            's8':{
                'rule_message': 'This way is a separated path because cycleway* is defined as \'opposite_track\'.',
                'simple_message': r'cycleway* $=$ "opposite_track"',
                },
            # Bike Lanes - is_bike_lane()
            'b1':{ # Fix Speed
                'rule_message': 'LTS is 1 because there is parking present, the maxspeed is less than or equal to 40, highway="residential", and there are 2 lanes or less.',
                'simple_message': r'bike lane w/ parking, $\leq$ 40 km/h, highway $=$ "residential", $\leq$ 2 lanes',
                },
            'b2':{
                'rule_message': 'Increasing LTS to 3 because there are 3 or more lanes and parking present.',
                'simple_message': r'bike lane w/ parking, 3 or more lanes',
                },
            'b3':{
                'rule_message': 'Increasing LTS to 3 because the bike lane width is less than 4.1m and parking present.',
                'simple_message': r'bike lane width $<$ 4.1m, parking',
                },
            'b4':{
                'rule_message': 'Increasing LTS to 2 because the bike lane width is less than 4.25m and parking present.',
                'simple_message': r'bike lane width $<$ 4.25m, parking',
                },
            'b5':{ # Fix Speed
                'rule_message': 'Increasing LTS to 2 because the bike lane width is less than 4.5m, maxspeed is less than 40 on a residential street and parking present.',
                'simple_message': r'bike lane width $<$ 4.5m, $\leq$ 40 km/h, residential, parking',
                },
            'b6':{ # Fix Speed
                'rule_message': 'Increasing LTS to 2 because the maxspeed is between 41-50 km/h and parking present.',
                'simple_message': r'bike lane w/ parking, speed 41-50 km/h',
                },
            'b7':{ # Fix Speed
                'rule_message': 'Increasing LTS to 3 because the maxspeed is between 51-54 km/h and parking present.',
                'simple_message': r'bike lane w/ parking, speed 51-54 km/h',
                },
            'b8':{ # Fix Speed
                'rule_message': 'Increasing LTS to 4 because the maxspeed is over 55 km/h and parking present.',
                'simple_message': r'bike lane w/ parking, speed $>$ 55 km/h',
                },
            'b9':{
                'rule_message': 'Increasing LTS to 3 because highway is not \'residential\'.',
                'simple_message': r'bike lane w/ parking, highway $\neq$ "residential"',
                },
            'c1':{ # Fix Speed
                'rule_message': 'LTS is 1 because there is no parking, maxspeed is less than or equal to 50, highway=\'residential\', and there are 2 lanes or less.',
                'simple_message': r'bike lane no parking, $\leq$ 50 km/h, highway $=$ "residential", $\leq$ 2 lanes',
                },
            'c3':{ # Fix Speed
                'rule_message': 'Increasing LTS to 3 because there are 3 or more lanes and no parking.',
                'simple_message': r'bike lane no parking, $\leq$ 65 km/h, $\geq$ 3 lanes',
                },
            'c4':{
                'rule_message': 'Increasing LTS to 2 because the bike lane width is less than 1.7 metres and no parking.',
                'simple_message': r'bike lane width $<$ 1.7m, no parking',
                },
            'c5':{ # Fix Speed
                'rule_message': 'Increasing LTS to 3 because the maxspeed is between 51-64 km/h and no parking.',
                'simple_message': r'bike lane no parking, speed 51-64 km/h',
                },
            'c6':{ # Fix Speed
                'rule_message': 'Increasing LTS to 4 because the maxspeed is over 65 km/h and no parking.',
                'simple_message': r'bike lane no parking, speed $>$ 65 km/h',
                },
            'c7':{
                'rule_message': 'Increasing LTS to 3 because highway with bike lane is not \'residential\' and no parking.',
                'simple_message': r'bike lane no parking, highway $\neq$ "residential"',
                },
            'm17':{
                'rule_message': 'Setting LTS to 1 because motor_vehicle=\'no\'.',
                'simple_message': r'mixed traffic, motor_vehicle $=$ "no"',
                },
            'm13':{
                'rule_message': 'Setting LTS to 1 because highway=\'pedestrian\'.',
                'simple_message': r'mixed traffic, highway $=$ "pedestrian"',
                },
            'm14':{
                'rule_message': 'Setting LTS to 2 because highway=\'footway\' and footway=\'crossing\'.',
                'simple_message': r'mixed traffic, highway $=$ "footway", footway $=$ "crossing"',
                },
            'm2':{
                'rule_message': 'Setting LTS to 2 because highway=\'service\' and service=\'alley\'.',
                'simple_message': r'mixed traffic, highway $=$ "service", service $=$ "alley"',
                },
            'm15':{
                'rule_message': 'Setting LTS to 2 because highway=\'track\'.',
                'simple_message': r'mixed traffic, highway $=$ "track"',
                },
            'm3':{ # Fix Speed
                'rule_message': 'Setting LTS to 2 because maxspeed is 50 km/h or less and service is \'parking_aisle\'.',
                'simple_message': r'mixed traffic, speed $\leq$ 50 km/h, service $=$ "parking_aisle"',
                },
            'm4':{ # Fix Speed
                'rule_message': 'Setting LTS to 2 because maxspeed is 50 km/h or less and service is \'driveway\'.',
                'simple_message': r'mixed traffic, speed $\leq$ 50 km/h, service $=$ "driveway"',
                },
            'm16':{ # Fix Speed
                'rule_message': 'Setting LTS to 2 because maxspeed is less than 35 km/h and highway=\'service\'.',
                'simple_message': r'mixed traffic, speed $\leq$ 35 km/h, highway $=$ "service"',
                },
            'm5':{ # Fix Speed
                'rule_message': 'Setting LTS to 2 because maxspeed is up to 40 km/h, 3 or fewer lanes and highway=\'residential\'.',
                'simple_message': r'mixed traffic, speed $\leq$ 40 km/h, highway $=$ "residential", $\leq$ 3 lanes',
                },
            'm6':{ # Fix Speed
                'rule_message': 'Setting LTS to 3 because maxspeed is up to 40 km/h and 3 or fewer lanes on non-residential highway.',
                'simple_message': r'mixed traffic, speed $\leq$ 40 km/h, highway $\neq$ "residential", $\leq$ 3 lanes',
                },
            'm7':{ # Fix Speed
                'rule_message': 'Setting LTS to 3 because maxspeed is up to 40 km/h and 4 or 5 lanes.',
                'simple_message': r'mixed traffic, speed $\leq$ 40 km/h, 4 or 5 lanes',
                },
            'm8':{ # Fix Speed
                'rule_message': 'Setting LTS to 4 because maxspeed is up to 40 km/h and the number of lanes is greater than 5.',
                'simple_message': r'mixed traffic, speed $\leq$ 40 km/h, lanes $>$ 5',
                },
            'm9':{ # Fix Speed
                'rule_message': 'Setting LTS to 2 because maxspeed is up to 50 km/h and lanes are 2 or less and highway=\'residential\'.',
                'simple_message': r'mixed traffic, speed $\leq$ 50 km/h, highway $=$ "residential",$\leq$ 2 lanes',
                },
            'm10':{ # Fix Speed
                'rule_message': 'Setting LTS to 3 because maxspeed is up to 50 km/h and lanes are 3 or less on non-residential highway.',
                'simple_message': r'mixed traffic, speed $\leq$ 50 km/h, highway $\neq$ "residential", $\leq$ 3 lanes',
                },
            'm11':{ # Fix Speed
                'rule_message': 'Setting LTS to 4 because the number of lanes is greater than 3.',
                'simple_message': r'mixed traffic, speed $\leq$ 50 km/h, lanes $>$ 3',
                },
            'm12':{ # Fix Speed
                'rule_message': 'Setting LTS to 4 because maxspeed is greater than 50 km/h.',
                'simple_message': r'mixed traffic, speed $>$ 50 km/h',
                },
            }

        all_lts['message'] = all_lts['rule'].map(rule_message_dict)
        all_lts['short_message'] = all_lts['rule'].map(simplified_message_dict)

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
