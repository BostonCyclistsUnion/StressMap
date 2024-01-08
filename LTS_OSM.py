'''
# Level of Traffic Stress maps with Open Street Map

This notebook calculates Level of Traffic Stress from Open Street Map data. It uses 
the [osmnx](https://osmnx.readthedocs.io/en/stable/) Python package to download a street network.

### Setup

- download a JSON file from Open Street Map that roughly corresponds to the target area. 
    This is used only to get a list of tags to download with the `osmnx` package. The command 
  wget -nv -O victoriaosm.osm --post-file=victoriaosm.query "http://overpass-api.de/api/interpreter"
    downloads a file named victoriaosm.osm using the query in the file victoriaosm.query. This 
    procedure will be simplified in the future.
'''

import json
import os
import pandas as pd
import numpy as np
# import geopandas as gpd
import osmnx as ox
# from matplotlib import pyplot as plt
# import networkx as nx
from tqdm import tqdm
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib

# import lts calculation functions
from lts_functions import (biking_permitted, is_separated_path, is_bike_lane, parking_present,
                           bike_lane_analysis_no_parking, bike_lane_analysis_with_parking,
                           mixed_traffic)

# %% Extract OSM tags to use in download

# load the data
dataFile = 'Cambridge'
dataFolder = 'data'
wayTagsCSV = os.path.join(dataFolder, f'{dataFile}_way_tags.csv')

if os.path.exists(wayTagsCSV):
    way_tags_series = pd.read_csv(wayTagsCSV)
else:
    with open(os.path.join(dataFolder, f'{dataFile}.json'), 'r') as f:
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
    tag_value_counts = tags_df.value_counts().reset_index()
    # count all the unique tags
    tag_counts = tags_df['tag'].value_counts().reset_index()

    # explore the tags that start with 'cycleway'
    print(tag_counts[tag_counts['tag'].str.contains('cycleway')])

    way_tags_series = tag_counts['tag'] # all unique tags from the OSM download
    way_tags_series.to_csv(wayTagsCSV)

way_tags = list(way_tags_series) 

# add the above list to the global osmnx settings
ox.settings.useful_tags_way += way_tags
ox.settings.osm_xml_way_tags = way_tags

# %% Download data

# create a filter to download selected data
# this filter is based on osmfilter = ox.downloader._get_osm_filter("bike")
# keeping the footway and construction tags
osmfilter = '["highway"]["area"!~"yes"]["access"!~"private"]["highway"!~"abandoned|bus_guideway|corridor|elevator|escalator|motor|planned|platform|proposed|raceway|steps"]["bicycle"!~"no"]["service"!~"private"]'

# check if data has already been downloaded; if not, download
filepath = f"data/{dataFile}.graphml"
if os.path.exists(filepath):
    # load graph
    print("loading saved graph")
    G = ox.load_graphml(filepath)
else:
    # download data - this can be slow
    print("downloading data")
    G = ox.graph_from_place(
        f"{dataFile}, Massachusetts",
        retain_all=True,
        truncate_by_edge=True,
        simplify=False,
        # custom_filter=osmfilter2,
        custom_filter=osmfilter,
    )
    # save graph
    print("saving graph")
    ox.save_graphml(G, filepath)

# plot downloaded graph - this is slow for a large area
fig, ax = ox.plot_graph(G, node_size=0, edge_color="w", edge_linewidth=0.2)

# %% Analyze LTS
# Start with is biking allowed, get edges where biking is not *not* allowed.

# convert graph to node and edge GeoPandas GeoDataFrames
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

print(gdf_edges.shape)
gdf_allowed, gdf_not_allowed = biking_permitted(gdf_edges)
print(gdf_allowed.shape)
print(gdf_not_allowed.shape)

# check for separated path
separated_edges, unseparated_edges = is_separated_path(gdf_allowed)
# assign separated ways lts = 1
separated_edges['lts'] = 1
print(separated_edges.shape)
print(unseparated_edges.shape)

to_analyze, no_lane = is_bike_lane(unseparated_edges)
print(to_analyze.shape)
print(no_lane.shape)

parking_detected, parking_not_detected = parking_present(to_analyze)
print(parking_detected.shape)
print(parking_not_detected.shape)

parking_lts = bike_lane_analysis_with_parking(parking_detected)

no_parking_lts = bike_lane_analysis_no_parking(parking_not_detected)

# Next, go to the last step - mixed traffic

lts_no_lane = mixed_traffic(no_lane)

# final components: lts_no_lane, parking_lts, no_parking_lts, separated_edges
# these should all add up to gdf_allowed
print(gdf_allowed.shape)
print(lts_no_lane.shape[0] + parking_lts.shape[0] + \
      no_parking_lts.shape[0] + separated_edges.shape[0])

gdf_not_allowed['lts'] = 0

all_lts = pd.concat([separated_edges, parking_lts, no_parking_lts, lts_no_lane, gdf_not_allowed])

# decision rule glossary
# these are from Bike Ottawa's stressmodel code
# pylint: disable=line-too-long
rule_message_dict = {'p2':'Cycling not permitted due to bicycle=\'no\' tag.',
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

simplified_message_dict = {'p2':r'bicycle $=$ "no"',
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

all_lts['message'] = all_lts['rule'].map(rule_message_dict)
all_lts['short_message'] = all_lts['rule'].map(simplified_message_dict)

# %% Node LTS
'''
Calculate node LTS.

- An intersection without either was assigned the highest LTS of its intersecting roads.
- Stop signs reduced an otherwise LTS2 intersection to LTS1.
- A signalized intersection of two lowstress links was assigned LTS1.
- Assigned LTS2 to signalized intersections where a low-stress (LTS1/ 2) link crosses a high-stress (LTS3/4) link.
'''

gdf_nodes['highway'].value_counts()

gdf_nodes['lts'] = np.nan # make lts column
gdf_nodes['message'] = np.nan # make message column

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

# %% Save data for plotting

gdf_nodes.to_csv(f"data/{dataFile}_gdf_nodes.csv")

all_lts_small = all_lts[['osmid', 'lanes', 'name', 'highway', 'maxspeed', 'geometry', 'length', 'rule', 'lts',
                         'lanes_assumed', 'maxspeed_assumed', 'message', 'short_message']]
all_lts_small.to_csv(f"data/{dataFile}_all_lts.csv")

# make graph with LTS information
G_lts = ox.graph_from_gdfs(gdf_nodes, all_lts_small)

# save LTS graph
filepath = f"data/{dataFile}_lts.graphml"
ox.save_graphml(G_lts, filepath)
