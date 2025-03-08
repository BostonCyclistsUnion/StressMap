'''
# How to use:
Run main.py to have downloaded OSM data
Edit `CITY` as needed
Run this script then open http://localhost:8001/
Edit `config\filter_dict.yml` to change filters and refresh the web page


# Design Plan
- Run map server
- on reload, read yaml with filters
- run filter(s) on osm data (*_3.graphml)
    - use same function as full process
    - progressive filters, the first filter to catch something is what it is marked as
- plot filtered data on map
    - each filter is different color, legend is string of filter
    - simple popup, just name and osmid (w/link)
'''
import http.server
import socketserver
# import sys
import yaml
# import numpy as np
import osmnx as ox

PORT = 8001
PLOT = 'map/filter_test.html'
CITY = 'Boston'
dataFolder = 'data'

def read_filters():
    with open('config/filter_test.yml', 'r') as yml_file:
        filter_dict = yaml.safe_load(yml_file)
    print('loaded filters')
    return filter_dict

def apply_rules(gdf_edges, filter_dict):

    rules = {k:v for (k,v) in filter_dict.items()}
    gdf_edges['filterName'] = 'default'

    for key, value in rules.items():
        condition = value['condition']
        gdf_filter = gdf_edges.eval(f"{condition} & (`filterName` == 'default')")
        gdf_edges.loc[gdf_filter, 'filterName'] = key
        gdf_edges.loc[gdf_filter, 'filterValue'] = condition

    gdf_edges_filtered = gdf_edges[gdf_edges['filterName'] != 'default'].copy()

    return gdf_edges_filtered
    
def update_filter():
    filter_dict = read_filters()

    gdf_edges = GDF_EDGES.copy()

    print('Applying filters...')
    gdf_edges_filtered = apply_rules(gdf_edges, filter_dict)
    print(f'Filters showing {len(gdf_edges_filtered['osmid'].unique())} elements')
    
    print('Converting to json...')
    geo_json = gdf_edges_filtered.to_json()

    print('Saving json...')
    json_plot_file = 'plots/filters.json'
    with open(json_plot_file, 'w') as f:
        f.write(geo_json + '\n')

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if self.path == "/":
            return PLOT
        if self.path.startswith("/plots"):
            update_filter()
            return path[1:]  # strip the prefix slash so it can find the plots directory
        else:
            return http.server.SimpleHTTPRequestHandler.translate_path(self, path)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='/', **kwargs)
        

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print('Loading graphml data...')
    filepath = f"{dataFolder}/{CITY}_3.graphml"
    G = ox.load_graphml(filepath)
    _, GDF_EDGES = ox.graph_to_gdfs(G)

    print("serving at port", PORT)
    httpd.serve_forever()