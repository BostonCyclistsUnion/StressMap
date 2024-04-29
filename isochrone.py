'''
Isochrone analysis

Plot isochrones from a starting location based on an LTS threshold.

Plotting code adapted from 
https://github.com/gboeing/osmnx-examples/blob/v0.13.0/notebooks/13-isolines-isochrones.ipynb

## Thoughts for making heatmap of isochrones
area of a polygon
    https://gis.stackexchange.com/questions/218450/getting-polygon-areas-using-geopandas
equally spaced points on a polygon
    https://stackoverflow.com/questions/66010964/fastest-way-to-produce-a-grid-of-points-that-fall-within-a-polygon-or-shape
    
'''


import geopandas as gpd
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point, LineString, Polygon
from shapely.prepared import prep
import alphashape
import cartopy.crs as ccrs

import LTS_OSM
import LTS_plot

# %% Settings
# whether to remove nodes by lts or not
remove_nodes = True

iso_colors = ox.plot.get_colors(n=4, cmap='plasma', start=0, return_hex=True)

# %% Load files
def load_files(city):
    # gdf_nodes = pd.read_csv(f"data/{city}_6_gdf_nodes.csv", index_col=0, 
    #                     low_memory=False) # solve DtypeWarning columns having mixed types
    
    gdf_nodes = LTS_OSM.read_gdf_nodes_csv(f"data/{city}_6_gdf_nodes.csv")

    # all_lts = pd.read_csv(f"data/{city}_4_all_lts.csv", index_col=[0,1,2], 
    #                     low_memory=False) # solve DtypeWarning columns having mixed types
    all_lts = LTS_plot.load_data(city)

    # load graph
    lts_graphml = ox.load_graphml(f"data/{city}_7_lts.graphml")

    return gdf_nodes, all_lts, lts_graphml

# %% Create Map-Graphs by LTS
def lts_map_graphs(G_lts, all_lts, gdf_nodes, remove_nodes=True):
    Glts = G_lts.copy()
    S = [Glts.subgraph(c).copy() for c in nx.weakly_connected_components(Glts)][0]

    G1 = S.copy()
    G2 = S.copy()
    G3 = S.copy()
    G4 = S.copy()

    lts1 = all_lts[(all_lts['lts'] > 1) | (all_lts['lts'] == 0)]
    lts2 = all_lts[(all_lts['lts'] > 2) | (all_lts['lts'] == 0)]
    lts3 = all_lts[(all_lts['lts'] > 3) | (all_lts['lts'] == 0)]
    lts4 = all_lts[all_lts['lts'] == 0]

    # delete edges and nodes by lts level
    G1.remove_edges_from(zip(lts1['u'].values, lts1['v'].values))
    G2.remove_edges_from(zip(lts2['u'].values, lts2['v'].values))
    G3.remove_edges_from(zip(lts3['u'].values, lts3['v'].values))
    G4.remove_edges_from(zip(lts4['u'].values, lts4['v'].values))

    if remove_nodes is True:
        G1.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] > 1]['osmid'])
        G2.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] > 2]['osmid'])
        G3.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] > 3]['osmid'])
        G1.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] == 0]['osmid'])
        G2.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] == 0]['osmid'])
        G3.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] == 0]['osmid'])
        G4.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] == 0]['osmid'])

    # Remove independent graphs
    G1 = [G1.subgraph(c).copy() for c in nx.weakly_connected_components(G4)][0]
    G2 = [G2.subgraph(c).copy() for c in nx.weakly_connected_components(G4)][0]
    G3 = [G3.subgraph(c).copy() for c in nx.weakly_connected_components(G4)][0]
    G4 = [G4.subgraph(c).copy() for c in nx.weakly_connected_components(G4)][0]
    # Remove isolates
    isolates = list(nx.isolates(G4))
    # print(f'{len(isolates)=}')
    for G in [G1, G2, G3, G4]:
        # print(G.number_of_nodes())
        G.remove_nodes_from(isolates)
        # print(G.number_of_nodes())
        # print()


    G1b = ox.project_graph(G1) # this is slow - do we have to do this?
    G2b = ox.project_graph(G2) # This might be better as subgraphs? Need to understand those better
    G3b = ox.project_graph(G3) # https://networkx.org/documentation/stable/reference/classes/generated/networkx.classes.graphviews.subgraph_view.html
    G4b = ox.project_graph(G4)

    return G1, G2, G3, G4, G1b, G2b, G3b, G4b


def edge_travel_times(travel_speed, G1b, G2b, G3b, G4b):
    # add an edge attribute for time in minutes required to traverse each edge
    meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
    for _, _, _, data in G1b.edges(data=True, keys=True): # for u, v, k, data
        data['time'] = data['length'] / meters_per_minute

    for _, _, _, data in G2b.edges(data=True, keys=True):
        data['time'] = data['length'] / meters_per_minute
        
    for _, _, _, data in G3b.edges(data=True, keys=True):
        data['time'] = data['length'] / meters_per_minute
        
    for _, _, _, data in G4b.edges(data=True, keys=True):
        data['time'] = data['length'] / meters_per_minute

    return G1b, G2b, G3b, G4b


def point_isochrone(nodeID, trip_time, G1b, G2b, G3b, G4b):
    # colour all nodes by LTS - go in descending order of LTS so that the lowest reachable level takes precedence
    graphs = [G4b, G3b, G2b, G1b]

    node_colors = {}
    node_count = []
    for i, G in enumerate(graphs):
        subgraph = nx.ego_graph(G, nodeID, radius=trip_time, distance='time')
        for node in subgraph.nodes():
            node_colors[node] = iso_colors[i]
        node_count.append(list(node_colors.values()).count(iso_colors[i]))
        print(f'LTS {4-i}: {node_count[-1]} nodes')

    return node_colors, node_count

def point_isochrone_plot(city, point, node_colors, trip_time, G4b):

    # get x and y in the correct projection
    point_geom_proj, crs = ox.projection.project_geometry(point, to_crs=G4b.graph['crs'])

    G4b.graph['crs']

    # color the nodes according to isochrone then plot the street network
    nc = [node_colors[node] if node in node_colors else 'none' for node in G4b.nodes()]
    ns = [5 if node in node_colors else 0 for node in G4b.nodes()]
    fig, ax = ox.plot_graph(G4b, node_color=nc, node_size=ns, node_alpha=0.4, node_zorder=0,
                            bgcolor='w', edge_linewidth=0.05, edge_color='#999999', 
                            figsize=(15, 15),
                            show=False, close=False)

    ax.scatter([point_geom_proj.x], [point_geom_proj.y], marker = '*', s = 50, color = 'k', zorder = 2)

    # add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)

    cmap = (matplotlib.colors.ListedColormap(iso_colors[::-1]).with_extremes(over='0.25', under='0.75'))

    bounds = np.arange(5)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax, orientation='vertical',
                label="LTS")
    # move label ticks to the centre
    labels = np.arange(1,5)
    loc = labels - 0.5
    cbar.set_ticks(loc)
    cbar.set_ticklabels(labels)

    plotPath = f'plots/{city}_isochrone_times_lts_remove_nodes_{remove_nodes}_time_{trip_time}'
    # plt.savefig(plotPath + '.pdf')
    plt.savefig(plotPath + '.png', dpi = 300)

def nearest_node(x, y, G):
    nodeID = ox.distance.nearest_nodes(G, x, y) # use the same starting node for each graph
    x = G.nodes[nodeID]['x']
    y = G.nodes[nodeID]['y']

    return Point(x, y), nodeID

def boundry_polygon(gdf_nodes, alpha=200):
    alpha_shape = alphashape.alphashape(gdf_nodes, alpha=alpha)

    return alpha_shape

def grid_points(alpha_shape, gridCount=25):
    

    alpha_polygon = alpha_shape.iloc[0,0]

    # determine maximum edges
    latmin, lonmin, latmax, lonmax = alpha_polygon.bounds
    latres = (latmax - latmin) / (gridCount + 1)
    lonres = (lonmax - lonmin) / (gridCount + 1)

    # create prepared polygon
    prep_polygon = prep(alpha_polygon)

    # construct a rectangular mesh
    points = []
    for lat in np.arange(latmin, latmax, latres):
        for lon in np.arange(lonmin, lonmax, lonres):
            points.append(Point((round(lat,4), round(lon,4))))

    # validate if each point falls inside shape using
    # the prepared polygon
    valid_points = list(filter(prep_polygon.contains, points))

    print(f'{len(valid_points)} grid points in boundary region.')

    return valid_points

def plot_grid_boundary(gdf_nodes, alpha_shape, valid_points):
     # Initialize plot
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot input points
    gdf_proj = gdf_nodes.to_crs(ccrs.Robinson().proj4_init)
    ax.scatter([p.x for p in gdf_proj['geometry']],
                [p.y for p in gdf_proj['geometry']],
                transform=ccrs.Robinson(),
                marker='.', s=1)

    ax.scatter([p.x for p in valid_points],
                [p.y for p in valid_points],
            #   transform=ccrs.Robinson(),
                marker='.', s=10, c='r')

    # Plot alpha shape
    ax.add_geometries(
        alpha_shape.to_crs(ccrs.Robinson().proj4_init)['geometry'],
        crs=ccrs.Robinson(), alpha=.2)

    plt.show()

# %% Run as script
def main(city, trip_time, travel_speed, x, y):
    # Load files and prep for calcs
    gdf_nodes, all_lts, G_lts = load_files(city)

    G1, G2, G3, G4, G1b, G2b, G3b, G4b = lts_map_graphs(G_lts, all_lts, gdf_nodes)

    # Bulk Calculations
    G1b, G2b, G3b, G4b = edge_travel_times(travel_speed, G1b, G2b, G3b, G4b)

    # Use the node closest to the given point
    point, nodeID = nearest_node(x, y, G1)

    node_colors, node_count = point_isochrone(nodeID, trip_time, G1b, G2b, G3b, G4b)

    point_isochrone_plot(city, point, node_colors, trip_time, G4b)

    # Create sampling points for heatmap
    alpha_shape = boundry_polygon(gdf_nodes, 200)
    valid_points = grid_points(alpha_shape, 25)
    plot_grid_boundary(gdf_nodes, alpha_shape, valid_points)

if __name__ == '__main__':
    # Settings
    city = "Cambridge"
    # city = "GreaterBoston"

    # point to start isochrone plot from
    y_init = 42.3732
    x_init = -71.1108
    
    travelSpeed = 15 #biking speed in km/hour
    tripTime = 15 # minutes

    main(city, tripTime, travelSpeed, x_init, y_init)
    