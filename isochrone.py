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

city = "GreaterBoston"

gdf_nodes = pd.read_csv(f"data/{city}_6_gdf_nodes.csv", index_col=0, 
                      low_memory=False) # solve DtypeWarning columns having mixed types

all_lts = pd.read_csv(f"data/{city}_4_all_lts.csv", index_col=[0,1,2], 
                      low_memory=False) # solve DtypeWarning columns having mixed types

# load graph
filepath = f"data/{city}_7_lts.graphml"
G_lts = ox.load_graphml(filepath)

# whether to remove nodes by lts or not
remove_nodes = True

G1 = G_lts.copy()
G2 = G_lts.copy()
G3 = G_lts.copy()
G4 = G_lts.copy()

# +
# delete edges and nodes by lts level
G1.remove_edges_from(all_lts[all_lts['lts'] != 1].index)
G2.remove_edges_from(all_lts[(all_lts['lts'] > 2)
                            & (all_lts['lts'] == 0)].index)
G3.remove_edges_from(all_lts[(all_lts['lts'] > 3)
                            & (all_lts['lts'] == 0)].index)
G4.remove_edges_from(all_lts[(all_lts['lts'] == 0)].index)

if remove_nodes is True:
    G1.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] != 1].index)
    G2.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] > 2].index)
    G3.remove_nodes_from(gdf_nodes[gdf_nodes['lts'] > 3].index)

# +
# point to start isochrone plot from
y = 42.3732
x = -71.1108

center_node = ox.distance.nearest_nodes(G1, x ,y) # use the same starting node for each graph
G1b = ox.project_graph(G1) # this is slow - do we have to do this?
G2b = ox.project_graph(G2)
G3b = ox.project_graph(G3)
G4b = ox.project_graph(G4)
# -

iso_colors = ox.plot.get_colors(n=4, cmap='plasma', start=0, return_hex=True)

#trip_times = [10] 
travel_speed = 15 #biking speed in km/hour
trip_time = 15 #in minutes

# +
# add an edge attribute for time in minutes required to traverse each edge
meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
for u, v, k, data in G1b.edges(data=True, keys=True):
    data['time'] = data['length'] / meters_per_minute

for u, v, k, data in G2b.edges(data=True, keys=True):
    data['time'] = data['length'] / meters_per_minute
    
for u, v, k, data in G3b.edges(data=True, keys=True):
    data['time'] = data['length'] / meters_per_minute
    
for u, v, k, data in G4b.edges(data=True, keys=True):
    data['time'] = data['length'] / meters_per_minute

# +
node_colors = {}

# colour all nodes by LTS - go in descending order of LTS so that the lowest
# reachable level takes precedence

graphs = [G4b, G3b, G2b, G1b]

for i, G in enumerate(graphs):
    subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance='time')
    for node in subgraph.nodes():
        node_colors[node] = iso_colors[i]

nc = [node_colors[node] if node in node_colors else 'none' for node in G4b.nodes()]
ns = [5 if node in node_colors else 0 for node in G4b.nodes()]
# -

# get x and y in the correct projection
point = (x,y)
point_geom_proj, crs = ox.projection.project_geometry(
        Point(point), to_crs=G4b.graph['crs'])

G4b.graph['crs']

# +
# color the nodes according to isochrone then plot the street network
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
