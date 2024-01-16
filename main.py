'''
Run full workflow from here. Choose which city to run and all data 
will be downloaded, calculations performed, and stressmap plotted.
Intermidiary files will be saved to make subsequent runs faster.
Just delete the file you want to start from and everything after 
will be recreated.
'''

import build_query
import LTS_OSM
import LTS_plot

# Choose the city to perform analysis on. 
# [city name for documentaion, query key, query value]
# Query key and value can be determinined by inspecting regions
# on OpenStreetMaps

# city = ['Cambridge', 'wikidata', '1933745']
# city = ['Boston', 'wikidata', '2315704']
city = ['Somerville', 'wikipedia', 'en:Somerville, Massachusetts']
# city = ['Brookline', 'wikipedia', 'en:Brookline, Massachusetts']
# city = ['Medford', 'wikipedia', 'en:Medford, Massachusetts']

# Run LTS analysis and plotting
build_query.build_query(*city)
LTS_OSM.main(city[0])
LTS_plot.main(city[0])
