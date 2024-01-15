'''
Run full workflow from here
'''

import build_query
import LTS_OSM
import LTS_plot

city = ['Cambridge', 'wikidata', '1933745']
# city = ['Boston', 'wikidata', '2315704']

build_query.build_query(*city)
LTS_OSM.main(city[0])