'''
Run full workflow from here. Choose which city to run and all data 
will be downloaded, calculations performed, and stressmap plotted.
Intermidiary files will be saved to make subsequent runs faster.
Just delete the file you want to start from and everything after 
will be recreated.
'''

import LTS_OSM
import LTS_plot

# Query key and value can be determinined by inspecting regions
# on OpenStreetMaps

cities = {
       'Arlington':
              {'key': 'wikipedia',
              'value': 'en:Arlington, Massachusetts'},
       'Belmont':
              {'key': 'wikipedia',
              'value': 'en:Belmont, Massachusetts'},
       'Boston':
              {'key': 'wikipedia',
              'value': 'en:Boston'},
       'Brookline':
              {'key': 'wikipedia',
              'value': 'en:Brookline, Massachusetts'},
       'Cambridge':
              {'key': 'wikipedia',
              'value': 'en:Cambridge, Massachusetts'},
       'Chelsea':
              {'key': 'wikipedia',
              'value': 'en:Chelsea, Massachusetts'},
       'Everett':
              {'key': 'wikipedia',
              'value': 'en:Everett, Massachusetts'},
       'Malden':
              {'key': 'wikipedia',
              'value': 'en:Malden, Massachusetts'},
       'Medford':
              {'key': 'wikipedia',
              'value': 'en:Medford, Massachusetts'},
       'Newton':
              {'key': 'wikipedia',
              'value': 'en:Newton, Massachusetts'},
       'Lexington':
              {'key': 'wikipedia',
              'value': 'en:Lexington, Massachusetts'},
       'Somerville':
              {'key': 'wikipedia',
              'value': 'en:Somerville, Massachusetts'},
       'Waltham':
              {'key': 'wikipedia',
              'value': 'en:Waltham, Massachusetts'},
       'Watertown':
              {'key': 'wikipedia',
              'value': 'en:Watertown, Massachusetts'},
       }

city = 'Cambridge'
# city = 'Boston'
# city = 'Somerville'

# Run LTS analysis and plotting
rebuild = False

LTS_OSM.main(city, cities[city]['key'], cities[city]['value'], rebuild)
LTS_plot.main(city)

# for city in cities:
#     LTS_OSM.main(city, cities[city]['key'], cities[city]['value'], rebuild)
#     LTS_plot.main(city)

# Create a combined map from all cities analyzed
# LTS_OSM.combine_data('GreaterBoston', list(cities.keys()))
# LTS_plot.plot_all_regions()
