'''

https://towardsdatascience.com/loading-data-from-openstreetmap-with-python-and-the-overpass-api-513882a27fd0
'''

import requests
import json
import os

queryFile = 'Cambridge'
# queryFile = 'Boston'

overpass_url = "http://overpass-api.de/api/interpreter"

queryFolder = 'query'
queryFilepath = os.path.join(queryFolder, f'{queryFile}.query')
with open(queryFilepath, 'r') as f:
    lines = f.readlines()

overpass_query = ''.join(lines)
print(overpass_query)

response = requests.get(overpass_url,
                        params={'data': overpass_query},
                        timeout=60*5)
data = response.json()

dataFolder = 'data'
dataFilepath = os.path.join(dataFolder, f'{queryFile}.json')
print('Downloaded map data')

with open(dataFilepath, 'w') as f:
    json.dump(data, f)
    print('Saved map data')
