# LTS-OSM - Calculating Level of Traffic Stress using Open Street Map Data

This code calculates cycling [Level of Traffic Stress](https://peterfurth.sites.northeastern.edu/level-of-traffic-stress/) for all road and path segments in a region using data from Open Street Map.

## Background

This code is adapted from [Bike Ottawa's LTS code](https://github.com/BikeOttawa/stressmodel), modified to include Level of Traffic Stress for intersections. 

## Usage

1. Generate an OSM query file for the region you want to study. 
	- Go to [openstreetmap.org](https://www.openstreetmap.org), search for the region you want to download (i.e. "Victoria"), then find the matching region in the list on the left side. 
	- Click on the region you want, which should open a page for a `relation` that says something like "Relation: Victoria (2221062)" at the top. 		- Scroll down in the list of tags until you find `wikidata`, i.e. `wikidata:Q2132` for Victoria, Canada. 
	- In a terminal, run `build_query.py` with this key-value pair: run `python build_query.py victoria wikidata Q2132` to generate a file `victoria.query`.
2. Download OSM data to get a list of tags. In a terminal, run `wget -nv -O victoria.osm --post-file=victoria.query "http://overpass-api.de/api/interpreter"`, changing the input and output filenames as needed.
3. Change the city name in the script `LTS_OSM.py`. Run `LTS_OSM.py` to download the network for the region and calculate LTS. This outputs a dataframe called `all_lts.csv` which contains all the ways, their assigned LTS, and the decision criteria that led to that LTS. In particular, the columns `lanes` and `maxspeed` may contain `NaN` if this information is not present in the OSM tags and the data can be filtered to exclude these.
4. Plot resulting LTS map with `LTS_plot.py`. 
5. Plot an isochrone map for different LTS thresholds with `isochrone.py`. This requires both the saved graph object and the dataframe with LTS levels calculated.

## Coming Soon

This code is under active development. Check out the [Issues](https://github.com/mbonsma/LTS-OSM/issues) for a list of improvements that will be made soon.
