# LTS-OSM - Calculating Level of Traffic Stress using Open Street Map Data

This code calculates cycling [Level of Traffic Stress](https://peterfurth.sites.northeastern.edu/level-of-traffic-stress/) for all road and path segments in a region using data from Open Street Map.

## Background

This code is adapted from [Bike Ottawa's LTS code](https://github.com/BikeOttawa/stressmodel), modified to include Level of Traffic Stress for intersections by [Madeleine Bonsma-Fisher](https://github.com/mbonsma/LTS-OSM).

## Usage

1. From `main` you can change the city you want to perform the analysis on.
	- To get the right query details, inspect [openstreetmap.org](https://www.openstreetmap.org)
	for a `relation` of the region you want
	- Fill in this info in the city info
2. pip install requirements.txt, ideally in a virtual environment (venv)


## TBD:
Plot an isochrone map for different LTS thresholds with `isochrone.py`. This requires both the saved graph object and the dataframe with LTS levels calculated.
