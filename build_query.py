#!/usr/bin/env python3
"""
Usage:
python build_query.py region key value
     
Example usage: 
python build_query.py eastyork wikidata Q167585
"""

import argparse
import os
from pathlib import Path

def build_query(region, key, value):
    filepath = Path('query') / (region + '.query')
    filepath.parent.mkdir(exist_ok=True)
    if filepath.exists():
        print(f"{region} query already exists")
    else:
        with filepath.open(mode='w') as f:
            f.write('[timeout:600][out:json][maxsize:2000000000];\n')
            f.write(f'area["{key}"="{value}"]->.search_area;\n')
            f.write('.search_area out body;\n')
            f.write("""
(
    way[highway][footway!=sidewalk][service!=parking_aisle](area.search_area);
    way[footway=sidewalk][bicycle][bicycle!=no][bicycle!=dismount](area.search_area);
    way[footway=traffic_island](area.search_area);
);
out;
            """)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build OSM Overpass query file based on location')
    parser.add_argument('region', type=str, nargs='?',
                        help='Name of region (used for file saving)')
    parser.add_argument('key', type=str, default="wikidata", nargs='?',
                        help='OSM key for relation to download')
    parser.add_argument('value', type=str, nargs='?',
                        help='OSM value for relation to download')

    args = parser.parse_args()

    regionIn = args.region
    keyIn = args.key
    valueIn = args.value

    # build query file
    build_query(regionIn, keyIn, valueIn)
