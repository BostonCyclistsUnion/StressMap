#!/usr/bin/env python3
"""
Usage:
python build_query.py region key value
     
Example usage: 
python build_query.py eastyork wikidata Q167585
"""

import argparse
import os

def build_query(region, key, value):
    filepath = f'query/{region}.query'
    if os.path.exists(filepath):
        print(f"{region} query already exists")
    else:
        with open(filepath, 'w') as f:
            f.write('[timeout:600][out:json][maxsize:2000000000];\n')
            f.write(f'area["{key}"="{value}"];\n')
            f.write('out body;\n')
            f.write('(\n')
            f.write('way["highway"][footway!="sidewalk"][area];\n')
            f.write('way[footway="sidewalk"][bicycle][bicycle!="no"][bicycle!="dismount"][area];\n')
            f.write(')->.candidate_ways;\n')
            f.write('(\n')
            f.write('way.candidate_ways[service!="parking_aisle"];\n')
            f.write('node(w);\n')
            f.write(');\n')
            # f.write('((way["highway"][area]; - way[footway="sidewalk"][area];);\n')
            # f.write('  node(w)->.h;\n')
            # f.write('   (way[footway="sidewalk"][bicycle][bicycle!="no"][bicycle!="dismount"][area];);\n')
            # f.write('  node(w)->.s;\n')
            # f.write('  node.h.s;\n')
            # f.write(');\n')
            # f.write('out;\n')

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
