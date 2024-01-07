#!/usr/bin/env python3
"""
Usage:
python build_query.py region key value
     
Example usage: 
python build_query.py eastyork wikidata Q167585
"""

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build OSM Overpass query file based on location')
    parser.add_argument('region', type=str, nargs='?',
                        help='Name of region (used for file saving)')
    parser.add_argument('key', type=str, default="wikidata", nargs='?',
                        help='OSM key for relation to download')
    parser.add_argument('value', type=str, nargs='?',
                        help='OSM value for relation to download')

    args = parser.parse_args()

    region = args.region
    key = args.key
    value = args.value

    # build query file

    with open(f'query\{region}.query', 'w') as f:
        f.write('[timeout:600][out:json][maxsize:2000000000];\narea\n')
        f.write(f'  ["{key}"="{value}"];\n')
        f.write('out body;\n')
        f.write('((way["highway"](area); - way[footway="sidewalk"](area););\n')
        f.write('  node(w)->.h;\n')
        f.write('   (way[footway="sidewalk"][bicycle](area); - way[footway="sidewalk"][bicycle="no"](area););\n')
        f.write('  node(w)->.s;\n')
        f.write('  node.h.s;\n')
        f.write(');\n')
        f.write('out;\n')
