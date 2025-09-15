from typing import Union
from geojson import FeatureCollection, Feature, LineString
import sqlite3
from sqlite3 import Cursor, Connection
import argparse
import json

cycleway_columns = [
    "wayOsmId",
    "cyclewayType",
    "cyclewayLane",
    "cyclewaySurface",
    "leftType",
    "leftLane",
    "leftWidth",
    "leftBuffer",
    "leftSeparation",
    "leftReversed",
    "rightType",
    "rightLane",
    "rightWidth",
    "rightBuffer",
    "rightSeparation",
    "rightReversed",
]

node_columns = [
    "osmId",
    "latitude",
    "longitude",
    "highway",
    "trafficCalming",
    "crossing",
    "crossingMarkings",
    "crossingIsland",
]


def to_camel_case(text):
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0].lower() + "".join(i.capitalize() for i in s[1:])


class WaySegment:
    def __init__(self, osm_id: int, node_subset: Union[list[int], None]):
        self.osm_id = osm_id
        self.node_subset = node_subset

    def to_geojson_feature(self, cursor: Cursor):
        way_record = cursor.execute(
            f"""
                SELECT W.*, LTS.LTS
                FROM WAY W
                    LEFT JOIN LEVEL_OF_TRAFFIC_STRESS LTS ON W.OSM_ID = LTS.WAY_OSM_ID
                WHERE OSM_ID = {self.osm_id}
            """
        ).fetchone()
        if not way_record:
            raise ValueError(f"Way [{self.osm_id} does not exist in the LTS database]")
        
        props = {
            "osmId": self.osm_id,
            "name": way_record[1],
            "highway": way_record[2],
            "speed": way_record[3],
            "speedSource": way_record[4],
            "lanes": way_record[5],
            "lanesSource": way_record[6],
            "oneWay": bool(way_record[7]),
            "condition": way_record[8],
            "lts": way_record[9]
        }

        cycleway_record = cursor.execute(
            f"SELECT * FROM CYCLEWAY WHERE WAY_OSM_ID = {self.osm_id}"
        ).fetchone()

        if cycleway_record:
            cycleway_dict = {}
            for index, column in enumerate(cycleway_columns):
                if cycleway_record[index]:
                    cycleway_dict[column] = cycleway_record[index]
            props["cycleway"] = cycleway_dict

        nodes_subset_query = (
            f"AND WN.NODE_OSM_ID IN ({",".join([str(node_id) for node_id in self.node_subset])})"
            if self.node_subset
            else ""
        )

        node_records = cursor.execute(
            f"""
                SELECT N.* FROM
                WAY_NODE WN
                    LEFT JOIN NODE N ON WN.NODE_OSM_ID = N.OSM_ID
                WHERE WAY_OSM_ID = {self.osm_id}
                {nodes_subset_query}
                ORDER BY POSITION ASC
            """
        ).fetchall()
        
        long_lats = []
        for node_record in node_records:
            node_dict = {}
            for index, column in enumerate(node_columns):
                if node_record[index]:
                    node_dict[column] = node_record[index]
            props["trafficCalmed"] = "trafficCalming" in node_dict
            long_lats.append([node_dict["longitude"], node_dict["latitude"]])

        line_string = LineString(long_lats)
        return Feature(f"way/{self.osm_id}", line_string, props)

def main():
    parser = create_argparser()
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    with open(args.way_json, "r") as f:
        ways = json.load(f)

    features = []
    for way in ways:
        way_segment = WaySegment(
            way["id"], way["nodesSubset"] if "nodesSubset" in way else None
        )
        feature = way_segment.to_geojson_feature(cursor)
        features.append(feature)
    
    feature_collection = FeatureCollection(features)
    
    with open(args.output, "w") as f:
        json.dump(feature_collection, f)
        

def create_argparser():
    parser = argparse.ArgumentParser(
        description="Turn a list of Ways (and optional Node Subsets) into a FeatureCollection",
    )
    parser.add_argument(
        "--way-json",
        type=str,
        help="A file containing a JSON array of Way selections",
        required=True,
    )
    parser.add_argument(
        "--db", type=str, help="The file for the SQLite database", required=True
    )
    parser.add_argument(
        "--output", type=str, help="The file to write the FeatureCollection to", required=True
    )
    return parser


if __name__ == "__main__":
    main()
