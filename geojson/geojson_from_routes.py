from typing import Union
from geojson import FeatureCollection, Feature, LineString
import sqlite3
from sqlite3 import Cursor, Connection
import argparse
import json
import sys
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer
import functools


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

def go_boston_sort_fn(feature: Feature):
    if "goBoston" not in feature.properties:
        return 4
    go_boston = feature.properties["goBoston"]
    if go_boston == "priority":
        return 0
    elif go_boston == "future":
        return 1
    elif go_boston == "existing":
        return 2
    else:
        return 3

def time_call(func, label, show = False):
    start = timer()
    result = func()
    end = timer()
    if (show):
        print(label, '{:f}'.format(end - start))
    return result

def to_camel_case(text):
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0].lower() + "".join(i.capitalize() for i in s[1:])


class WaysWithName:
    def __init__(self,
                 name: str,
                 go_boston: Union[str, None] = None,
                 exclude_ways: Union[list[int], None] = None,
                 highway: Union[str, None] = None):
        self.name = name
        self.go_boston = go_boston
        self.exclude_ways = exclude_ways
        self.highway = highway
        
    def get_way_ids(self, cursor: Cursor):
        exclude_ways_clause = f'and osm_id not in ({",".join(str(id) for id in self.exclude_ways)})' if self.exclude_ways else ""
        highway_clause = f'and highway = "{self.highway}"' if self.highway else ''
        way_records = cursor.execute(
            f"""
            select osm_id from way where way_name = "{self.name}" {exclude_ways_clause} {highway_clause}
            """
        ).fetchall()
        return [record[0] for record in way_records]


class WaysInRelation:
    def __init__(self,
                 name: str,
                 go_boston: Union[str, None] = None,
                 exclude_ways: Union[list[int], None] = None,):
        self.name = name
        self.go_boston = go_boston
        self.exclude_ways = exclude_ways
        
    def get_way_ids(self, cursor: Cursor):
        exclude_ways_clause = f'and wr.way_osm_id not in ({",".join(str(id) for id in self.exclude_ways)})' if self.exclude_ways else ""
        way_records = cursor.execute(
            f"""
            select way_osm_id from way_relation wr left join relation r on wr.relation_osm_id = r.osm_id where r.relation_name = "{self.name}" {exclude_ways_clause}
            """
        ).fetchall()
        return [record[0] for record in way_records]


class Geometry:
    def __init__(self, id: int | None, latitude: float, longitude: float):
        self.id = id
        self.latitude = latitude
        self.longitude = longitude
        
    def __init__(self, coords: dict[str, float]):
        self.id = coords["id"] if "id" in coords else None
        self.latitude = coords["latitude"]
        self.longitude = coords["longitude"]
        
    def to_coords_list(self):
        return [self.longitude, self.latitude]
        
    
class WayGeometry:
    def __init__(self, way: dict[str, any]):
        self.geometries = [Geometry(g) for g in way["geometries"]]
        self.order = way["order"] if "order" in way else None
        self.props = way["props"]
        if "goBoston" in way:
            self.props["goBoston"] = way["goBoston"]

    def to_geojson_feature(self):
        geometries = []
        if self.order:
            geometries_dict = {g.id: g for g in self.geometries}
            for id in self.order:
                geometries = [*geometries, geometries_dict[id]]
        else:
            geometries = self.geometries
        line_string = LineString([g.to_coords_list() for g in geometries])
        return Feature(f"way/{self.props["osmId"]}", line_string, self.props)
        

class WayStartEnd:
    def __init__(self, osm_id: int, name: str, min_node_id: int, max_node_id: int, go_boston: Union[str, None] = None):
        self.osm_id: str = osm_id
        self.name: str = name
        self.min_node_id: int = min_node_id
        self.max_node_id: int = max_node_id
        self.connections: list[WayStartEnd] = []

@functools.cache
def retrieve_way_start_ends(way_name: Union[str, None], way_highway: Union[str, None], cursor: Cursor) -> list:
    where_clause = ' and '.join(
        list(
            filter(
                lambda c: c is not None,
                [
                    f'w.way_name = "{way_name}"' if way_name else None,
                    f'w.highway = "{way_highway}"' if way_highway else None
                ]
            )
        )
    )
    way_records = cursor.execute(
        f"""
            select min_node.osm_id, min_node.way_name, min_node_id, max_node_id
            from (
                select w.osm_id, w.way_name, min(wn.position), n.osm_id as min_node_id 
                from way w
                    left join way_node wn on w.osm_id = wn.way_osm_id
                    left join node n on wn.node_osm_id = n.osm_id
                where {where_clause}
                group by wn.way_osm_id
            ) min_node
                left join (
                    select w.osm_id, w.way_name, max(wn.position), n.osm_id as max_node_id
                    from way w
                        left join way_node wn on w.osm_id = wn.way_osm_id
                        left join node n on wn.node_osm_id = n.osm_id 
                        where {where_clause}
                        group by wn.way_osm_id
                    ) max_node on min_node.osm_id = max_node.osm_id

        """
    ).fetchall()
    if len(way_records) == 0:
        return []
    way_start_ends = [WayStartEnd(tup[0], tup[1], tup[2], tup[3]) for tup in way_records]
    for way_start_end in way_start_ends:
        previous_ways = [w for w in way_start_ends if way_start_end.min_node_id == w.min_node_id or way_start_end.min_node_id == w.max_node_id and way_start_end.osm_id != w.osm_id]
        next_ways = [w for w in way_start_ends if way_start_end.max_node_id == w.min_node_id or way_start_end.max_node_id == w.max_node_id  and way_start_end.osm_id != w.osm_id]
        way_start_end.connections = [*previous_ways, *next_ways]
    return way_start_ends


class WayRange:
    def __init__(self,
                 from_osm_id: int,
                 to_osm_id: int,
                 name: Union[str, None] = None,
                 highway: Union[str, None] = None,
                 exclude_nodes: Union[list[int], None] = None,
                 no_goes: Union[list[int], None] = None,
                 go_boston: Union[str, None] = None):
        self.from_osm_id = from_osm_id
        self.to_osm_id = to_osm_id
        self.name = name
        self.highway = highway
        self.exclude_nodes = exclude_nodes
        self.no_goes = no_goes if no_goes else []
        self.go_boston = go_boston
        
    def get_way_ids(self, cursor: Cursor):
        way_start_ends = retrieve_way_start_ends(self.name, self.highway, cursor)
        starting_way_end: WayStartEnd = next((x for x in way_start_ends if x.osm_id == self.from_osm_id), None)
        ending_way_end: WayStartEnd = next((x for x in way_start_ends if x.osm_id == self.to_osm_id), None)
        if not starting_way_end or not ending_way_end:
            return []
        path = self.find_path(starting_way_end, ending_way_end, [], self.no_goes)
        if not path:
            return []
        return path

    def find_path(self, start: WayStartEnd, end, acc: list[int], no_goes: list[int]):
        if len(start.connections) == 0 or start.osm_id in acc or start.osm_id in no_goes:
            return None
        if start.osm_id == end.osm_id:
            return [*acc, end.osm_id]
        for connection in start.connections:
            path = self.find_path(connection, end, [*acc, start.osm_id], no_goes)
            if path:
                return path
        return None


class WayFill:
    def __init__(self,
                 start_osm_id: int,
                 name: Union[str, None] = None,
                 highway: Union[str, None] = None,
                 exclude_nodes: Union[list[int], None] = None,
                 no_goes: Union[list[int], None] = None,
                 go_boston: Union[str, None] = None):
        self.start_osm_id = start_osm_id
        self.name = name
        self.highway = highway
        self.exclude_nodes = exclude_nodes
        self.no_goes = no_goes if no_goes else []
        self.go_boston = go_boston
        
    def get_way_ids(self, cursor: Cursor):
        way_start_ends = retrieve_way_start_ends(self.name, self.highway, cursor)
        starting_way_end: WayStartEnd = next((x for x in way_start_ends if x.osm_id == self.start_osm_id), None)
        if not starting_way_end:
            return []
        path = self.find_paths(starting_way_end, [], self.no_goes)
        if not path:
            return []
        return path
    

    def find_paths(self, start: WayStartEnd, acc: set[int], no_goes: list[int]):
        if len(start.connections) == 0:
            return set([*acc, start.osm_id])
        if start.osm_id in acc or start.osm_id in no_goes:
            return acc
        way_ids = []
        for connection in start.connections:
            if connection.osm_id not in acc:
                way_ids = set([*way_ids, *self.find_paths(connection, [*acc, start.osm_id], no_goes)])
        return way_ids


class WaySegment:
    def __init__(self, osm_id: int, node_subset: Union[list[int], None] = None, exclude_nodes: Union[list[int], None] = None, go_boston: Union[str, None] = None):
        self.osm_id = osm_id
        self.node_subset = node_subset
        self.exclude_nodes = exclude_nodes
        self.go_boston = go_boston

    def to_geojson_feature(self, cursor: Cursor):
        query = f"""
                SELECT W.*, LTS.LTS
                FROM WAY W
                    LEFT JOIN LEVEL_OF_TRAFFIC_STRESS LTS ON W.OSM_ID = LTS.WAY_OSM_ID
                WHERE OSM_ID = {self.osm_id}
            """
        way_record = cursor.execute(query).fetchone()
        if not way_record:
            print(f"Way [{self.osm_id} does not exist in the LTS database. Did you load all required cities?]", sys.stderr)
            return None
            # raise ValueError(f"Way [{self.osm_id} does not exist in the LTS database]")
        
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
            "length": way_record[9],
            "lts": way_record[10],
            "goBoston": self.go_boston
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
        excluded_nodes_query = (
            f"AND WN.NODE_OSM_ID NOT IN ({",".join([str(node_id) for node_id in self.exclude_nodes])})"
            if self.exclude_nodes
            else ""
        )

        node_records = cursor.execute(
            f"""
                SELECT N.* FROM
                WAY_NODE WN
                    LEFT JOIN NODE N ON WN.NODE_OSM_ID = N.OSM_ID
                WHERE WAY_OSM_ID = {self.osm_id}
                {nodes_subset_query}
                {excluded_nodes_query}
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


def process_file(route_json: str, cursor: Cursor):
    with open(route_json, "r") as f:
        route = json.load(f)
    ways = route["ways"]
    features = []
    for way in ways:
        if way["type"] == "way":
            way_segment = WaySegment(
                way["id"],
                way["nodesSubset"] if "nodesSubset" in way else None,
                way["excludeNodes"] if "excludeNodes" in way else None,
                way["goBoston"] if "goBoston" in way else None
            )
            feature = way_segment.to_geojson_feature(cursor)
            if feature:
                features.append(feature)
                
        if way["type"] == "way_range":
            way_range = WayRange(
                way["fromId"],
                way["toId"],
                way["name"] if "name" in way else None,
                way["highway"] if "highway" in way else None,
                way["excludeNodes"] if "excludeNodes" in way else None,
                way["noGoes"] if "noGoes" in way else None,
                way["goBoston"] if "goBoston" in way else None
            )
            way_ids = time_call(lambda: way_range.get_way_ids(cursor), "way_range_get_ids " + way["highway"] if "highway" in way else way["name"])
            if way_ids:
                features = [*features, *[WaySegment(way_id, exclude_nodes=way_range.exclude_nodes, go_boston=way_range.go_boston).to_geojson_feature(cursor) for way_id in way_ids]]
        
        if way["type"] == "way_fill":
            way_fill = WayFill(
                way["startId"],
                way["name"] if "name" in way else None,
                way["highway"] if "highway" in way else None,
                way["excludeNodes"] if "excludeNodes" in way else None,
                way["noGoes"] if "noGoes" in way else None,
                way["goBoston"] if "goBoston" in way else None
            )
            way_ids = time_call(lambda: way_fill.get_way_ids(cursor), "way_fill_get_ids " + f'{way["highway"] if "highway" in way else ""} - {way["name"] if "name" in way else ""}')
            if way_ids:
                features = [*features, *[WaySegment(way_id, exclude_nodes=way_fill.exclude_nodes, go_boston=way_fill.go_boston).to_geojson_feature(cursor) for way_id in way_ids]]
        

        if way["type"] == "ways_with_name":
            ways_with_name = WaysWithName(
                way["name"],
                way["goBoston"] if "goBoston" in way else None,
                way["excludeWays"] if "excludeWays" in way else None,
                way["highway"] if "highway" in way else None,
            )
            way_ids = time_call(lambda: ways_with_name.get_way_ids(cursor), "ways_with_name")
            features = [*features, *[WaySegment(way_id, go_boston=ways_with_name.go_boston).to_geojson_feature(cursor) for way_id in way_ids]]
            
        if way["type"] == "ways":
            features = [*features, *[WaySegment(way_id, go_boston=way["goBoston"] if "goBoston" in way else None, exclude_nodes=way["excludeNodes"] if "excludeNodes" in way else None).to_geojson_feature(cursor) for way_id in way["ids"]]]
        
        if way["type"] == "way_geometry":
            features = [*features, WayGeometry(way).to_geojson_feature()]
        
        if way["type"] == "way_relation":
            ways_in_relation = WaysInRelation(
                way["name"],
                way["goBoston"] if "goBoston" in way else None,
                way["excludeWays"] if "excludeWays" in way else None,
            )
            way_ids = time_call(lambda: ways_in_relation.get_way_ids(cursor), "ways_in_relation")
            features = [*features, *[WaySegment(way_id, go_boston=ways_in_relation.go_boston).to_geojson_feature(cursor) for way_id in way_ids]]
         
    return features


def main():
    parser = create_argparser()
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    features = []
    if args.route_json:
        features = [*process_file(args.route_json, cursor)]
    
    if args.route_jsons:
        for route_json in args.route_jsons:
            features = [*features, *process_file(route_json, cursor)]
            
    if args.routes_dir:
        routes_dir = args.routes_dir
        files = [join(routes_dir, file) for file in listdir(routes_dir) if isfile(join(routes_dir, file))]
        for route_json in files:
            features = [*features, *process_file(route_json, cursor)]
    
    features.sort(key=go_boston_sort_fn, reverse=True)
    
    feature_collection = FeatureCollection(features)

    output_dict = {
        "featureCollection": feature_collection
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f)
        

def create_argparser():
    parser = argparse.ArgumentParser(
        description="Turn a list of Ways (and optional Node Subsets) into a FeatureCollection",
    )
    parser.add_argument(
        "--route-json",
        type=str,
        help="A file containing a Route JSON object with array of Way selections",
        required=False,
    ),
    parser.add_argument(
        "--route-jsons",
        type=str,
        nargs='+',
        help="Files containing a Route JSON object with array of Way selections. They will be concatenated in the output",
        required=False,
    ),
    parser.add_argument(
        "--routes-dir",
        type=str,
        help="A directory containing JSON files of Route objects with arrays of Way selections",
        required=False,
    ),
    parser.add_argument(
        "--db", type=str, help="The file for the SQLite database", required=True
    )
    parser.add_argument(
        "--output", type=str, help="The file to write the FeatureCollection to", required=True
    )
    return parser


if __name__ == "__main__":
    main()
