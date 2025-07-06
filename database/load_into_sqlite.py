import sqlite3
from sqlite3 import Connection, Cursor
import pandas as pd
from pandas import DataFrame, Series
import math
import sys
import json
import argparse



def main():
    """ Loads OpenStreetMap Way, and Node data, along with a bunch of tags related to cycling, into a sqlite database."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="A CSV file containing OSM data annotated with LTS tags")
    parser.add_argument("--db", type=str, default="lts.db", help="The file for the SQLite database")
    parser.add_argument("--schema", type=str, default="way_schema.sql", help="The file containing the schema to initialize the SQLite database with")
    parser.add_argument("--nodes", type=str, default="nodes.json", help="""
                        An Overpass-returned JSON file containing Node information, as well as Way information relating Ways to Nodes.
                        This data can be fetched from Overpass using this query for Boston as an example:
                        ```
                        [timeout:600][out:json][maxsize:2000000000];
                        area["wikipedia"="en:Boston"]->.search_area;
                        (
                            way[highway][footway!=sidewalk][service!=parking_aisle](area.search_area);
                            way[footway=sidewalk][bicycle][bicycle!=no][bicycle!=dismount](area.search_area);
                        )->.ways;
                        (.ways;>;)->.nodes;
                        .nodes out;
                        .ways out skel;
                        ```
                        """)
    args = parser.parse_args()
    
    
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("PRAGMA synchronous = normal")

    try:
        create_tables(args.schema, cursor)

        process_data(args.data, cursor)
    except ValueError as ve:
        print(ve, file=sys.stderr)
        conn.rollback()
        conn.close()
        exit(1)

    conn.close()


def create_tables(schema_file: str, cursor: Cursor):
    """Creates tables in the SQLite database according to the schema_file
    
    Parameters:
    schema_file (str): A file containing SQL statements to initialize/migrate the database
    cursor (Cursor): Cursor to execute statements in the database with
    
    """
    with open(schema_file, "r", encoding="utf-8") as f:
        create_tables_queries = f.read()

    for query in create_tables_queries.split(";"):
        cursor.execute(query)
    cursor.connection.commit()


def process_data(data_file: str, cursor: Cursor):
    """Inserts OSM data into the database
    
    Parameters:
    data_file (str): File containing LTS-annotated OSM data
    cursor (Cursor): Cursor to connect to the database with
    """
    df = pd.read_csv(data_file, index_col="osdmid", low_memory=False)

    osm_ids: set[int] = set(df.index.array)
    insert_ways(df, osm_ids, cursor)
    cursor.connection.commit()

    insert_lts(df, osm_ids, cursor)
    cursor.connection.commit()

    insert_cycleways(df, cursor)
    cursor.connection.commit()

    insert_nodes(cursor)
    cursor.connection.commit()


def insert_ways(df: DataFrame, osm_ids, cursor):
    """Inserts data into the WAY table
    
    Parameters:
    df (DataFrame): DataFrame containing the LTS-annotated way data
    osm_ids (set[int]): All the OSM IDs to insert
    cursor (Cursor): Cursor to connect to the database with

    """
    way_db_columns = [
        ("OSM_ID", "INTEGER"),
        ("WAY_NAME", "TEXT"),
        ("HIGHWAY", "TEXT"),
        ("MAXSPEED_MPH", "INTEGER"),
        ("MAXSPEED_RULE", "TEXT"),
        ("LANE_COUNT", "INTEGER"),
        ("LANE_COUNT_RULE", "TEXT"),
        ("ONE_WAY", "BOOLEAN"),
        ("CONDITION", "TEXT"),
    ]

    way_column_labels = [
        "name",
        "highway",
        "speed",
        "speed_rule",
        "lane_count",
        "lane_rule",
        "oneway",
        "condition",
    ]

    db_insert_dataframe(
        df.get(way_column_labels),
        osm_ids,
        way_db_columns,
        "WAY",
        way_column_labels,
        prepare_row,
        cursor,
    )


def insert_cycleways(df: DataFrame, cursor: Cursor):
    """Inserts data into the CYCLEWAY table. 
    Filters the DataFrame to and only iterates over rows that have cycleway data.
    
    Parameters:
    df (DataFrame): DataFrame containing the LTS-annotated way data
    cursor (Cursor): Cursor to connect to the database with

    """
    cycleway_db_columns = [
        ("WAY_OSM_ID", "INTEGER"),
        ("CYCLEWAY_TYPE", "TEXT"),
        ("CYCLEWAY_LANE", "TEXT"),
        ("CYCLEWAY_SURFACE", "TEXT"),
        ("LEFT_TYPE", "TEXT"),
        ("LEFT_LANE", "TEXT"),
        ("LEFT_WIDTH", "TEXT"),
        ("LEFT_BUFFER", "BOOLEAN"),
        ("LEFT_SEPARATION", "TEXT"),
        ("LEFT_REVERSED", "BOOLEAN"),
        ("RIGHT_TYPE", "TEXT"),
        ("RIGHT_LANE", "TEXT"),
        ("RIGHT_WIDTH", "TEXT"),
        ("RIGHT_BUFFER", "BOOLEAN"),
        ("RIGHT_SEPARATION", "TEXT"),
        ("RIGHT_REVERSED", "BOOLEAN"),
    ]

    cycleway_column_labels = [
        "cycleway",
        "cycleway:lane",
        "cycleway:surface",
        "cycleway:left",
        "cycleway:left:lane",
        "cycleway:left:width",
        "cycleway:left:buffer",
        "cycleway:left:separation",
        "cycleway:left:oneway",
        "cycleway:right",
        "cycleway:right:lane",
        "cycleway:right:width",
        "cycleway:right:buffer",
        "cycleway:right:separation",
        "cycleway:right:oneway",
    ]

    db_insert_cycleway_dataframe(
        df.get(cycleway_column_labels),
        cycleway_db_columns,
        "CYCLEWAY",
        cycleway_column_labels,
        cursor,
    )


def prepare_cycleway_row(
    osm_id: int, column_data_types: list[str], series_labels: list[str], series: Series
) -> list:
    """Prepares a Series for insertion into the CYCLEWAY table as values
    
    Parameters:
    osm_id (int): OSM ID of the Way this data belongs to
    column_data_types (list[str]): An ordered list of SQLite data types for each column
    series_labels (list[str]): An ordered list of labels to extract data from the Series with
    series (Series): A row of values from a DataFrame
    
    Returns:
    list[any]: Values appropriately transformed and ready to be inserted into the database

    """
    values = [osm_id] + [series.at[label] for label in series_labels]

    for index, value in enumerate(values):
        if (
            value is None
            or (column_data_types[index] in ("INTEGER", "REAL") and math.isnan(value))
            or (str(value) == "nan")
        ):
            values[index] = "NULL"
        else:
            match (column_data_types[index]):
                case "TEXT":
                    values[index] = f'"{value}"'
                case "INTEGER":
                    values[index] = f"{int(value)}"
                case "REAL":
                    values[index] = f"{float(value)}"
                case "BOOLEAN":
                    values[index] = handle_cycleway_boolean(value)
                case _:
                    raise ValueError(
                        f"cannot handle column data type [{column_data_types[index]}]"
                    )
    return values


def handle_cycleway_boolean(boolValue) -> str:
    """Special Boolean handling for Cycleway information, as "1" and "-1" have special meaning.
    
    Parameters:
    boolValue: Boolean value to transform
    
    Retuns:
    str: The TRUE/FALSE value to insert into the database
    """
    match (boolValue):
        case True:
            return "TRUE"
        case False:
            return "FALSE"
        case "yes":
            return "TRUE"
        case "no":
            return "FALSE"
        case -1 | "-1":
            return "TRUE"
        case 1 | "1":
            return "FALSE"


def db_insert_cycleway_dataframe(
    df: DataFrame,
    db_column_names_types: list[tuple[str, str]],
    df_column_labels: list[str],
    cursor: Cursor,
):
    """Insert all the rows in the DataFrame into the database for the CYCLEWAY table
    
    Parameters:
    df (DataFrame): DataFrame with all the LTS-annotated Way information
    db_column_names_types (list[tuple[str, str]]): A list of tuples associating database column names with data types
    df_column_labels (list[str]): An ordered list of column lables for the DataFrame that correspond to the database columns
    cursor (Cursor): Cursor to connect to the database with

    """
    db_table_name = "CYCLEWAY"

    query_cycleway_result = df.query(
        f"`cycleway`.notnull() or `cycleway:lane`.notnull() or `cycleway:surface`.notnull() or `cycleway:left`.notnull() or `cycleway:left:lane`.notnull() or `cycleway:left:width`.notnull() or `cycleway:left:buffer`.notnull() or `cycleway:left:separation`.notnull() or `cycleway:left:oneway`.notnull() or `cycleway:right`.notnull() or `cycleway:right:lane`.notnull() or `cycleway:right:width`.notnull() or `cycleway:right:buffer`.notnull() or `cycleway:right:separation`.notnull() or `cycleway:right:oneway`.notnull()"
    )

    osm_ids_with_cycleway = set(query_cycleway_result.index.array)

    osm_ids_to_load = find_new_osm_ids(
        osm_ids_with_cycleway, db_table_name, db_column_names_types[0], cursor
    )

    done = 0
    for osm_id in osm_ids_to_load:
        query_result = df.query(f"osdmid == {osm_id}")
        sub_df = query_result

        first_series = sameness_check(sub_df, osm_id, db_table_name)
        if first_series is None:
            raise ValueError(f"{db_table_name} data for ${osm_id} is not consistent")

        values = prepare_cycleway_row(
            osm_id,
            [name_type[1] for name_type in db_column_names_types],
            df_column_labels,
            first_series,
        )

        insert_statement = f"""
        INSERT INTO {db_table_name}
            ({",".join([name_type[0] for name_type in db_column_names_types])})
            VALUES({",".join(values)})
        """

        cursor.execute(insert_statement)

        done += 1
        if done % 1000 == 0:
            print(f"Analyzed and inserted {db_table_name} rows for {done} OSM_WAY_IDs")


def insert_lts(df: DataFrame, osm_ids: set[int], cursor: Cursor):
    """Inserts data into the LEVEL_OF_TRAFFIC_STRESS table.
    
    Parameters:
    df (DataFrame): DataFrame containing the LTS-annotated way data
    osm_ids (set[int]): All the OSM IDs to insert
    cursor (Cursor): Cursor to connect to the database with

    """
    lts_db_columns = [
        ("WAY_OSM_ID", "INTEGER"),
        ("LTS", "INTEGER"),
        ("LTS_FWD", "INTEGER"),
        ("LTS_REV", "INTEGER"),
        ("BIKE_ACCESS", "INTEGER"),
        ("BIKE_ACCESS_FWD", "INTEGER"),
        ("BIKE_ACCESS_REV", "INTEGER"),
        ("SEPARATION_FWD", "INTEGER"),
        ("SEPARATION_REV", "INTEGER"),
        ("MIXED_FWD", "INTEGER"),
        ("MIXED_REV", "INTEGER"),
        ("BIKE_LANE_NO_PARKING_FWD", "INTEGER"),
        ("BIKE_LANE_NO_PARKING_REV", "INTEGER"),
        ("BIKE_LANE_YES_PARKING_FWD", "INTEGER"),
        ("BIKE_LANE_YES_PARKING_REV", "INTEGER"),
    ]

    lts_column_labels = [
        "LTS",
        "LTS_fwd",
        "LTS_rev",
        "LTS_bike_access",
        "LTS_bike_access_fwd",
        "LTS_bike_access_rev",
        "LTS_separation_fwd",
        "LTS_separation_rev",
        "LTS_mixed_fwd",
        "LTS_mixed_rev",
        "LTS_bikelane_noparking_fwd",
        "LTS_bikelane_noparking_rev",
        "LTS_bikelane_yesparking_fwd",
        "LTS_bikelane_yesparking_rev",
    ]

    db_insert_dataframe(
        df.get(lts_column_labels),
        osm_ids,
        lts_db_columns,
        "LEVEL_OF_TRAFFIC_STRESS",
        lts_column_labels,
        prepare_row,
        cursor,
    )


def prepare_row(
    osm_id: int, column_data_types: list[str], series_labels: list[str], series: Series
) -> list:
    """Prepares a Series for insertion into a table as values
    
    Parameters:
    osm_id (int): OSM ID of the Way this data belongs to
    column_data_types (list[str]): An ordered list of SQLite data types for each column
    series_labels (list[str]): An ordered list of labels to extract data from the Series with
    series (Series): A row of values from a DataFrame
    
    Returns:
    list[any]: Values appropriately transformed and ready to be inserted into the database

    """
    values = [osm_id] + [series.at[label] for label in series_labels]

    for index, value in enumerate(values):
        if (
            value is None
            or (column_data_types[index] in ("INTEGER", "REAL") and math.isnan(value))
            or (str(value) == "nan")
        ):
            values[index] = "NULL"
        else:
            match (column_data_types[index]):
                case "TEXT":
                    values[index] = f'"{value}"'
                case "INTEGER":
                    values[index] = f"{int(value)}"
                case "REAL":
                    values[index] = f"{float(value)}"
                case "BOOLEAN":
                    values[index] = handle_boolean(value)
                case _:
                    raise ValueError(
                        f"cannot handle column data type [{column_data_types[index]}]"
                    )
    return values


def handle_boolean(boolValue) -> str:
    """Boolean handling for translating all the ways OSM/Python represents true and false
    
    Parameters:
    boolValue: Boolean value to transform
    
    Retuns:
    str: The TRUE/FALSE value to insert into the database
    """
    match (boolValue):
        case True:
            return "TRUE"
        case False:
            return "FALSE"
        case "yes":
            return "TRUE"
        case "no":
            return "FALSE"
        case 1:
            return "TRUE"
        case 0:
            return "FALSE"


def db_insert_dataframe(
    df: DataFrame,
    osm_ids: set[int],
    db_column_names_types: list[tuple[str, str]],
    db_table_name: str,
    df_column_labels: list[str],
    cursor: Cursor,
):
    """Insert all the rows in the DataFrame into the database for a table
    
    Parameters:
    df (DataFrame): DataFrame with all the LTS-annotated Way information
    osm_ids (set[int]): OSM IDs of Ways information belongs to
    db_column_names_types (list[tuple[str, str]]): A list of tuples associating database column names with data types
    db_table_name (str): Table to insert data into
    df_column_labels (list[str]): An ordered list of column lables for the DataFrame that correspond to the database columns
    cursor (Cursor): Cursor to connect to the database with

    """

    osm_ids_to_load = find_new_osm_ids(
        osm_ids, db_table_name, db_column_names_types[0], cursor
    )

    done = 0
    for osm_id in osm_ids_to_load:
        query_result = df.query(f"osdmid == {osm_id}")
        sub_df = query_result.get(df_column_labels)

        first_series = sameness_check(sub_df, osm_id, db_table_name)
        if first_series is None:
            raise ValueError(f"{db_table_name} data for ${osm_id} is not consistent")

        values = prepare_row(
            osm_id,
            [name_type[1] for name_type in db_column_names_types],
            df_column_labels,
            first_series,
        )

        insert_statement = f"""
        INSERT INTO {db_table_name}
            ({",".join([name_type[0] for name_type in db_column_names_types])})
            VALUES({",".join(values)})
        """

        cursor.execute(insert_statement)

        done += 1
        if done % 1000 == 0:
            print(f"Analyzed and inserted {db_table_name} rows for {done} OSM_WAY_IDs")


def find_new_osm_ids(
    osm_ids: set[int],
    table_name: str,
    id_column_name_type: tuple[str, str],
    cursor: Cursor,
) -> set[int]:
    """Find OSM IDs that have already been inserted and subtract them from the set of OSM IDs to load
    
    Parameters:
    osm_ids (set[int]): OSM IDs of Ways information belongs to
    table_name (str): Table to insert data into
    id_column_name_type (tuple[str, str]): A tuples associating database ID column name with data type
    cursor (Cursor): Cursor to connect to the database with
    
    Returns:
    set[int]: A set of OSM IDs that do not exist in the database

    """
    print(
        f"Found {len(osm_ids)} candidate OSM_IDs to insert into the {table_name} table"
    )
    osm_ids_from_db_result = cursor.execute(
        f"SELECT {id_column_name_type[0]} FROM {table_name}"
    )
    db_osm_ids = set(map(lambda row: row[0], osm_ids_from_db_result.fetchall()))
    print(f"Found {len(db_osm_ids)} already loaded into the {table_name} table")
    osm_ids_to_load = osm_ids - db_osm_ids
    print(f"Found {len(osm_ids_to_load)} that do not exist in the {table_name} table")
    return osm_ids_to_load


def sameness_check(df: DataFrame, osm_id: int, type: str) -> Series:
    """Check to see if all rows in the DataFrame are equal. 
    This check is to ensure that all the node-pairs in the DataFrame for a Way match.
    They should, but this makes absolutely sure.

    Returns:
    Series: A representative Series containing values to insert. None if rows are not equal.
    """
    all_same = True
    first_series = None
    for index, data in df.iterrows():
        if first_series is None:
            first_series = data
        else:
            if not first_series.equals(data):
                all_same = False
                break
    if not all_same:
        print(f"OSM ID {osm_id} has differing {type} data across rows", file=sys.stderr)
        print(df, file=sys.stderr)
        return None

    return first_series


def insert_nodes(cursor: Cursor):
    """Insert Nodes into the NODE table of the database.
    Also inserts relations between WAY and NODE via WAY_NODE.
    WAY table _must_ be populated before this function is run, or foreign-key errors will occur.
    
    Parameters:
    cursor (Cursor): Cursor to connect to the database with
    """
    with open("data/Boston_nodes_1.json", "r") as f:
        nodes_response = json.load(f)

    nodes_df = pd.json_normalize(
        [node for node in nodes_response["elements"] if node["type"] == "node"]
    )
    ways_df = pd.json_normalize(
        [
            {k: way[k] for k in ("id", "type", "nodes")}
            for way in nodes_response["elements"]
            if way["type"] == "way"
        ]
    )

    node_db_columns = [
        ("OSM_ID", "INTEGER"),
        ("LATITUDE", "REAL"),
        ("LONGITUDE", "REAL"),
        ("HIGHWAY", "TEXT"),
        ("TRAFFIC_CALMING", "TEXT"),
        ("CROSSING", "TEXT"),
        ("CROSSING_MARKINGS", "TEXT"),
        ("CROSSING_ISLAND", "TEXT"),
    ]

    node_column_labels = [
        "lat",
        "lon",
        "tags.highway",
        "tags.traffic_calming",
        "tags.crossing",
        "tags.crossing:markings",
        "tags.crossing:island",
    ]
    
    node_ids = set(nodes_df.id.array)
    node_ids_to_load = find_new_osm_ids(
        node_ids, "NODE", node_db_columns[0], cursor
    )
    way_ids_in_db = set([row[0] for row in cursor.execute("SELECT OSM_ID FROM WAY").fetchall()])
    
    done = 0
    for _, node in nodes_df.get(["id"] + node_column_labels).iterrows():
        node_id = node.at["id"]
        if node_id not in node_ids_to_load:
            continue
        values = prepare_row(
            node_id,
            [name_type[1] for name_type in node_db_columns],
            node_column_labels,
            node.get(node_column_labels),
        )
        
        insert_node_statement = f"""
            INSERT INTO NODE
                ({",".join([name_type[0] for name_type in node_db_columns])})
            VALUES({",".join(values)})
        """
        cursor.execute(insert_node_statement)
        
        done += 1
        if done % 1000 == 0:
            print(f"Analyzed and inserted NODE rows for {done} OSM_NODE_IDs")
    
    node_way_done = 0
    for _, way in ways_df.iterrows():
        way_id = way.at["id"]
        if way_id not in way_ids_in_db:
            continue
        for position, way_node_id in enumerate(way.at["nodes"]):
            insert_node_way_statement = f"""
                INSERT INTO NODE_WAY
                    (WAY_OSM_ID, NODE_OSM_ID, POSITION)
                VALUES({way_id}, {way_node_id}, {position})
            """
            cursor.execute(insert_node_way_statement)
        node_way_done += 1
        if node_way_done % 1000 == 0:
            print(f"Analyzed and inserted NODE_WAY rows for {node_way_done} OSM_WAY_IDs")


if __name__ == "__main__":
    main()
