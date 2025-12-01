import sqlite3
from sqlite3 import Connection, Cursor
import pandas as pd
from pandas import DataFrame, Series
import math
import sys
import json
import argparse
import textwrap


def main():
    """Loads OpenStreetMap Way, and Node data, along with a bunch of tags related to cycling, into a sqlite database."""

    parser = create_argparser()
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("PRAGMA synchronous = normal")

    try:
        if args.schema:
            create_tables(args.schema, cursor)

        process_data(args.lts_data, args.node_data, cursor)
    except ValueError as ve:
        print(ve, file=sys.stderr)
        conn.rollback()
        conn.close()
        exit(1)

    conn.close()


def create_argparser():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
                       Loads data annotated with Level of Traffic Stress scores into a SQLite database.
                       Assumes that LTS_OSM.py has already been run for a city, and that Node information has separately been downloaded from OpenStreeMap.
                       """
        ),
        epilog=textwrap.dedent(
            """
                           Downloading OSM Node Data:
                               `sed -e 's/WIKI_CITY/Chelsea, Massachusetts/g' database/nodes/base.query | curl -X GET --data @- https://overpass-api.de/api/interpreter`
                                              """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lts-data",
        type=str,
        help="The LTS-annotated CSV for the city to load into the db",
        required=True,
    )
    parser.add_argument(
        "--node-data",
        type=str,
        help="The OpenStreetMap Node information for the city to load into the db",
        required=True,
    )
    parser.add_argument(
        "--db", type=str, default="lts.db", help="The file for the SQLite database"
    )
    parser.add_argument(
        "--schema",
        type=str,
        help="The file containing the schema to initialize the SQLite database with",
    )
    return parser


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


def process_data(lts_data_file: str, node_data_file: str, cursor: Cursor):
    """Inserts OSM data into the database

    Parameters:
    lts_data_file (str): File containing LTS-annotated OSM data
    node_data_file (str): File containing Node OSM data
    cursor (Cursor): Cursor to connect to the database with
    """
    df = pd.read_csv(lts_data_file, index_col="osmid", low_memory=False)

    osm_ids: set[int] = set(df.index.array)
    insert_ways(df, osm_ids, cursor)
    cursor.connection.commit()

    insert_lts(df, osm_ids, cursor)
    cursor.connection.commit()

    insert_cycleways(df, cursor)
    cursor.connection.commit()

    insert_nodes(node_data_file, cursor)
    cursor.connection.commit()


def insert_ways(df: DataFrame, osm_ids, cursor):
    """Inserts data into the WAY table

    Parameters:
    df (DataFrame): DataFrame containing the LTS-annotated way data
    osm_ids (set[int]): All the OSM IDs to insert
    cursor (Cursor): Cursor to connect to the database with

    """
    way_columns = [
        ("OSM_ID", "INTEGER", "osmid"),
        ("WAY_NAME", "TEXT", "name"),
        ("HIGHWAY", "TEXT", "highway"),
        ("MAXSPEED_MPH", "INTEGER", "speed"),
        ("MAXSPEED_RULE", "TEXT", "speed_rule"),
        ("LANE_COUNT", "INTEGER", "lane_count"),
        ("LANE_COUNT_RULE", "TEXT", "lane_rule"),
        ("ONE_WAY", "BOOLEAN", "oneway"),
        ("CONDITION", "TEXT", "condition"),
    ]

    db_insert_dataframe(
        df.get([column[2] for column in way_columns[1:]]),
        osm_ids,
        way_columns,
        "WAY",
        cursor,
    )


def insert_cycleways(df: DataFrame, cursor: Cursor):
    """Inserts data into the CYCLEWAY table.
    Filters the DataFrame to and only iterates over rows that have cycleway data.

    Parameters:
    df (DataFrame): DataFrame containing the LTS-annotated way data
    cursor (Cursor): Cursor to connect to the database with

    """
    cycleway_columns = [
        ("WAY_OSM_ID", "INTEGER", "osmid"),
        ("CYCLEWAY_TYPE", "TEXT", "cycleway"),
        ("CYCLEWAY_LANE", "TEXT", "cycleway:lane"),
        ("CYCLEWAY_SURFACE", "TEXT", "cycleway:surface"),
        ("LEFT_TYPE", "TEXT", "cycleway:left"),
        ("LEFT_LANE", "TEXT", "cycleway:left:lane"),
        ("LEFT_WIDTH", "TEXT", "cycleway:left:width"),
        ("LEFT_BUFFER", "BOOLEAN", "cycleway:left:buffer"),
        ("LEFT_SEPARATION", "TEXT", "cycleway:left:separation"),
        ("LEFT_REVERSED", "BOOLEAN", "cycleway:left:oneway"),
        ("RIGHT_TYPE", "TEXT", "cycleway:right"),
        ("RIGHT_LANE", "TEXT", "cycleway:right:lane"),
        ("RIGHT_WIDTH", "TEXT", "cycleway:right:width"),
        ("RIGHT_BUFFER", "BOOLEAN", "cycleway:right:buffer"),
        ("RIGHT_SEPARATION", "TEXT", "cycleway:right:separation"),
        ("RIGHT_REVERSED", "BOOLEAN", "cycleway:right:oneway"),
    ]

    db_insert_cycleway_dataframe(
        df,
        cycleway_columns,
        cursor,
    )


def prepare_row(
    osm_id: int,
    columns: list[tuple[str, str, str]],
    series: Series,
) -> list:
    """Prepares a Series for insertion into the CYCLEWAY table as values

    Parameters:
    osm_id (int): OSM ID of the Way this data belongs to
    columns (list[tuple[str, str, str]]): An ordered list tuples containing DB column names, data types, and DataFrame column labels
    series (Series): A row of values from a DataFrame

    Returns:
    list[any]: Values appropriately transformed and ready to be inserted into the database

    """

    values = [
        str(osm_id)
    ]  # + [series.at[label] for label in series_labels if label in]

    for db_column, db_type, df_column in columns[1:]:
        if df_column not in series.index.array:
            values.append("NULL")
        else:
            value = series.at[df_column]
            if (
                value is None
                or (db_type in ("INTEGER", "REAL") and math.isnan(value))
                or (str(value) == "nan")
            ):
                values.append("NULL")
            else:
                match (db_type):
                    case "TEXT":
                        values.append(f'"{value.replace('"', "'")}"')
                    case "INTEGER":
                        values.append(f"{int(value)}")
                    case "REAL":
                        values.append(f"{float(value)}")
                    case "BOOLEAN":
                        values.append(handle_cycleway_boolean(value))
                    case _:
                        raise ValueError(
                            f"cannot handle column data type [{db_type}] for column [{db_column}]"
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
        case _:
            return "TRUE"


def db_insert_cycleway_dataframe(
    df: DataFrame,
    columns: list[tuple[str, str, str]],
    cursor: Cursor,
):
    """Insert all the rows in the DataFrame into the database for the CYCLEWAY table

    Parameters:
    df (DataFrame): DataFrame with all the LTS-annotated Way information
    columns (list[tuple[str, str, str]]): A list of tuples associating database column names with data types and DataFrame columns
    cursor (Cursor): Cursor to connect to the database with

    """
    db_table_name = "CYCLEWAY"

    df_labels = set(df.columns.values)
    cycleway_labels_that_exist = set.intersection(
        set([column[2] for column in columns]), df_labels
    )
    query = set([f"`{label}`.notnull()" for label in cycleway_labels_that_exist])

    query_cycleway_result = df.query("or".join(query))

    osm_ids_with_cycleway = set(query_cycleway_result.index.array)

    osm_ids_to_load = find_new_osm_ids(
        osm_ids_with_cycleway, db_table_name, columns[0], cursor
    )

    done = 0
    for osm_id in osm_ids_to_load:
        query_result = df.query(f"osmid == {osm_id}").get(
            list(cycleway_labels_that_exist)
        )
        sub_df = query_result

        first_series = sameness_check(sub_df, osm_id, db_table_name)
        if first_series is None:
            raise ValueError(f"{db_table_name} data for ${osm_id} is not consistent")

        values = prepare_row(
            osm_id,
            columns,
            first_series,
        )

        insert_statement = f"""
        INSERT INTO {db_table_name}
            ({",".join([name_type[0] for name_type in columns])})
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
    lts_columns = [
        ("WAY_OSM_ID", "INTEGER", "osmid"),
        ("LTS", "INTEGER", "LTS"),
        ("LTS_FWD", "INTEGER", "LTS_fwd"),
        ("LTS_REV", "INTEGER", "LTS_rev"),
        ("BIKE_ACCESS", "INTEGER", "LTS_bike_access"),
        ("BIKE_ACCESS_FWD", "INTEGER", "LTS_bike_access_fwd"),
        ("BIKE_ACCESS_REV", "INTEGER", "LTS_bike_access_rev"),
        ("SEPARATION_FWD", "INTEGER", "LTS_separation_fwd"),
        ("SEPARATION_REV", "INTEGER", "LTS_separation_rev"),
        ("MIXED_FWD", "INTEGER", "LTS_mixed_fwd"),
        ("MIXED_REV", "INTEGER", "LTS_mixed_rev"),
        ("BIKE_LANE_NO_PARKING_FWD", "INTEGER", "LTS_bikelane_noparking_fwd"),
        ("BIKE_LANE_NO_PARKING_REV", "INTEGER", "LTS_bikelane_noparking_rev"),
        ("BIKE_LANE_YES_PARKING_FWD", "INTEGER", "LTS_bikelane_yesparking_fwd"),
        ("BIKE_LANE_YES_PARKING_REV", "INTEGER", "LTS_bikelane_yesparking_rev"),
    ]

    db_insert_dataframe(
        df.get([column[2] for column in lts_columns[1:]]),
        osm_ids,
        lts_columns,
        "LEVEL_OF_TRAFFIC_STRESS",
        cursor,
    )


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
    columns: list[tuple[str, str, str]],
    db_table_name: str,
    cursor: Cursor,
):
    """Insert all the rows in the DataFrame into the database for a table

    Parameters:
    df (DataFrame): DataFrame with all the LTS-annotated Way information
    osm_ids (set[int]): OSM IDs of Ways information belongs to
    columns (list[tuple[str, str, str]]): A list of tuples associating database column names with data types and DataFrame column labels
    db_table_name (str): Table to insert data into
    cursor (Cursor): Cursor to connect to the database with

    """

    osm_ids_to_load = find_new_osm_ids(osm_ids, db_table_name, columns[0], cursor)

    done = 0

    for osm_id in osm_ids_to_load:
        query_result = df.query(f"osmid == {osm_id}")
        sub_df = query_result.get([column[2] for column in columns[1:]])

        first_series = sameness_check(sub_df, osm_id, db_table_name)
        if first_series is None:
            raise ValueError(f"{db_table_name} data for ${osm_id} is not consistent")

        db_insert(db_table_name, osm_id, columns, first_series, cursor)

        done += 1
        if done % 1000 == 0:
            print(f"Analyzed and inserted {db_table_name} rows for {done} OSM_WAY_IDs")


def db_insert(table_name, id, columns, series, cursor):
    values = prepare_row(
        id,
        columns,
        series,
    )

    insert_statement = f"""
    INSERT INTO {table_name}
        ({",".join([column[0] for column in columns])})
        VALUES({",".join(values)})
    """

    cursor.execute(insert_statement)


def find_new_osm_ids(
    osm_ids: set[int],
    table_name: str,
    id_column: tuple[str, str, str],
    cursor: Cursor,
) -> set[int]:
    """Find OSM IDs that have already been inserted and subtract them from the set of OSM IDs to load

    Parameters:
    osm_ids (set[int]): OSM IDs of Ways information belongs to
    table_name (str): Table to insert data into
    id_column (tuple[str, str, str]): A tuples associating database ID column name with data type
    cursor (Cursor): Cursor to connect to the database with

    Returns:
    set[int]: A set of OSM IDs that do not exist in the database

    """
    print(
        f"Found {len(osm_ids)} candidate OSM_IDs to insert into the {table_name} table"
    )
    osm_ids_from_db_result = cursor.execute(f"SELECT {id_column[0]} FROM {table_name}")
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


def insert_nodes(node_data_file: str, cursor: Cursor):
    """Insert Nodes into the NODE table of the database.
    Also inserts relations between WAY and NODE via WAY_NODE.
    WAY table _must_ be populated before this function is run, or foreign-key errors will occur.

    Parameters:
    node_data_file (str): File containing Node OSM data
    cursor (Cursor): Cursor to connect to the database with
    """
    with open(node_data_file, "r") as f:
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

    node_columns = [
        ("OSM_ID", "INTEGER", "id"),
        ("LATITUDE", "REAL", "lat"),
        ("LONGITUDE", "REAL", "lon"),
        ("HIGHWAY", "TEXT", "tags.highway"),
        ("TRAFFIC_CALMING", "TEXT", "tags.traffic_calming"),
        ("CROSSING", "TEXT", "tags.crossing"),
        ("CROSSING_MARKINGS", "TEXT", "tags.crossing:markings"),
        ("CROSSING_ISLAND", "TEXT", "tags.crossing:island"),
    ]

    df_labels = set(nodes_df.columns.values)
    node_labels_that_exist = set.intersection(
        set([column[2] for column in node_columns]), df_labels
    )

    node_ids = set(nodes_df.id.array)
    node_ids_to_load = find_new_osm_ids(node_ids, "NODE", node_columns[0], cursor)

    way_ids_in_db = set(
        [row[0] for row in cursor.execute("SELECT OSM_ID FROM WAY").fetchall()]
    )

    done = 0
    for _, node in nodes_df.get(list(node_labels_that_exist)).iterrows():
        node_id = node.at["id"]
        if node_id not in node_ids_to_load:
            continue

        db_insert(
            "NODE",
            node_id,
            node_columns,
            node.get(list(node_labels_that_exist)),
            cursor,
        )

        done += 1
        if done % 1000 == 0:
            print(f"Analyzed and inserted NODE rows for {done} OSM_NODE_IDs")

    way_node_done = 0
    way_nodes_in_db = set(
        cursor.execute("SELECT WAY_OSM_ID, NODE_OSM_ID FROM WAY_NODE").fetchall()
    )
    for _, way in ways_df.iterrows():
        way_id = way.at["id"]
        if way_id not in way_ids_in_db:
            continue
        for position, way_node_id in enumerate(way.at["nodes"]):
            if (way_id, way_node_id) in way_nodes_in_db:
                continue
            insert_way_node_statement = f"""
                INSERT INTO WAY_NODE
                    (WAY_OSM_ID, NODE_OSM_ID, POSITION)
                VALUES({way_id}, {way_node_id}, {position})
                ON CONFLICT DO NOTHING
            """
            cursor.execute(insert_way_node_statement)
            way_node_done += 1
            if way_node_done % 1000 == 0:
                print(
                    f"Analyzed and inserted WAY_NODE rows for {way_node_done} OSM_WAY_IDs"
                )


if __name__ == "__main__":
    main()
