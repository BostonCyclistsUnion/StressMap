import re
import yaml

import pandas as pd
import numpy as np

# %% Read Configuration Files
def read_tables():
    with open('config/tables.yml', 'r') as yml_file:
        tables = yaml.safe_load(yml_file)
    return tables

def read_rating():
    with open('config/rating_dict.yml', 'r') as yml_file:
        rating_dict = yaml.safe_load(yml_file)
    return rating_dict

# %% Reused Functions

def apply_rules(gdf_edges, rating_dict, prefix):
    rules = {k:v for (k,v) in rating_dict.items() if prefix in k}

    for key, value in rules.items():
        # Check rules in order, once something has been updated, leave it be
        # FIXME gracefully handle if condition is not found
        # FIXME need to handle single sided tags so that can include both sides in outputs
        # print(key)
        try:
            gdf_filter = gdf_edges.eval(f"{value['condition']} & (`{prefix}_condition` == 'default')")
            gdf_edges.loc[gdf_filter, prefix] = value[prefix]
            gdf_edges.loc[gdf_filter, f'{prefix}_rule_num'] = key
            gdf_edges.loc[gdf_filter, f'{prefix}_rule'] = value['rule_message']
            gdf_edges.loc[gdf_filter, f'{prefix}_condition'] = value['condition']
            if 'LTS' in value:
                gdf_edges.loc[gdf_filter, f'LTS_{prefix}'] = value['LTS']
        except pd.errors.UndefinedVariableError as e:
            print(f'Column used in condition does not exsist in this region:\n\t{e}')

    # Save memory by setting as category, need to set categories first
    for col in [
        # prefix,
        f'{prefix}_rule_num',
        f'{prefix}_condition',
        f'{prefix}_rule',
        ]:

        gdf_edges[col] = gdf_edges[col].astype('category')

    return gdf_edges

def convert_feet_with_quotes(series):
    series = series.copy()
    # Calculate decimal feet and inches when each given separately
    quoteValues = series.str.contains('\'')
    meterValues = quoteValues == False

    quoteValues[quoteValues.isna()] = False
    quoteValues = quoteValues.astype(bool)

    feetinch = series[quoteValues].str.strip('"').str.split('\'', expand=True)
    if feetinch.shape[0] > 0:
        feetinch.loc[feetinch[1] == '', 1] = 0
        feetinch = feetinch.apply(lambda x: np.array(x, dtype = 'int'))
    # if feetinch.shape[0] > 0:
        feet = feetinch[0] + feetinch[1] / 12
        series[quoteValues] = feet

    # Use larger value if given multiple
    multiWidth = series.str.contains(';', na=False)

    maxWidth = series[multiWidth].str.split(';', expand=True).fillna(value=np.nan).astype(float).max(axis=1)
    series[multiWidth] = maxWidth

    series = pd.to_numeric(series, errors='coerce')
    # series = series.apply(lambda x: np.array(x, dtype = 'float'))

    # Convert (assumed) meter values to feet
    series[meterValues] = series[meterValues].astype(float) * 3.28084

    series_notes = pd.Series('No Width', index=series.index)
    series_notes[quoteValues] = 'Converted ft-in to decimal feet'
    series_notes[meterValues] = 'Converted m to feet'

    return series, series_notes

# %% Pre-Processing Functions

def biking_permitted(gdf_edges, rating_dict):
    prefix = 'biking_permitted'
    defaultRule = f'{prefix}0'

    gdf_edges[prefix] = 'yes'
    gdf_edges[f'{prefix}_rule_num'] = defaultRule
    gdf_edges[f'{prefix}_rule'] = 'Assume biking permitted'
    gdf_edges[f'{prefix}_condition'] = 'default'

    gdf_edges = apply_rules(gdf_edges, rating_dict, prefix)

    return gdf_edges

def is_separated_path(gdf_edges, rating_dict):
    prefix = 'bike_lane_separation'
    defaultRule = f'{prefix}0'

    gdf_edges[prefix] = 'no'
    gdf_edges[f'{prefix}_rule_num'] = defaultRule
    gdf_edges[f'{prefix}_rule'] = 'Assume no bike lane separation'
    gdf_edges[f'{prefix}_condition'] = 'default'

    # get the columns that start with 'cycleway'
    # tags = gdf_edges.columns[gdf_edges.columns.str.contains('cycleway')]
    # for tag in tags:
    #     print(tag, gdfEdges[tag].unique())

    gdf_edges = apply_rules(gdf_edges, rating_dict, prefix)

    return gdf_edges

def is_bike_lane(gdf_edges, rating_dict):
    """
    Check if there's a bike lane, use road features to assign LTS
    """
    # get the columns that start with 'cycleway'
    # tags = gdf_edges.columns[gdf_edges.columns.str.contains('cycleway')]
    # for tag in tags:
    #     print(tag, gdfEdges[tag].unique())

    prefix = 'bike_lane_exist'
    defaultRule = f'{prefix}0'

    gdf_edges[prefix] = 'no'
    gdf_edges[f'{prefix}_rule_num'] = defaultRule
    gdf_edges[f'{prefix}_rule'] = 'Assume no bike lane'
    gdf_edges[f'{prefix}_condition'] = 'default'

    gdf_edges = apply_rules(gdf_edges, rating_dict, prefix)

    return gdf_edges

def parking_present(gdf_edges, rating_dict):
    """
    Detect where parking is and isn't allowed.
    """
    # tags = gdfEdges.columns[gdfEdges.columns.str.contains('parking')]
    # for tag in tags.sort_values():
    #     print(tag, gdfEdges[tag].unique())

    prefix = 'parking'
    defaultRule = f'{prefix}0'

    gdf_edges[prefix] = 'yes'
    gdf_edges[f'{prefix}_rule_num'] = defaultRule
    gdf_edges[f'{prefix}_rule'] = 'Assume street parking is allowed on both sides'
    gdf_edges[f'{prefix}_condition'] = 'default'

    gdf_edges = apply_rules(gdf_edges, rating_dict, prefix)


    gdf_edges['width_parking'] = 0.0
    gdf_edges.loc[gdf_edges[prefix]=='yes', 'width_parking'] = 8.5 # ft

    return gdf_edges

def get_prevailing_speed(gdf_edges, rating_dict):
    """
    Get the speed limit for ways
    If not available, make assumptions based on road type
    This errs on the high end of assumptions
    """
    prefix = 'speed'
    speedRules = {k:v for (k,v) in rating_dict.items() if prefix in k}
    defaultRule = f'{prefix}0'

    # FIXME if change to apply assumed values first then replace with OSM data, can use common function
    gdf_edges['speed'] = gdf_edges['maxspeed']
    gdf_edges['speed'] = gdf_edges['speed'].fillna(0)
    gdf_edges.loc[gdf_edges['speed'] == 'signals', 'speed'] = 0
    gdf_edges['speed_rule_num'] = defaultRule
    gdf_edges['speed_rule'] = 'Use signed speed limits.'
    gdf_edges['speed_condition'] = 'default'

    for key, value in speedRules.items():
        # Check rules in order, once something has been updated, leave it be
        gdf_filter = gdf_edges.eval(f"{value['condition']} & (`speed` == 0)")
        gdf_edges.loc[gdf_filter, 'speed'] = value['speed']
        gdf_edges.loc[gdf_filter, 'speed_rule_num'] = key
        gdf_edges.loc[gdf_filter, 'speed_rule'] = value['rule_message']
        gdf_edges.loc[gdf_filter, 'speed_condition'] = value['condition']

    # If mph
    if gdf_edges[gdf_edges['speed'].astype(str).str.contains('mph')].shape[0] > 0:
        mph = gdf_edges['speed'].astype(str).str.contains('mph', na=False)
        gdf_edges.loc[mph, 'speed'] = gdf_edges['speed'][mph].str.split(
            ' ', expand=True)[0].apply(lambda x: np.array(x, dtype = 'int'))

    # # if multiple speed values present, use the largest one
    # gdf_edges['maxspeed_assumed'] = gdf_edges['maxspeed_assumed'].apply(
    #     lambda x: np.array(x, dtype = 'int')).apply(lambda x: np.max(x))

    # Make sure all speeds are numbers
    gdf_edges['speed'] = gdf_edges['speed'].astype(int)
    # Save memory by setting as category, need to set categories first
    for col in ['speed_rule_num', 'speed_rule', 'speed_condition']:
        gdf_edges[col] = gdf_edges[col].astype('category')

    return gdf_edges

def get_lanes(gdf_edges, default_lanes = 2):

    # make new assumed lanes column for use in calculations

    # fill na with default lanes
    # if multiple lane values present, use the largest one
    # this usually happens if multiple adjacent ways are included in the edge and 
    # there's a turning lane
    gdf_edges['lane_count'] = gdf_edges['lanes'].fillna(default_lanes).apply(
        lambda x: np.array(re.split(r'; |, |\*|\n', str(x)), dtype = 'int')).apply(
            # Converted to a raw string to avoid 'SyntaxWarning: invalid escape sequence '\*' python re',
            # check that this is doing the right thing
            lambda x: np.max(x))

    gdf_edges['lane_source'] = 'OSM'
    assumed = gdf_edges['lanes'] == np.nan
    gdf_edges.loc[assumed, 'lane_source'] = 'Assumed lane count'

    return gdf_edges

def get_centerlines(gdf_edges, rating_dict):

    prefix = 'centerline'
    defaultRule = f'{prefix}0'

    gdf_edges[prefix] = 'yes'
    gdf_edges[f'{prefix}_rule_num'] = defaultRule
    gdf_edges[f'{prefix}_rule'] = 'Assume centerlines.'
    gdf_edges[f'{prefix}_condition'] = 'default'

    gdf_edges = apply_rules(gdf_edges, rating_dict, prefix)

    return gdf_edges

def width_ft(gdf_edges):
    '''
    Convert OSM width columns to use decimal feet
    '''

    gdf_edges['width_street'], gdf_edges['width_street_notes'] = convert_feet_with_quotes(gdf_edges['width'])

    try:
        gdf_edges['width_bikelane'], gdf_edges['width_bikelane_notes'] = convert_feet_with_quotes(gdf_edges['cycleway:width'])
    except KeyError:
        print('No cycleway:width column')
        gdf_edges['width_bikelane'] = 0.0
        gdf_edges['width_bikelane_notes'] = 'No cycleway:width column'

    try:
        if 'yes' in gdf_edges['cycleway:buffer'].values:
            gdf_edges.loc[gdf_edges['cycleway:buffer']=='yes','cycleway:buffer'] = "2'"
        if 'no' in gdf_edges['cycleway:buffer'].values:
            gdf_edges.loc[gdf_edges['cycleway:buffer']=='yes','cycleway:buffer'] = "0.0"
        gdf_edges['width_bikelanebuffer'], gdf_edges['width_bikelanebuffer_notes'] = convert_feet_with_quotes(gdf_edges['cycleway:buffer'])
    except KeyError:
        print('No cycleway:buffer column')
        gdf_edges['width_bikelanebuffer'] = 0.0
        gdf_edges['width_bikelanebuffer_notes'] = 'No cycleway:buffer column'

    # FIXME make this work for asymmetric layouts
    gdf_edges['bikelane_reach'] = gdf_edges['width_bikelane'] + gdf_edges['width_parking'] + gdf_edges['width_bikelanebuffer']

    return gdf_edges

def define_narrow_wide(gdf_edges):

    gdf_edges['street_narrow_wide'] = 'wide'

    gdf_edges[(gdf_edges['oneway'] == 'True') & (gdf_edges['width_street'] < 30) & (gdf_edges['parking'] == 'yes')] = 'narrow'

    # FIXME make sure only single side has parking and not ignoring where one side is explicit yes and the other is explicit no
    gdf_edges[(gdf_edges['oneway'] == 'True') & (gdf_edges['width_street'] < 22) & (gdf_edges['parking'] == 'left:yes')] = 'narrow'
    gdf_edges[(gdf_edges['oneway'] == 'True') & (gdf_edges['width_street'] < 22) & (gdf_edges['parking'] == 'right:yes')] = 'narrow'

    gdf_edges[(gdf_edges['oneway'] == 'True') & (gdf_edges['width_street'] < 15) & (gdf_edges['parking'] == 'no')] = 'narrow'

    return gdf_edges

def define_adt(gdf_edges, rating_dict):
    '''
    Add the Average Daily Traffic (ADT) value to each segment to use the right row of LTS tables.

    Use assumptions based on roadway type.

    FUTURE: Get ADT measurements from cities or Streetlight to improve values.
    '''

    prefix = 'ADT'
    defaultRule = f'{prefix}0'

    gdf_edges[prefix] = 1500 # FIXME is this the right default?
    gdf_edges[f'{prefix}_rule_num'] = defaultRule
    gdf_edges[f'{prefix}_rule'] = 'Assume centerlines.'
    gdf_edges[f'{prefix}_condition'] = 'default'

    gdf_edges = apply_rules(gdf_edges, rating_dict, prefix)

    return gdf_edges

# %% LTS Calculations
def evaluate_lts_table(gdf_edges, tables, tableName):
    baseName = tableName[6:]
    table = tables[tableName]

    subTables = [key for key in table.keys() if tableName in key]
    print(subTables)

    speedMin = tables['cols_speeds']['min']
    speedMax = tables['cols_speeds']['max']

    gdf_edges[f'LTS_{baseName}'] = np.nan

    conditionTable = table['conditions']

    for subTable in subTables:
        # print(f'\n{subTable=}')
        # print(f'{table[subTable]['conditions']=}')
        for conditionTableName in table['conditions']:
            conditionTable = table['conditions'][conditionTableName]
            # print(conditionTable)
            for conditionName in table[subTable]['conditions']:
                bucketColumn = table['bucketColumn']
                bucketTable = table[subTable][f'table_{bucketColumn}']
                ltsSpeeds = table[subTable]['table_speed']
                for bucket, ltsSpeed in zip(bucketTable, ltsSpeeds):
                    conditionBucket = f'(`{bucketColumn}` >= {bucket[0]}) & (`{bucketColumn}` < {bucket[1]})'
                    for sMin, sMax, lts in zip(speedMin, speedMax, ltsSpeed):
                        condition = table[subTable]['conditions'][conditionName]
                        conditionSpeed = f'(`speed` > {sMin}) & (`speed` < {sMax})'
                        condition = f'{condition} & {conditionSpeed} & {conditionBucket} & {conditionTable}'
                        # print(f'\t{conditionName} | {condition}')
                        gdf_filter = gdf_edges.eval(f"{condition}")
                        gdf_edges.loc[gdf_filter, f'LTS_{baseName}'] = lts
                # gdf_edges.loc[gdf_filter, f'{prefix}_rule_num'] = key


    return gdf_edges

def calculate_lts(gdf_edges, tables):
    tablesList = [key for key in tables.keys() if 'table_' in key]

    for tableName in tablesList:
        gdf_edges = evaluate_lts_table(gdf_edges, tables, tableName)

    # Use the lowest calculated LTS score for a segment (in case mixed is lower than bike lane)
    gdf_edges['LTS'] = gdf_edges.loc[:, gdf_edges.columns.str.contains('LTS')].min(axis=1, skipna=True, numeric_only=True)

    return gdf_edges