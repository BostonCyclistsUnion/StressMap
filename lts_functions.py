import re
import yaml

import pandas as pd
import numpy as np

SIDES = ['left', 'right']
DIRS = ['fwd', 'rev']

# %% Read Configuration Files
def read_tables():
    with open('config/tables.yml', 'r') as yml_file:
        tables = yaml.safe_load(yml_file)
    return tables

def read_rating():
    with open('config/rating_dict.yml', 'r') as yml_file:
        rating_dict = yaml.safe_load(yml_file)
    return rating_dict

def read_parse():
    with open('config/lane_parse.yml', 'r') as yml_file:
        parse_dict = yaml.safe_load(yml_file)
    return parse_dict

# %% Reused Functions

def apply_rules(gdf_edges, rating_dict, prefix):
    def apply_rule(SYM, LEFT, RIGHT):
        sides = set()
        if SYM:
            sides.add('')
        if LEFT:
            sides.add('_left')
        if RIGHT:
            sides.add('_right')

        for side in sides:
            try:
                gdf_filter = gdf_edges.eval(f"{condition} & (`{prefix}_condition{side}` == 'default')")
                gdf_edges.loc[gdf_filter, f'{prefix}{side}'] = value[prefix]
                gdf_edges.loc[gdf_filter, f'{prefix}_rule_num{side}'] = key
                gdf_edges.loc[gdf_filter, f'{prefix}_rule{side}'] = value['rule_message']
                gdf_edges.loc[gdf_filter, f'{prefix}_condition{side}'] = condition
                if 'LTS' in value:
                    gdf_edges.loc[gdf_filter, f'LTS_{prefix}{side}'] = value['LTS']
            
            except pd.errors.UndefinedVariableError as e:
                print(f'Column used in condition does not exsist in this region:\n\t{e}')

    rules = {k:v for (k,v) in rating_dict.items() if prefix in k}
        
    for key, value in rules.items():
        # Check rules in order, once something has been updated, leave it be

        condition = value['condition']
        namespace = re.findall(r'\[(.*)\]', condition)
        if len(namespace) > 0:
            namespaceSplit = namespace[0].split(',')
            if len(namespaceSplit) > 0:
                for namespaceVal in namespaceSplit:
                    condition = value['condition'].replace('[' + namespace[0] + ']', namespaceVal)
                    if namespaceVal == 'both':
                        LEFT = True
                        RIGHT = True
                    elif namespaceVal == 'left':
                        LEFT = True
                        RIGHT = False
                    elif namespaceVal == 'right':
                        LEFT = False
                        RIGHT = True
                    else:
                        print(namespaceVal)
                    SYM = False
                    apply_rule(SYM, LEFT, RIGHT)
        elif prefix in ['bike_lane_exist', 'biking_permitted', 'bike_lane_separation']:
            condition = value['condition']
            SYM = False
            LEFT = True
            RIGHT = True
            apply_rule(SYM, LEFT, RIGHT)
        else:
            condition = value['condition']
            SYM = True
            LEFT = False
            RIGHT = False
            apply_rule(SYM, LEFT, RIGHT)
    

    # Save memory by setting as category, need to set categories first
    if SYM:
        cols = [
            # prefix,
            f'{prefix}_rule_num',
            f'{prefix}_condition', 
            f'{prefix}_rule',
            ]
    else:
        cols = [
            # prefix,
            f'{prefix}_rule_num_left',
            f'{prefix}_condition_left', 
            f'{prefix}_rule_left',
            f'{prefix}_rule_num_right',
            f'{prefix}_condition_right', 
            f'{prefix}_rule_right',
            ]

    for col in cols:
        try:
            gdf_edges[col] = gdf_edges[col].astype('category')
        except KeyError as e:
            print(f'Key error attempting to set column as category: {e}')

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

# %% Lane Direction Parsing
def convert_both_tag(gdf_edges):
    '''
    For all columns that have a *:both suffix, set the value of the *:left and *:right columns to
    both equal the value of the *:both column. 

    This allows all further processing to ignore the *:both suffix and only use the sided suffixes.
    
    If there is a way that has both *:both and *:left/*:right columns, the side columns will be 
    overwritten. In this case, it is indeterminate which is correct and should be fixed in OSM. This 
    choice is due to programming ease.

    FUTURE: Create a report of ways where there is overlapping *:both and *:left/*:right columns
    '''

    # Move tags with *:both suffix to both *:left/*:right suffix columns
    tags = gdf_edges.columns[gdf_edges.columns.str.contains('both')]
    for tag in tags:
        tag_left = tag.replace('both', 'left')
        tag_right = tag.replace('both', 'right')

        gdf_filter = gdf_edges.loc[~gdf_edges[tag].isna()]
        gdf_edges.loc[gdf_filter.index, tag_left] = gdf_filter[tag]
        gdf_edges.loc[gdf_filter.index, tag_right] = gdf_filter[tag]
    # Remove *:both columns to prevent accidental usage
    gdf_edges = gdf_edges.drop(columns=tags)

    # Convert tags implicit with *:both suffix to both *:left/*:right suffix columns
    tags = ['cycleway', 'cycleway:buffer', 'cycleway:separation', 'cycleway:width']
    for tag in tags:
        tag_left = tag + ':left'
        tag_right = tag + ':right'

        try:
            gdf_filter = gdf_edges.loc[~gdf_edges[tag].isna()]
            gdf_edges.loc[gdf_filter.index, tag_left] = gdf_filter[tag]
            gdf_edges.loc[gdf_filter.index, tag_right] = gdf_filter[tag]

            # Remove column to prevent accidental usage
            gdf_edges = gdf_edges.drop(columns=[tag])
        except KeyError:
            print(f'No {tag} column')
    

    # Merge direction suffixes
    tagsPairs = [
        ['cycleway:left:buffer',  'cycleway:buffer:left'],
        ['cycleway:right:buffer', 'cycleway:buffer:right'],
        ['cycleway:left:separation',  'cycleway:separation:left'],
        ['cycleway:right:separation', 'cycleway:separation:right'],
        ['cycleway:left:width',  'cycleway:width:left'],
        ['cycleway:right:width', 'cycleway:width:right'],
        ]
    for pairs in tagsPairs:
        if pairs[1] in gdf_edges.columns:
            if pairs[0] in gdf_edges.columns:
                gdf_filter = gdf_edges.loc[~gdf_edges[pairs[0]].isna()]
                gdf_edges.loc[gdf_filter.index, pairs[0]] = gdf_filter[pairs[1]]
            else:
                gdf_edges.loc[gdf_edges.index, pairs[0]] = gdf_edges[pairs[1]]
            # Remove columns to prevent accidental usage
            gdf_edges = gdf_edges.drop(columns=pairs[1])

    return gdf_edges

def parse_lanes(gdf_edges):
    '''
    Parse which side of the street bike lanes are based on OSM tags and which direction they travel.
    Then coorelate street features to the respective direction of bike travel. 
    '''
    parse_dict = read_parse()

    gdf_edges['parse'] = ''
    gdf_edges['LTS_bike_access'] = np.nan
    gdf_edges['LTS_bike_access_fwd'] = np.nan
    gdf_edges['LTS_bike_access_rev'] = np.nan

    cols = ['bike_allowed_fwd', 'bike_allowed_rev',
            'bike_lane_fwd', 'bike_lane_rev', 
            'parking_fwd', 'parking_rev',
            'parking_width_fwd', 'parking_width_rev',
            'buffer_fwd', 'buffer_rev',
            'bike_width_fwd', 'bike_width_rev',
            'separation_fwd', 'separation_rev']
    for key in cols:
        # gdf_edges[key] = 'not evaluated'
        gdf_edges[key] = np.nan
        gdf_edges[key] = gdf_edges[key].astype(object)

    logdf = pd.DataFrame(columns=['condition'] + cols)
    for key, value in parse_dict.items():
        condition = value['condition']
        print(f'Processing condition {key}: {condition}')
        logdf.loc[key, 'condition'] = condition
        try:
            # gdf_filter = gdf_edges.eval(f"{condition} & (`parse` == 'not evaluated')")
            gdf_filter = gdf_edges.eval(condition)
            gdf_edges.loc[gdf_filter, 'parse'] = gdf_edges.loc[gdf_filter, 'parse'].astype(str) + condition + '\n'
            if 'LTS' in value:
                gdf_edges.loc[gdf_edges['LTS_bike_access_fwd'].isna() & gdf_filter, 'LTS_bike_access_fwd'] = value['LTS']
                gdf_edges.loc[gdf_edges['LTS_bike_access_rev'].isna() & gdf_filter, 'LTS_bike_access_rev'] = value['LTS']
            if 'LTS_fwd' in value:
                gdf_edges.loc[gdf_edges['LTS_bike_access_fwd'].isna() & gdf_filter, 'LTS_bike_access_fwd'] = value['LTS_fwd']
            if 'LTS_rev' in value:
                gdf_edges.loc[gdf_edges['LTS_bike_access_rev'].isna() & gdf_filter, 'LTS_bike_access_rev'] = value['LTS_rev']
            for col in cols:
                # print(f'\t{col}')
                if col in value:
                    gdf_uneval = gdf_filter & gdf_edges[col].isna()
                    logdf.loc[key, col] = gdf_uneval.values.sum()
                    if isinstance(value[col], bool):
                        gdf_edges.loc[gdf_uneval[gdf_uneval].index, col] = value[col]
                    else:
                        gdf_edges.loc[gdf_uneval[gdf_uneval].index, col] = gdf_edges.loc[gdf_uneval[gdf_uneval].index, value[col]]
        
        except pd.errors.UndefinedVariableError as e:
            print(f'\tColumn used in condition does not exsist in this region:\n\t\t{e}')
        except KeyError as e:
            print(f'\tColumn does not exsist in this region: {e}')

    logdf.to_csv('data/log_parse.csv')
    # gdf_edges.loc[gdf_edges['bike_allowed_fwd'].isna()].to_csv('data/log_bike_allowed_fwd_na.csv')
    # gdf_edges.loc[gdf_edges['bike_allowed_rev'].isna()].to_csv('data/log_bike_allowed_rev_na.csv')

    return gdf_edges

# %% Pre-Processing Functions

def parking_present(gdf_edges, rating_dict):
    """
    Detect where parking is and isn't allowed.
    """
    # tags = gdfEdges.columns[gdfEdges.columns.str.contains('parking')]
    # for tag in tags.sort_values():
    #     print(tag, gdfEdges[tag].unique())

    prefix = 'parking'
    defaultRule = f'{prefix}0'

    for side in SIDES:
        gdf_edges[f'{prefix}_{side}'] = 'yes'
        gdf_edges[f'{prefix}_rule_num_{side}'] = defaultRule
        gdf_edges[f'{prefix}_rule_{side}'] = 'Assumed'
        gdf_edges[f'{prefix}_condition_{side}'] = 'default'

    gdf_edges = apply_rules(gdf_edges, rating_dict, prefix)

    for side in SIDES:
        gdf_edges[f'parking_width_{side}'] = 0.0
        gdf_edges.loc[gdf_edges[f'{prefix}_{side}']=='yes', f'parking_width_{side}'] = 8.5 # ft
        gdf_edges.loc[gdf_edges[f'{prefix}_{side}']=='yes', f'parking_width_rule_{side}'] = 'Assumed'

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
    gdf_edges['speed_rule'] = 'Signed speed limit'
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
    gdf_edges['lane_count'] = gdf_edges['lanes'].fillna(default_lanes).apply( # FIXME: change so footways default to 1 lane, maybe other road types have other defaults
        lambda x: np.array(re.split(r'; |, |\*|\n', str(x)), dtype = 'int')).apply( 
            # Converted to a raw string to avoid 'SyntaxWarning: invalid escape sequence '\*' python re',
            # check that this is doing the right thing
            lambda x: np.max(x))
    
    gdf_edges['lane_rule'] = 'OSM'
    assumed = gdf_edges['lanes'] == np.nan
    gdf_edges.loc[assumed, 'lane_rule'] = 'Assumed'

    return gdf_edges

def get_centerlines(gdf_edges, rating_dict):

    prefix = 'centerline'
    defaultRule = f'{prefix}0'

    # for side in SIDES:
    gdf_edges[f'{prefix}'] = 'yes'
    gdf_edges[f'{prefix}_rule_num'] = defaultRule
    gdf_edges[f'{prefix}_rule'] = 'Assumed'
    gdf_edges[f'{prefix}_condition'] = 'default'

    gdf_edges = apply_rules(gdf_edges, rating_dict, prefix)

    return gdf_edges

def width_ft(gdf_edges):
    '''
    Convert OSM width columns to use decimal feet
    '''
    gdf_edges['width_street'], gdf_edges['width_street_rule'] = convert_feet_with_quotes(gdf_edges['width'])

    try:
        for dir in DIRS:
            width_bikelane, width_bikelane_rule = convert_feet_with_quotes(gdf_edges[f'bike_width_{dir}'])
            gdf_edges.loc[width_bikelane.notna(), f'bike_width_{dir}'] = width_bikelane
            gdf_edges.loc[width_bikelane.notna(), f'bike_width_rule_{dir}'] = width_bikelane_rule
    except KeyError:
        print(f'No bike_width_{dir} column')
    
    # Default values
    for dir in DIRS:
        gdf_edges.loc[gdf_edges[f'bike_width_{dir}'].isna(), f'bike_width_{dir}'] = 5.0
        gdf_edges.loc[gdf_edges[f'bike_width_{dir}'].isna(), f'bike_width_rule_{dir}'] = 'Assumed'
        gdf_edges.loc[gdf_edges[f'bike_width_{dir}'].isna(), f'buffer_{dir}'] = 0.0
        gdf_edges.loc[gdf_edges[f'bike_width_{dir}'].isna(), f'buffer_rule_{dir}'] = 'Assumed'

    try:
        for dir in DIRS:
            if 'yes' in gdf_edges[f'buffer_{dir}'].values:
                gdf_edges.loc[gdf_edges[f'buffer_{dir}']=='yes', f'buffer_{dir}'] = "2'"
            if 'no' in gdf_edges[f'buffer_{dir}'].values:
                gdf_edges.loc[gdf_edges[f'buffer_{dir}']=='yes', f'buffer_{dir}'] = "0.0"
            width_bikelanebuffer, width_bikelanebuffer_rule = convert_feet_with_quotes(gdf_edges[f'buffer_{dir}'])
            gdf_edges.loc[width_bikelanebuffer.notna(), f'buffer_{dir}'] = width_bikelanebuffer
            gdf_edges.loc[width_bikelanebuffer.notna(), f'buffer_rule_{dir}'] = width_bikelanebuffer_rule
    except KeyError:
        print(f'No buffer_{dir} column')

    for dir in DIRS:
        gdf_edges[f'bike_reach_{dir}'] =    gdf_edges[f'bike_width_{dir}'].fillna(0) + \
                                            gdf_edges[f'parking_width_{dir}'].fillna(0) + \
                                            gdf_edges[f'buffer_{dir}'].fillna(0)

    return gdf_edges

def define_narrow_wide(gdf_edges):
    gdf_edges['street_narrow_wide'] = 'not oneway'

    gdf_edges.loc[(gdf_edges['oneway']), 'street_narrow_wide'] = 'wide'

    gdf_edges.loc[(gdf_edges['oneway']) & 
                  (gdf_edges['width_street'] < 30) & 
                  (gdf_edges['parking_fwd'] == 'yes') & 
                  (gdf_edges['parking_rev'] == 'yes'), 'street_narrow_wide'] = 'narrow'

    for dir in DIRS:
        gdf_edges.loc[(gdf_edges['oneway']) & (gdf_edges['width_street'] < 22) & (gdf_edges[f'parking_{dir}'] == 'yes'), 'street_narrow_wide'] = 'narrow'

    gdf_edges.loc[(gdf_edges['oneway']) & 
                  (gdf_edges['width_street'] < 15) & 
                  (gdf_edges['parking_fwd'] == 'no') & 
                  (gdf_edges['parking_rev'] == 'no'), 'street_narrow_wide'] = 'narrow'

    return gdf_edges

def define_adt(gdf_edges, rating_dict):
    '''
    Add the Average Daily Traffic (ADT) value to each segment to use the right row of LTS tables.

    Use assumptions based on roadway type. 

    FUTURE: Get ADT measurements from cities or Streetlight to improve values.
    '''

    prefix = 'ADT'
    defaultRule = f'{prefix}0'

    gdf_edges[f'{prefix}'] = 1500 # FIXME is this the right default?
    gdf_edges[f'{prefix}_rule_num'] = defaultRule
    gdf_edges[f'{prefix}_rule'] = 'Assumed'
    gdf_edges[f'{prefix}_condition'] = 'default'

    gdf_edges = apply_rules(gdf_edges, rating_dict, prefix)

    return gdf_edges

# %% LTS Calculations
def column_value_counts(gdf_edges):
    '''
    This is a debugging function. Save what values and their quantities are in the data for each 
    filter column.
    '''
    cols = ['bike_allowed_dir', 'centerline', 'lane_count', 'oneway', 'street_narrow_wide', 
            'bike_lane_dir', 'parking_dir', 'ADT', 'bike_width_dir', 'bike_reach_dir']
    cols_vc = []

    for col in cols:
        if 'dir' in col:
            cols_vc.append(col.replace('dir', 'fwd'))
            cols_vc.append(col.replace('dir', 'rev'))
        else:
            cols_vc.append(col)

    vc_df = pd.DataFrame()
    for col in cols_vc:
        vc = gdf_edges[col].value_counts(dropna=False)
        if len(vc_df) > len(vc):
            dif = len(vc_df) - len(vc)
            vc = pd.concat([vc, pd.Series(['_'] * dif)])
        elif len(vc_df) < len(vc):
            dif = len(vc) - len(vc_df)
            vc_df = pd.concat([vc_df, pd.DataFrame([['_']*vc_df.shape[1]]*dif, columns=vc_df.columns)], ignore_index=True)

        vc_df[f'{col}_values'] = vc.index
        vc_df[f'{col}_counts'] = vc.values

    vc_df.to_csv('data/log_filter_column_counts.csv')
    print('Saved columns values and counts of filters')

def evaluate_lts_table(gdf_edges, tables, tableName):
    baseName = tableName[6:]
    table = tables[tableName]
    print(f'Evalutating LTS use {baseName} table...')

    subTables = [key for key in table.keys() if tableName in key]
    # print(subTables)

    speedMin = tables['cols_speeds']['min']
    speedMax = tables['cols_speeds']['max']

    for dir in DIRS:
        gdf_edges[f'LTS_{baseName}_{dir}'] = np.nan

    # conditionTable = table['conditions']

    logdf = pd.DataFrame(columns=['subTable', 'conditionTableStr', 'conditionName', 'dir', 'condition', 'lts', 'count'])

    # Each LTS table split by number of lane classifications
    for subTable in subTables:
        # print(f'\n{subTable=}')
        # print(f'{table[subTable]['conditions']=}')
        for conditionTableName in table['conditions']:
            conditionTableStr = table['conditions'][conditionTableName]
            for dir in DIRS:
                conditionTable = conditionTableStr.replace('dir', dir)
                # print(conditionTable)
                for conditionName in table[subTable]['conditions']:
                    bucketColumn = table['bucketColumn']
                    bucketTable = table[subTable][f'table_{bucketColumn}'.replace('_dir', '')]
                    ltsSpeeds = table[subTable]['table_speed']
                    # print(f'{subTable=} | {conditionTable=} | {conditionName=}')
                    bucketColumn = bucketColumn.replace('dir', dir)
                    for bucket, ltsSpeed in zip(bucketTable, ltsSpeeds):
                        conditionBucket = f'(`{bucketColumn}` >= {bucket[0]}) & (`{bucketColumn}` < {bucket[1]})'
                        for sMin, sMax, lts in zip(speedMin, speedMax, ltsSpeed):
                            condition = table[subTable]['conditions'][conditionName]
                            condition = condition.replace('dir', dir)
                            conditionSpeed = f'(`speed` > {sMin}) & (`speed` < {sMax})'
                            condition = f'{condition} & {conditionSpeed} & {conditionBucket} & {conditionTable}'
                            # print(f'\t{conditionName} | {condition}')
                            gdf_filter = gdf_edges.eval(f"{condition}")
                            # print(f'{baseName}: LTS={lts}\n{condition}\n{gdf_filter.value_counts()}\n')
                            gdf_edges.loc[gdf_filter, f'LTS_{baseName}_{dir}'] = lts

                            logdf.loc[len(logdf)] = [subTable, conditionTableStr, conditionName, dir, condition, lts, gdf_filter.values.sum()]
                    # gdf_edges.loc[gdf_filter, f'{prefix}_rule_num'] = key

    logdf.to_csv(f'data/log_lts_{baseName}.csv')

    return gdf_edges

def calculate_lts(gdf_edges, tables):
    tablesList = [key for key in tables.keys() if 'table_' in key]

    for tableName in tablesList:
        gdf_edges = evaluate_lts_table(gdf_edges, tables, tableName)

    # Use the lowest calculated LTS score for a segment (in case mixed is lower than bike lane)
    for dir in DIRS:
        gdf_edges[f'LTS_{dir}'] = gdf_edges.loc[:,( gdf_edges.columns.str.startswith('LTS') & gdf_edges.columns.str.endswith(dir))].min(axis=1, skipna=True, numeric_only=True)

    gdf_edges['LTS'] = gdf_edges[['LTS_fwd', 'LTS_rev']].max(axis=1)

    return gdf_edges


