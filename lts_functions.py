# import pytest
import pandas as pd
import numpy as np

# unit = 'metric'
unit = 'english'

ref =   {
        # Speeds
        's1': {
            'metric': 35,
            'english': 20
        },
        's2': {
            'metric': 40,
            'english': 25
            },
        's3': {
            'metric': 50,
            'english': 30
        },
        's4': {
            'metric': 55,
            'english': 35
        },
        's5': {
            'metric': 65,
            'english': 40
        },
        's6': {
            'metric': 80,
            'english': 50
        },
        's7': {
            'metric': 100,
            'english': 65
        },
        # Widths
        'w1': {
            'metric': 1.7,
            'english': 5.5
        },
        'w2': {
            'metric': 4.1,
            'english': 13.45
        },
        'w3': {
            'metric': 4.25,
            'english': 13.95
        },
        'w4': {
            'metric': 4.5,
            'english': 14.75
        },
        }

# Defaults
national = ref['s2'][unit] # 40
local = ref['s3'][unit] # 50
motorway = ref['s7'][unit] # 100
primary = ref['s6'][unit] # 80
secondary = ref['s6'][unit] # 80

def biking_permitted(gdf_edges):
    """
    Categorize ways as biking permitted or not permitted. 
    Checks if it's a sidewalk without a bicycle=yes tag,
    or if it's a highway,
    or if it's a proposed way that doesn't exist.
    These classification decisions are adapted from Bike Ottawa's
    stressmodel code: https://github.com/BikeOttawa/stressmodel/blob/master/stressmodel.js
    
    Inputs: 
    gdf_edges : a dataframe of network edges downloaded using the package osmnx
    
    Returns:
    gdf_allowed : a dataframe of edges after removing ways where cycling is not permitted
    gdf_not_allowed : a dataframe of edges where cycling is not permitted
    
    gdf_allowed and gdf_not_allowed together are the entire contents of gdf_edges.
    """
    conditions = [(gdf_edges['bicycle'] == 'no'), # p2
                  (gdf_edges['access'] == 'no'), # p6
                  (gdf_edges['highway'] == 'motorway'), # p3
                  (gdf_edges['highway'] == 'motorway_link'), #p4
                  (gdf_edges['highway'] == 'proposed'), # p7
                  ((gdf_edges['footway'] == 'sidewalk') & ~(gdf_edges['bicycle'] == 'yes')
                    & ((gdf_edges['highway'] == 'footway') | (gdf_edges['highway'] == 'path'))) # p5
                 ]


    values = ['p2', 'p6', 'p3', 'p4', 'p7', 'p5']

    # create a new column and use np.select to assign values to it using our lists as arguments
    gdf_edges['rule'] = np.select(conditions, values, default='p0')

    # filter based on rule assignment
    gdf_allowed = gdf_edges[gdf_edges['rule'] == 'p0']
    gdf_not_allowed = gdf_edges[~(gdf_edges['rule'] == 'p0')]

    # gdf_not_allowed = gdf_not_allowed

    return gdf_allowed, gdf_not_allowed

def is_separated_path(gdf_edges):
    """
    Bike Ottawa's code includes a construction filter:
        elif has_tag_value(way, 'highway', 'construction'):
        if has_tag_value(way, 'construction', 'path'):
            message.append('This way is a separated path because highway="construction"' 
                           + 'and construction="path".')
        elif has_tag_value(way, 'construction', 'footway'):
            message.append('This way is a separated path because highway="construction"' 
                            + 'and construction="footway".')
        elif has_tag_value(way, 'construction', 'cycleway'):
            message.append('This way is a separated path because highway="construction"' 
                            + 'and construction="cycleway".') 
    
    I'm not sure we actually want to keep the construction tag - this represents things 
    under construction.
    
    construction bit:
                   | ((gdf_edges['highway'] == 'construction') 
                      & ((gdf_edges['construction'] == 'path') | 
                      (gdf_edges['construction'] == 'footway') | 
                      (gdf_edges['construction'] == 'cycleway')))
    """



    # get the columns that start with 'cycleway'
    cycleway_tags = gdf_edges.columns[gdf_edges.columns.str.contains('cycleway')]

    conditions = [(gdf_edges['highway'] == 'cycleway'), # s3
                  (gdf_edges['highway'] == 'path'), #s1
                  ((gdf_edges['highway'] == 'footway') & ~(gdf_edges['footway'] == 'crossing')), #s2
                  (np.any(gdf_edges[cycleway_tags] == 'track', axis = 1)) , # s7
                  (np.any(gdf_edges[cycleway_tags] == 'opposite_track', axis = 1)) # s8
                  ]

    values = ['s3', 's1', 's2', 's7', 's8']

    # create a new column and use np.select to assign values to it using our lists as arguments
    gdf_edges['rule'] = np.select(conditions, values, default='s0')

    separated = gdf_edges[gdf_edges['rule'] != 's0']
    not_separated = gdf_edges[gdf_edges['rule'] == 's0']
    not_separated = not_separated.drop(columns = 'rule')

    return separated, not_separated

def is_bike_lane(gdf_edges):
    """
    Check if there's a bike lane, use road features to assign LTS
    """
    # tags that start with 'cycleway'
    cycleway_tags = gdf_edges.columns[gdf_edges.columns.str.contains('cycleway')]
    lane_identifiers = ['crossing', 'lane', 'left', 'opposite', 'opposite_lane', 'right', 'yes']

    if 'shoulder:access:bicycle' in gdf_edges.columns:
        lane_check = ((np.any(gdf_edges[cycleway_tags].isin(lane_identifiers), axis = 1))
                              | (gdf_edges['shoulder:access:bicycle'] == 'yes'))
    else:
        lane_check = np.any(gdf_edges[cycleway_tags].isin(lane_identifiers), axis = 1)

    to_analyze = gdf_edges[lane_check]
    no_lane = gdf_edges[~lane_check]

    return to_analyze, no_lane

def parking_present(gdf_edges):
    """
    Splits gdf_edges into two dataframes, one where parking is detected, the oterh where it isn't.
    
    """
    parking_tags = gdf_edges.columns[gdf_edges.columns.str.contains('parking')]
    parking_identifiers = ['yes', 'parallel', 'perpendicular', 'diagonal', 'marked']
    parking_check = np.any(gdf_edges[parking_tags].isin(parking_identifiers), axis = 1)

    parking_detected = gdf_edges[parking_check]
    parking_not_detected = gdf_edges[~parking_check]

    return parking_detected, parking_not_detected

def get_lanes(gdf_edges, default_lanes = 2):

    # make new assumed lanes column for use in calculations

    # fill na with default lanes
    # if multiple lane values present, use the largest one
    # this usually happens if multiple adjacent ways are included in the edge and 
    # there's a turning lane
    gdf_edges['lanes_assumed'] = gdf_edges['lanes'].fillna(default_lanes).apply(
        lambda x: np.array(x, dtype = 'int')).apply(lambda x: np.max(x))

    return gdf_edges

def get_max_speed(gdf_edges):
    """
    Get the speed limit for ways
    If not available, make assumptions based on road type
    This errs on the high end of assumptions
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    # create a list of conditions
    # When multiple conditions are satisfied, the first one encountered in conditions is used
    conditions = [
        (gdf_edges['maxspeed'] == 'national'),
        (gdf_edges['maxspeed'].isna()) & (gdf_edges['highway'] == 'motorway'),
        (gdf_edges['maxspeed'].isna()) & (gdf_edges['highway'] == 'primary'),
        (gdf_edges['maxspeed'].isna()) & (gdf_edges['highway'] == 'secondary'),
        (gdf_edges['maxspeed'].isna()),
        ]

    # create a list of the values we want to assign for each condition
    values = [national, motorway, primary, secondary, local]

    # create a new column and use np.select to assign values to it using our lists as arguments
    gdf_edges['maxspeed_assumed'] = np.select(conditions, values, default=gdf_edges['maxspeed'])

    # If mph
    mph = gdf_edges['maxspeed_assumed'].str.contains('mph', na=False)
    gdf_edges['maxspeed_assumed'].loc[mph] = gdf_edges['maxspeed_assumed'][mph].str.split(
        ' ', expand=True)[0].apply(lambda x: np.array(x, dtype = 'int'))

    # if multiple speed values present, use the largest one
    gdf_edges['maxspeed_assumed'] = gdf_edges['maxspeed_assumed'].apply(
        lambda x: np.array(x, dtype = 'int')).apply(lambda x: np.max(x))

    return gdf_edges

def convert_feet_with_quotes(series):
    # Calculate decimal feet and inches when each given separately
    quoteValues = series.str.contains('\'')
    feetinch = series[quoteValues.fillna(False)].str.strip('"').str.split('\'', expand=True)
    feetinch = feetinch.apply(lambda x: np.array(x, dtype = 'int'))
    if feetinch.shape[0] > 0:
        feet = feetinch[0] + feetinch[1] / 12
        series[quoteValues.fillna(False)] = feet

    # Use larger value if given multiple
    multiWidth = series.str.contains(';')
    maxWidth = series[multiWidth.fillna(False)].str.split(';', expand=True).max(axis=1)
    series[multiWidth.fillna(False)] = maxWidth

    series = series.apply(lambda x: np.array(x, dtype = 'float'))

    return series

def bike_lane_analysis_with_parking(gdf_edges):
    # get lanes, width, speed
    gdf_edges = get_lanes(gdf_edges)
    gdf_edges = get_max_speed(gdf_edges)
    if unit == 'english':
        gdf_edges['width'] = convert_feet_with_quotes(gdf_edges['width'])

    # create a list of lts conditions
    # When multiple conditions are satisfied, the first one encountered in conditions is used
    conditions = [
        (gdf_edges['lanes_assumed'] >= 3) & (gdf_edges['maxspeed_assumed'] <= ref['s4'][unit]),
        (gdf_edges['width'] <= ref['w2'][unit]),
        (gdf_edges['width'] <= ref['w3'][unit]),
        (gdf_edges['width'] <= ref['w4'][unit]) &
            ((gdf_edges['maxspeed_assumed'] <= ref['s2'][unit]) &
                (gdf_edges['highway'] == 'residential')),
        (gdf_edges['maxspeed_assumed'] > ref['s2'][unit]) &
            (gdf_edges['maxspeed_assumed'] <= ref['s3'][unit]),
        (gdf_edges['maxspeed_assumed'] > ref['s3'][unit]) & 
            (gdf_edges['maxspeed_assumed'] <= ref['s4'][unit]),
        (gdf_edges['maxspeed_assumed'] > ref['s4'][unit]),
        (gdf_edges['highway'] != 'residential')
        ]

    # create a list of the values we want to assign for each condition
    values = ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9']
    gdf_edges['rule'] = np.select(conditions, values, default='b1')
    rule_dict = {'b1':1, 'b2':3, 'b3':3, 'b4':2, 'b5':2, 'b6':2, 'b7':3, 'b8':4, 'b9':3}
    gdf_edges['lts'] = gdf_edges['rule'].map(rule_dict)

    return gdf_edges

def bike_lane_analysis_no_parking(gdf_edges):
    """
    LTS depends on presence of median, but this is not commonly tagged in OSM. 
    Possibly check the 'dual_carriageway' tag. 
    """

    # get lanes, width, speed
    gdf_edges = get_lanes(gdf_edges)
    gdf_edges = get_max_speed(gdf_edges)
    if unit == 'english':
        gdf_edges['width'] = convert_feet_with_quotes(gdf_edges['width'])

    # assign widths that are a string to nan
    gdf_edges.loc[gdf_edges[['width']].applymap(
        lambda x: isinstance(x, str))['width'], 'width'] = np.nan

    # create a list of lts conditions
    # When multiple conditions are satisfied, the first one encountered in conditions is used
    conditions = [
        (gdf_edges['lanes_assumed'] >= 3) & (gdf_edges['maxspeed_assumed'] <= ref['s5'][unit]),
        (gdf_edges[['width']].applymap(lambda x: isinstance(x, float))['width']) &
            (gdf_edges['width'] <= ref['w1'][unit]),
        (gdf_edges['maxspeed_assumed'] > ref['s3'][unit]) &
            (gdf_edges['maxspeed_assumed'] <= ref['s5'][unit]),
        (gdf_edges['maxspeed_assumed'] > ref['s5'][unit]),
        (gdf_edges['highway'] != 'residential')
        ]

    values = ['c3', 'c4', 'c5', 'c6', 'c7']
    gdf_edges['rule'] = np.select(conditions, values, default='c1')
    rule_dict = {'c1':1, 'c3':3, 'c4':2, 'c5':3, 'c6':4, 'c7':3}
    gdf_edges['lts'] = gdf_edges['rule'].map(rule_dict)

    return gdf_edges

def mixed_traffic(gdf_edges):
    # get lanes, width, speed
    gdf_edges = get_lanes(gdf_edges)
    gdf_edges = get_max_speed(gdf_edges)

    # create a list of lts conditions
    # When multiple conditions are satisfied, the first one encountered in conditions is used
    conditions = [
        (gdf_edges['motor_vehicle'] == 'no'),
        (gdf_edges['highway'] == 'pedestrian'),
        (gdf_edges['highway'] == 'footway') & (gdf_edges['footway'] == 'crossing'),
        (gdf_edges['highway'] == 'service') & (gdf_edges['service'] == 'alley'),
        (gdf_edges['highway'] == 'track'),
        (gdf_edges['maxspeed_assumed'] <= ref['s3'][unit]) & (gdf_edges['highway'] == 'service') &
            (gdf_edges['service'] == 'parking_aisle'),
        (gdf_edges['maxspeed_assumed'] <= ref['s3'][unit]) & (gdf_edges['highway'] == 'service') &
            (gdf_edges['service'] == 'driveway'),
        (gdf_edges['maxspeed_assumed'] <= ref['s1'][unit]) & (gdf_edges['highway'] == 'service'),
        (gdf_edges['maxspeed_assumed'] <= ref['s2'][unit]) & (gdf_edges['lanes_assumed'] <= 3) &
            (gdf_edges['highway'] == 'residential'),
        (gdf_edges['maxspeed_assumed'] <= ref['s2'][unit]) & (gdf_edges['lanes_assumed'] <= 3),
        (gdf_edges['maxspeed_assumed'] <= ref['s2'][unit]) & (gdf_edges['lanes_assumed'] <= 5),
        (gdf_edges['maxspeed_assumed'] <= ref['s2'][unit]) & (gdf_edges['lanes_assumed'] > 5),
        (gdf_edges['maxspeed_assumed'] <= ref['s3'][unit]) & (gdf_edges['lanes_assumed'] < 3) &
            (gdf_edges['highway'] == 'residential'),
        (gdf_edges['maxspeed_assumed'] <= ref['s3'][unit]) & (gdf_edges['lanes_assumed'] <= 3),
        (gdf_edges['maxspeed_assumed'] <= ref['s3'][unit]) & (gdf_edges['lanes_assumed'] > 3),
        (gdf_edges['maxspeed_assumed'] > ref['s3'][unit])
        ]

    # create a list of the values we want to assign for each condition
    values = ['m17', 'm13', 'm14', 'm2', 'm15', 'm3', 'm4', 'm16', 'm5', 'm6', 'm7',
              'm8', 'm9', 'm10', 'm11', 'm12']

    # create a new column and use np.select to assign values to it using our lists as arguments
    gdf_edges['rule'] = np.select(conditions, values, default='m0')

    rule_dict = {'m17':1, 'm13':1, 'm14':2, 'm2':2, 'm15':2, 'm3':2, 'm4':2, 'm16':2,
                 'm5':2, 'm6':3, 'm7':3, 'm8':4, 'm9':2, 'm10':3, 'm11':4, 'm12':4}

    gdf_edges['lts'] = gdf_edges['rule'].map(rule_dict)

    return gdf_edges

### TESTS ###
def test_answer():
    df = pd.DataFrame()
    assert biking_permitted(df) == 5
