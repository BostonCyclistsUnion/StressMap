'''
Calculate the amount of each stress level within sub-regions (city, neighborhood, council district).

Outputs:
    * a csv with the calculated tables for each region type
    * maps with each region's pie chart of stress levels
    ** raw values
    ** filtered to exclude ways that are not significant for traveling across town
'''
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import LTS_OSM

CRS = 'EPSG:2805' # Massachusetts Mainland https://spatialreference.org/ref/epsg/2805/
AREA_CONVERT = 3.861022e-7 # Convert from m^2 to mi^2
COLOR_SCALE = ['#007191', '#62c8d3', '#f47a00', '#d31f11'] # From https://github.com/BostonCyclistsUnion/LabsWebsiteMap/blob/1caed6ebf02f19a3556ace58faef8e78382d86df/src/Map.jsx#L32

# %% Load Data
def load_lts_data(data='data/GreaterBoston_4_all_lts.csv'):
    print(f'Loading LTS data from {data}...')
    gdf = LTS_OSM.read_lts_csv(data)
    gdf.to_crs(CRS, inplace=True)
    gdf = get_centroid(gdf)

    # Reduce bi-directional line pairs
    gdf.drop_duplicates(subset=['osmid', 'geometry_centroid'], inplace=True)

    print(f'\t...{data} loaded and cleaned')
    return gdf

def load_region(regionName, local=False):
    if local:
        geojsonFolder = '../osm-bikeparking/'
        regionFile = f'{geojsonFolder}data_{regionName}.geojson'
    else:
        print('FIXME: read files from github!!!')

    region = gpd.read_file(regionFile)
    region.to_crs(CRS, inplace=True)
    print(f'\t...Region type [{regionName}] loaded')
    return region

# %% Process Regions Data
def preprocess(regionType, regionData, regions):
    if regionType == 'cities':
        print('\t...Preprocessing cities')
        regionData = preprocess_city(regionData)
    elif regionType == 'neighborhoods':
        print('\t...Preprocessing neighborhoods')
        regionData = preprocess_neighborhood(regionData, regions[0])
    elif regionType == 'council_boston':
        print('\t...Preprocessing Boston council districts')
        regionData = preprocess_council_boston(regionData)
    else:
        print('A new type of region has been discovered!')
    return regionData

def preprocess_city(city):
    # Pre-process city data
    city['population'] = city['population'].str.replace(r'\D', '', regex=True).astype(float) # Remove non-digit values from populations, e.g. commas
    # city.rename(columns={'name': 'city'}, inplace=True)
    city['city'] = city['name']
    city['population'] = city['population'].fillna(-1)
    city = city.astype({"population": "int"})
    city['area_sqmi'] = city.area * AREA_CONVERT # https://spatialreference.org/ref/epsg/2805/
    city = city.sort_values(by='city').reset_index()
    return city

def preprocess_neighborhood(hood, city):
# Pre-process neighborhood data
    # hood.rename(columns={'name': 'neighborhood'}, inplace=True)
    hood['neighborhood'] = hood['name']
    hood['area_sqmi'] = hood.area * AREA_CONVERT # https://spatialreference.org/ref/epsg/2805/

    hoodCentroid = hood[['neighborhood', 'geometry']]
    hoodCentroid['geometry'] = hood.centroid
    hoodCentroid = hoodCentroid.sjoin(city[['city','geometry']], how='inner', predicate='intersects')
    hoodCentroid = hoodCentroid.drop('index_right', axis=1)
    hoodCentroid.set_index('neighborhood', inplace=True)
    hood = hood.join(hoodCentroid['city'], on='neighborhood')
    hood = hood.sort_values(by='neighborhood').reset_index()
    return hood

def preprocess_council_boston(council):
    # Pre-process city council districts data
    council['area_sqmi'] = council.area * AREA_CONVERT # https://spatialreference.org/ref/epsg/2805/
    council.rename(columns={'DISTRICT': 'DistNo', 'LONGNAME': 'District', 'SHORTNAME': 'Dist'}, inplace=True)
    council['name'] = council['District']
    council = council.sort_values(by='District').reset_index()
    return council

def get_centroid(gdf):
    gdf['geometry_way'] = gdf['geometry']
    gdf['geometry_centroid'] = gdf.centroid
    return gdf

def swtich_gdf_geometry(gdf, newGeometry):
    gdf['geometry'] = gdf[f'geometry_{newGeometry}']
    return gdf

def get_way_length(gdf):
    gdf['length'] = swtich_gdf_geometry(gdf, 'way').length
    return gdf

def get_name_col(df):
    '''
    Finds the column name for the type of region when it could be e.g. `District`, `City`, or `Neighborhood`
    '''
    nameCol = df.columns.to_list()
    nameCol.remove('LTS')
    nameCol.remove('length')
    return nameCol[0]

def ensure_four_lts(ltsLen):
    nameCol = get_name_col(ltsLen)
    name = ltsLen[nameCol].unique()

    for v in name:
        if len(ltsLen[ltsLen[nameCol] == v]) == 4:
            pass
        else:
            newRows = []
            for lts in ['1.0', '2.0', '3.0', '4.0']: # May want to remove other ratings e.g. 0.0 or 5.0 that could be placeholders and make their way into this stage
                if lts not in ltsLen[ltsLen[nameCol] == v]['LTS'].to_list():
                    newRows.append([lts, v, 0.0])
            newRowsdf = pd.DataFrame(newRows, columns=['LTS',nameCol, 'length'])
            ltsLen = pd.concat([ltsLen, newRowsdf])
            print(ltsLen)
    return ltsLen

def spatial_join(gdf, region, cols):
    print('\t...Conducting spatial join')
    gdf = gdf.sjoin(region[cols + ['geometry']], how='inner', predicate='intersects')
    gdf = gdf.drop('index_right', axis=1)
    # print(f'gdf Columns: {gdf.columns[-5:]}')
    return gdf

def get_stats(gdf, cols, outCSV, outFilter=False):
    '''
    gdf: geopandas dataframe to save summary stats for
    cols: [list] of column names to groupby
    outCSV: base filename for csv to save
    outFilter: [list] of indecies to limit saved output of. Use to eliminate cities outside of analaysis area due to ways crossing borders
    '''
    print(f'\t...Calculating summary statistics')
    sumFwd = gdf.groupby(cols + ['LTS_fwd', 'highway'])['length'].sum().rename_axis(cols + ['LTS', 'highway'])
    sumRev = gdf.groupby(cols + ['LTS_rev', 'highway'])['length'].sum().rename_axis(cols + ['LTS', 'highway'])
    sumLTS = sumFwd + sumRev
    sumLTS = sumLTS.reset_index()
    sumLTS.fillna(0, inplace=True)
    if outFilter:
        sumLTS = sumLTS[sumLTS.iloc[:,0].isin(outFilter)]
    sumLTS.to_csv(f'{outCSV}.csv')
    print(f'\tSaved {outCSV}.csv')

    highwayFilter = [
        'residential', # Current LTS method defaults to low LTS and is poorly calibrated. While residential streets need traffic calming and are critical to the bike network, the focus of improving the bike network needs to start with streets with significant destinations and roads to travel across the city efficiently.
        'service', # Not primary parts of network, eliminates driveways, etc.
        'construction', # Not yet built
        'emergency_bay', # Shouldn't be part of the bike network
        'footway', 'pedestrian', 'path', # Removes crossings and other places where biking may be allowed but not designated nor part of the street network. Future work could use multiple tags to filter more specicifcally.
        ]
    print('\t\tdrop highway filter')
    sumLTS.drop(sumLTS[sumLTS['highway'].isin(highwayFilter)].index, inplace=True)
    print('\t\tnon 0.0')
    sumLTS = sumLTS[sumLTS['LTS']!='0.0']
    sumLTS.to_csv(f'{outCSV}_filtered.csv')
    print(f'\tSaved {outCSV}_filtered.csv, removing `highway`= {highwayFilter}')

    # Calculate an overall stress grade
    ltsWeights = [1.0, 0.8, 0.2, 0.0]
    col1 = sumLTS.columns[0]
    print('\t\tltsLen')
    ltsLen = sumLTS.groupby(['LTS', col1])['length'].sum().reset_index()
    ltsLen = ensure_four_lts(ltsLen)
    print('\t\tregionLen')
    regionLen = ltsLen.reset_index().groupby([col1])['length'].sum().reset_index()
    ltsGrades = []
    for i, rL in regionLen.iterrows():
        # print(f'\t\t{i}')
        ltsLens = ltsLen[ltsLen[col1] == rL[col1]].sort_values(by='LTS')['length']
        totalLen = regionLen[regionLen[col1] == rL[col1]]['length'].iloc[0]
        ltsPercent = ltsLens / totalLen
        print('\t\tltsGrade')
        ltsGrade = int(np.floor(np.sum(ltsLens / totalLen * ltsWeights) * 100))
        ltsGrades.append([rL[col1], ltsGrade])
    ltsGrades = pd.DataFrame(ltsGrades, columns=['name', 'LTS_grade']).set_index('name')
    ltsGrades.to_csv(f'{outCSV}_grades.csv')
    print(f'\tSaved {outCSV}_grades.csv')

    return sumLTS, ltsGrades

def plot_region(dfRegion, sumLTS, title, posData):
    fig, ax = plt.subplots(
        1,1, 
        figsize=(9,9),
        )
    width, height = np.array([1.0, 1.0]) * 0.08 # Size of pie charts, % of plot

    posData = posData.sort_index()
    
    nanrows = posData.loc[posData['x_axes'].isna()]
    nanrows.to_csv(f'unplotted_{sumLTS.columns[0]}.csv')
    posData.dropna(subset=['x_axes','y_axes'], inplace=True)
    print(f'Unable to plot due to lack of position info: {nanrows.index.to_list()}')

    # print(f'dfRegion before check entries in sumLTS: {dfRegion.shape}')
    dfRegion = dfRegion[dfRegion[sumLTS.columns[0]].isin(sumLTS.iloc[:,0])].reset_index().drop(columns=['level_0', 'index'])
    # print(f'dfRegion after remove entries missing from sumLTS: {dfRegion.shape}')
    # print(dfRegion)

    dfRegion.plot(
        cmap='Set3',
        ax=ax,
        aspect='equal',
        )
    ax.axis(False)
    ax.set_title(title)

    bbox = ax.get_position()  
    transData = ax.transData + fig.transFigure.inverted()

    posData[['x_fig', 'y_fig']] = (ax.transAxes + fig.transFigure.inverted()).transform(posData[['x_axes','y_axes']])

    # regionCentroid = dfRegion.centroid.get_coordinates(ignore_index=True)
    dfRegion[['region_centroid_x', 'region_centroid_y']] = dfRegion.centroid.get_coordinates(ignore_index=True)
    # print(regionCentroid)
    # print(posData)
    for r, posRegion in posData.iterrows():
        # print(f'Adding [{r}] to figure')
        region = dfRegion[dfRegion['name'] == posRegion.name]
        # print(f'{r} | {posRegion.name} | {region['name']}')        
        coords = region[['region_centroid_x', 'region_centroid_y']].values.flatten().tolist()
        # print(f'coords: {coords}')
        x_fig, y_fig = transData.transform(coords)

        # print(f'Drawing annotation for [{r}]')
        ax.annotate(
            posRegion['pie_annotation'],
            xytext=(coords), textcoords='data',
            xy=posRegion[['x_axes','y_axes']], xycoords='axes fraction',
            arrowprops={
                'arrowstyle': "-",
                'connectionstyle': "arc3"
                }
            )

        # print(f'Plotting pie chart for [{r}]')
        pie_ax = fig.add_axes([posRegion['x_fig'] - width/2, posRegion['y_fig'] - height/2, width, height], aspect='equal')
        pie_ax.pie(
            sumLTS[sumLTS[sumLTS.columns[0]] == r].groupby('LTS')['length'].sum(), 
            # labels=sumLTS[sumLTS['District']==region['District']]['LTS'],
            startangle=90, colors=COLOR_SCALE
            )
        pie_ax.set_title(f'{posRegion.name}', y=1.0, pad=-4, fontdict={'size': 7})
        # FIXME: add subtitle conditionally to plot

def plot_layout_city(grades):
    title = 'Bike Stess by City'

    posLimitY = [0, 1.0]
    posSideX = [0.0, 0.9]
    sideCount = [2, 2]
    posData = pd.DataFrame(
        [
            ['Somerville',  posSideX[1], 0.8, None, 'blue'],
            ['Boston',      posSideX[1], 0.2, None, 'red'],
            ['Cambridge',   posSideX[0], 0.8, None, 'orange'],
            ['Brookline',   posSideX[0], 0.2, None, 'purple'],
        ], columns=['pie_title', 'x_axes', 'y_axes', 'pie_annotation', 'color']
        ).set_index('pie_title')
    posData = pd.concat([posData, grades], axis=1)

    return title, posData

def plot_layout_neighborhood(grades):
    title = 'Bike Stess by Neighborhood'

    # print(grades)
    posData = pd.DataFrame(
        [
            ['Allston',  0.0, 0.52, None],
            ['Back Bay', 0.5, 0.8, None],
            ['Baldwin', 0.4, 0.8, None],
            ['Bay Village', 0.5, 0.8, None],
            ['Beacon Hill', 0.4, 0.8, None],
            ['Brighton', 0.0, 0.64, None],
            ['Cambridge Highlands', 0.0, 1.0, None],
            ['Cambridgeport', 0.4, 0.8, None],
            ['Charlestown', 0.4, 0.8, None],
            ['Chinatown', 0.4, 0.8, None],
            ['Dorchester', 0.6, 0.2, None],
            ['Downtown', 0.4, 0.8, None],
            ['East Boston', 0.4, 0.8, None],
            ['East Cambridge', 0.4, 0.8, None],
            ['Fenway-Kenmore', 0.4, 0.8, None],
            ['Harbor Islands', 0.8, 0.5, None],
            ['Hyde Park', 0.2, 0.0, None],
            ['Jamaica Plain', 0.0, 0.4, None],
            ['Leather District', 0.4, 0.8, None],
            ['Longwood', 0.4, 0.8, None],
            ['MIT', 0.4, 0.8, None],
            ['Mid-Cambridge', 0.4, 0.8, None],
            ['Mission Hill', 0.4, 0.8, None],
            ['Neighborhood Nine', 0.3, 0.95, None],
            ['North Cambridge', 0.15, 1.0, None],
            ['North End', 1.0, 0.8, None],
            ['Riverside', 0.4, 0.8, None],
            ['Roslindale', 0.4, 0.8, None],
            ['Roxbury', 0.4, 0.8, None],
            ['Seaport', 1.0, 0.6, None],
            ['South Boston', 1.0, 0.5, None],
            ['South End', 1.0, 0.5, None],
            ['Strawberry Hill', 0.0, 0.88, None],
            ['The Port', 0.4, 0.8, None],
            ['Wellington-Harrington', 0.4, 0.8, None],
            ['West Cambridge', 0.0, 0.76, None],
            ['West End', 1.0, 0.9, None],
            ['West Roxbury', 0.0, 0.1, None],
        ], columns=['pie_title', 'x_axes', 'y_axes', 'pie_annotation']
        ).set_index('pie_title')
    posData = pd.concat([posData, grades], axis=1)

    return title, posData

def plot_layout_council(grades):
    title = 'Bike Stress by Boston Council District'

    posLimitY = [0, 1.0]
    posSideX = [0.0, 0.9]
    sideCount = [4, 4]
    posData = pd.DataFrame(
        [
            ['District 1', '\nname', posSideX[1], 4*(posLimitY[1]-posLimitY[0])/sideCount[1] + posLimitY[0], 'D1'],
            ['District 2', '\nname', posSideX[1], 2*(posLimitY[1]-posLimitY[0])/sideCount[1] + posLimitY[0], 'D2'],
            ['District 3', '\nname', posSideX[1], 3*(posLimitY[1]-posLimitY[0])/sideCount[1] + posLimitY[0], 'D3'],
            ['District 4', '\nname', posSideX[1], 1*(posLimitY[1]-posLimitY[0])/sideCount[1] + posLimitY[0], 'D4'],
            ['District 5', '\nname', 0.4,         0.15*(posLimitY[1]-posLimitY[0]),                          'D5'],
            ['District 6', '\nname', posSideX[0], 1*(posLimitY[1]-posLimitY[0])/sideCount[0] + posLimitY[0], 'D6'],
            ['District 7', '\nname', posSideX[0], 2*(posLimitY[1]-posLimitY[0])/sideCount[0] + posLimitY[0], 'D7'],
            ['District 8', '\nname', posSideX[0], 4*(posLimitY[1]-posLimitY[0])/sideCount[0] + posLimitY[0], 'D8'],
            ['District 9', '\nname', posSideX[0], 3*(posLimitY[1]-posLimitY[0])/sideCount[0] + posLimitY[0], 'D9'],
        ], columns=['pie_title', 'pie_subtitle', 'x_axes', 'y_axes', 'pie_annotation']
        ).set_index('pie_title')
    posData = pd.concat([posData, grades], axis=1)
    
    return title, posData

def main():
    ltsGDF = load_lts_data('data/GreaterBoston_4_all_lts.csv')
    ltsGDF = get_centroid(ltsGDF)
    ltsGDF = get_way_length(ltsGDF)

    regions = []
    regionsStats = []
    regionsGrades = []
    for regionName, cols in [
        ['cities', ['city']], 
        ['neighborhoods', ['neighborhood']], 
        ['council_boston', ['District', 'Councilor']]
        ]:
        print(f'Processing {regionName}...')
        regionData = load_region(regionName, True)
        regionData = preprocess(regionName, regionData, regions)
        ltsGDF = spatial_join(ltsGDF, regionData, cols)
        regions.append(regionData)

        if regionName == 'cities':
            regionStats, ltsGrades = get_stats(ltsGDF, cols, f'total_{regionName}', ['Boston', 'Cambridge', 'Somerville', 'Brookline'])
        else:
            regionStats, ltsGrades = get_stats(ltsGDF, cols, f'total_{regionName}')
        regionsStats.append(regionStats)
        regionsGrades.append(ltsGrades)

    return regions, regionsStats, regionsGrades
   
if __name__ == '__main__':
    main()
