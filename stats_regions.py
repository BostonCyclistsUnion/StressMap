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
    return region

# %% Process Regions Data
def preprocess(regionType, regionData, regions):
    if regionType == 'cities':
        # print('Preprocessing cities...')
        regionData = preprocess_city(regionData)
    elif regionType == 'neighborhoods':
        # print('Preprocessing neighborhoods...')
        regionData = preprocess_neighborhood(regionData, regions[0])
    elif regionType == 'council_boston':
        # print('Preprocessing council districts...')
        regionData = preprocess_council_boston(regionData)
    return regionData

def preprocess_city(city):
    # Pre-process city data
    city['population'] = city['population'].str.replace(r'\D', '', regex=True).astype(float) # Remove non-digit values from populations, e.g. commas
    city.rename(columns={'name': 'city'}, inplace=True)
    city['population'] = city['population'].fillna(-1)
    city = city.astype({"population": "int"})
    city['area_sqmi'] = city.area * AREA_CONVERT # https://spatialreference.org/ref/epsg/2805/
    return city

def preprocess_neighborhood(hood, city):
# Pre-process neighborhood data
    hood.rename(columns={'name': 'neighborhood'}, inplace=True)
    hood['area_sqmi'] = hood.area * AREA_CONVERT # https://spatialreference.org/ref/epsg/2805/

    hoodCentroid = hood[['neighborhood', 'geometry']]
    hoodCentroid['geometry'] = hood.centroid
    hoodCentroid = hoodCentroid.sjoin(city[['city','geometry']], how='inner', predicate='intersects')
    hoodCentroid = hoodCentroid.drop('index_right', axis=1)
    hoodCentroid.set_index('neighborhood', inplace=True)
    hood = hood.join(hoodCentroid['city'], on='neighborhood')
    return hood

def preprocess_council_boston(council):
    # Pre-process city council districts data
    council['area_sqmi'] = council.area * AREA_CONVERT # https://spatialreference.org/ref/epsg/2805/
    council.rename(columns={'DISTRICT': 'DistNo', 'LONGNAME': 'District', 'SHORTNAME': 'Dist'}, inplace=True)
    return council

def get_centroid(gdf):
    gdf['geometry_way'] = gdf['geometry']
    gdf['geometry_centroid'] = gdf.centroid
    return gdf

def swtich_gdf_geometry(gdf, newGeometry):
    gdf['geometry'] = gdf[f'geometry_{newGeometry}']
    return gdf

def get_way_length(gdf):
    # gdfSmall = gdf[['osmid', 'geometry_centroid', 'geometry_way', 'LTS_fwd', 'LTS_rev', 'geometry', 'highway']]
    # gdfSmall['length'] = gdfSmall.length
    # gdfSmall['geometry'] = gdfSmall['geometry_centroid']

    gdf['length'] = swtich_gdf_geometry(gdf, 'way').length

    return gdf

def spatial_join(gdf, region, cols):
    gdf = gdf.sjoin(region[cols + ['geometry']], how='inner', predicate='intersects')
    gdf = gdf.drop('index_right', axis=1)
    # print(gdf.columns)
    return gdf

# gdfCity = gdfSmall.sjoin(city[['city','geometry']], how='inner', predicate='intersects')
# gdfHood = gdfSmall.sjoin(hood[['neighborhood','geometry']], how='inner', predicate='intersects')
# gdfCouncil = gdfSmall.sjoin(council[['District', 'Councilor','geometry']], how='inner', predicate='intersects')

def get_stats(gdf, cols, outCSV, outFilter=False):
    '''
    gdf: geopandas dataframe to save summary stats for
    cols: [list] of column names to groupby
    outCSV: base filename for csv to save
    outFilter: [list] of indecies to limit saved output of. Use to eliminate cities outside of analaysis area due to ways crossing borders
    '''
    sumFwd = gdf.groupby(cols + ['LTS_fwd', 'highway'])['length'].sum().rename_axis(cols + ['LTS', 'highway'])
    sumRev = gdf.groupby(cols + ['LTS_rev', 'highway'])['length'].sum().rename_axis(cols + ['LTS', 'highway'])
    sumLTS = sumFwd + sumRev
    sumLTS = sumLTS.reset_index()
    sumLTS.fillna(0, inplace=True)
    if outFilter:
        sumLTS = sumLTS[sumLTS.iloc[:,0].isin(outFilter)]
    sumLTS.to_csv(outCSV)
    print(f'Saved {outCSV}.csv')

    highwayFilter = [
        'residential', # Current LTS method defaults to low LTS and is poorly calibrated. While residential streets need traffic calming and are critical to the bike network, the focus of improving the bike network needs to start with streets with significant destinations and roads to travel across the city efficiently.
        'service', # Not primary parts of network, eliminates driveways, etc.
        'construction', # Not yet built
        'emergency_bay', # Shouldn't be part of the bike network
        'footway', 'pedestrian', 'path', # Removes crossings and other places where biking may be allowed but not designated nor part of the street network. Future work could use multiple tags to filter more specicifcally.
        ]

    sumLTS.drop(sumLTS[sumLTS['highway'].isin(highwayFilter)].index, inplace=True)
    sumLTS = sumLTS[sumLTS['LTS']!='0.0']
    sumLTS.to_csv(f'{outCSV}_filtered.csv')
    print(f'Saved {outCSV}_filtered.csv, removing `highway`= {highwayFilter}')

    return sumLTS

# sumFwd = gdfCity.groupby(['city', 'LTS_fwd', 'highway'])['length'].sum().rename_axis(['city', 'LTS', 'highway'])
# sumRev = gdfCity.groupby(['city', 'LTS_rev', 'highway'])['length'].sum().rename_axis(['city', 'LTS', 'highway'])
# sumLTS = sumFwd + sumRev
# sumLTS.loc[['Cambridge','Boston','Somerville','Brookline']].to_csv('total_city.csv')

# sumFwd = gdfHood.groupby(['neighborhood', 'LTS_fwd', 'highway'])['length'].sum().rename_axis(['neighborhood', 'LTS', 'highway'])
# sumRev = gdfHood.groupby(['neighborhood', 'LTS_rev', 'highway'])['length'].sum().rename_axis(['neighborhood', 'LTS', 'highway'])
# sumLTS = sumFwd + sumRev
# sumLTS.to_csv('total_neighborhood.csv')

# sumFwd = gdfCouncil.groupby(['District', 'Councilor', 'LTS_fwd', 'highway'])['length'].sum().rename_axis(['District', 'Councilor', 'LTS', 'highway'])
# sumRev = gdfCouncil.groupby(['District', 'Councilor', 'LTS_rev', 'highway'])['length'].sum().rename_axis(['District', 'Councilor', 'LTS', 'highway'])
# sumLTS = sumFwd + sumRev
# sumLTS.to_csv('total_council.csv')
# sumLTS = sumLTS.reset_index()

# gdfCouncilFilter = gdfCouncil[~gdfCouncil['highway'].isin(['residential', 'service', 'construction', 'emergency_bay', 'footway'])]
# sumFwd = gdfCouncilFilter.groupby(['District', 'Councilor', 'LTS_fwd'])['length'].sum().rename_axis(['District', 'Councilor', 'LTS'])
# sumRev = gdfCouncilFilter.groupby(['District', 'Councilor', 'LTS_rev'])['length'].sum().rename_axis(['District', 'Councilor', 'LTS'])
# sumLTS = sumFwd + sumRev
# sumLTS = sumLTS.reset_index()
# sumLTS = sumLTS[sumLTS['LTS']!='0.0']
# sumLTS.fillna(0, inplace=True)
# sumLTS.to_csv('total_council_filter.csv')

def plot_region(dfRegion, sumLTS, title, posData):
    fig, ax = plt.subplots(
        1,1, 
        figsize=(9,9),
        )
    width, height = np.array([1.0, 1.0]) * 0.08 # Size of pie charts, % of plot

    dfRegion = dfRegion[dfRegion[sumLTS.columns[0]].isin(sumLTS.iloc[:,0])].reset_index()

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

    regionCentroid = dfRegion.centroid.get_coordinates(ignore_index=True)
    # print(regionCentroid)
    for r, region in dfRegion.iterrows():
        # print(region)
        print(r, posData.index[r])
        coords = regionCentroid.iloc[r]
        # print(coords)
        x_fig, y_fig = transData.transform(coords)

        # print(f'{region['District']} {x_fig=:.2f} {y_fig=:.2f} {width=:.2f} {height=:.2f}')

        half = (r+1) // ((len(dfRegion)+1)/2)
        posSide = (1 - half, 1 - (r/(len(dfRegion)/2))*(1-half) + half*(r/(len(dfRegion)/2)-1))

        # print('annotate')
        ax.annotate(
            posData['pie_annotation'].iloc[r],
            xytext=(coords), textcoords='data',
            xy=posData[['x_axes','y_axes']].iloc[r], xycoords='axes fraction',
            arrowprops={
                'arrowstyle': "-",
                'connectionstyle': "arc3"
                }
            )
        # print('annotated')

        pie_ax = fig.add_axes([posData.iloc[r]['x_fig'] - width/2, posData.iloc[r]['y_fig'] - height/2, width, height], aspect='equal')
        pie_ax.pie(
            sumLTS[sumLTS[sumLTS.columns[0]]==region[sumLTS.columns[0]]].groupby('LTS')['length'].sum(), 
            # labels=sumLTS[sumLTS['District']==region['District']]['LTS'],
            startangle=90, colors=COLOR_SCALE
            )
        pie_ax.set_title(f'{posData.index[r]}', y=1.0, pad=-4, fontdict={'size': 7})
        # FIXME: add subtitle conditionally to plot

def plot_layout_city():
    title = 'Bike Stess by City'

    posLimitY = [0, 1.0]
    posSideX = [0.0, 0.9]
    sideCount = [2, 2]
    posData = pd.DataFrame(
        [
            ['Somerville',  posSideX[1], 0.8, None],
            ['Boston',      posSideX[1], 0.2, None],
            ['Cambridge',   posSideX[0], 0.8, None],
            ['Brookline',   posSideX[0], 0.2, None],
        ], columns=['pie_title', 'x_axes', 'y_axes', 'pie_annotation']
        ).set_index('pie_title')

    return title, posData

def plot_layout_council():
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
    
    return title, posData

def main():
    ltsGDF = load_lts_data('data/GreaterBoston_4_all_lts.csv')
    ltsGDF = get_centroid(ltsGDF)
    ltsGDF = get_way_length(ltsGDF)

    regions = []
    regionsStats = []
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
            regionStats = get_stats(ltsGDF, cols, f'total_{regionName}', ['Boston', 'Cambridge', 'Somerville', 'Brookline'])
        else:
            regionStats = get_stats(ltsGDF, cols, f'total_{regionName}')
        regionsStats.append(regionStats)

    return regions, regionsStats
   
if __name__ == '__main__':
    main()
