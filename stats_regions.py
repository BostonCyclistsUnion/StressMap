# import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import lts_functions
import LTS_OSM

CRS = 'EPSG:2805' # Massachusetts Mainland https://spatialreference.org/ref/epsg/2805/
AREA_CONVERT = 3.861022e-7 # Convert from m^2 to mi^2
COLOR_SCALE = ['#007191', '#62c8d3', '#f47a00', '#d31f11'] # From https://github.com/BostonCyclistsUnion/LabsWebsiteMap/blob/1caed6ebf02f19a3556ace58faef8e78382d86df/src/Map.jsx#L32

# %% Load Data
def load_lts_data(data='data/GreaterBoston_4_all_lts.csv'):
    gdf = LTS_OSM.read_lts_csv(data)

    gdf.to_crs(CRS, inplace=True)

    gdf['geometry_centroid'] = gdf.centroid
    gdf['geometry_way'] = gdf['geometry']

    # Reduce bi-directional line pairs
    gdf.drop_duplicates(subset=['osmid', 'geometry_centroid'], inplace=True)

    return gdf

def load_region(regionFile, local=False):
    if local:
        geojsonFolder = '../osm-bikeparking/'
        regionFile = f'{geojsonFolder}{regionFile}'

    region = gpd.read_file(regionFile)
    region.to_crs(CRS, inplace=True)
    return region

# %% Process Regions Data
def preprocess_city():
    # Pre-process city data
    city['population'] = city['population'].str.replace(r'\D', '', regex=True).astype(float) # Remove non-digit values from populations, e.g. commas
    city.rename(columns={'name': 'city'}, inplace=True)
    city['population'] = city['population'].fillna(-1)
    city = city.astype({"population": "int"})
    city['area_sqmi'] = city.area * AREA_CONVERT # https://spatialreference.org/ref/epsg/2805/

def preprocess_neighborhood():
# Pre-process neighborhood data
    hood.rename(columns={'name': 'neighborhood'}, inplace=True)
    hood['area_sqmi'] = hood.area * AREA_CONVERT # https://spatialreference.org/ref/epsg/2805/

    hoodCentroid = hood[['neighborhood', 'geometry']]
    hoodCentroid['geometry'] = hood.centroid
    hoodCentroid = hoodCentroid.sjoin(city[['city','geometry']], how='inner', predicate='intersects')
    hoodCentroid.set_index('neighborhood', inplace=True)
    hood = hood.join(hoodCentroid['city'], on='neighborhood')

def preprocess_council():
    # Pre-process city council districts data
    council['area_sqmi'] = council.area * AREA_CONVERT # https://spatialreference.org/ref/epsg/2805/

gdfSmall = gdf[['osmid', 'geometry_centroid', 'geometry_way', 'LTS_fwd', 'LTS_rev', 'geometry', 'highway']]
gdfSmall['length'] = gdfSmall.length
gdfSmall['geometry'] = gdfSmall['geometry_centroid']


gdfCity = gdfSmall.sjoin(city[['city','geometry']], how='inner', predicate='intersects')
gdfHood = gdfSmall.sjoin(hood[['neighborhood','geometry']], how='inner', predicate='intersects')
gdfCouncil = gdfSmall.sjoin(council[['LONGNAME', 'Councilor','geometry']], how='inner', predicate='intersects')


sumFwd = gdfCity.groupby(['city', 'LTS_fwd', 'highway'])['length'].sum().rename_axis(['city', 'LTS', 'highway'])
sumRev = gdfCity.groupby(['city', 'LTS_rev', 'highway'])['length'].sum().rename_axis(['city', 'LTS', 'highway'])
sumLTS = sumFwd + sumRev
sumLTS.loc[['Cambridge','Boston','Somerville','Brookline']].to_csv('total_city.csv')

sumFwd = gdfHood.groupby(['neighborhood', 'LTS_fwd', 'highway'])['length'].sum().rename_axis(['neighborhood', 'LTS', 'highway'])
sumRev = gdfHood.groupby(['neighborhood', 'LTS_rev', 'highway'])['length'].sum().rename_axis(['neighborhood', 'LTS', 'highway'])
sumLTS = sumFwd + sumRev
sumLTS.to_csv('total_neighborhood.csv')

sumFwd = gdfCouncil.groupby(['LONGNAME', 'Councilor', 'LTS_fwd', 'highway'])['length'].sum().rename_axis(['District', 'Councilor', 'LTS', 'highway'])
sumRev = gdfCouncil.groupby(['LONGNAME', 'Councilor', 'LTS_rev', 'highway'])['length'].sum().rename_axis(['District', 'Councilor', 'LTS', 'highway'])
sumLTS = sumFwd + sumRev
sumLTS.to_csv('total_council.csv')
sumLTS = sumLTS.reset_index()

gdfCouncilFilter = gdfCouncil[~gdfCouncil['highway'].isin(['residential', 'service', 'construction', 'emergency_bay'])]
sumFwd = gdfCouncilFilter.groupby(['LONGNAME', 'Councilor', 'LTS_fwd'])['length'].sum().rename_axis(['District', 'Councilor', 'LTS'])
sumRev = gdfCouncilFilter.groupby(['LONGNAME', 'Councilor', 'LTS_rev'])['length'].sum().rename_axis(['District', 'Councilor', 'LTS'])
sumLTS = sumFwd + sumRev
sumLTS = sumLTS.reset_index()
sumLTS = sumLTS[sumLTS['LTS']!='0.0']
sumLTS.fillna(0, inplace=True)
sumLTS.to_csv('total_council_filter.csv')

def plot_region(dfRegion, posData, title):
    fig, ax = plt.subplots(
        1,1, 
        figsize=(9,9),
        )
    width, height = np.array([1.0, 1.0]) * 0.08 # Size of pie charts, % of plot

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
    for r, region in dfRegion.iterrows():
        # print(region)
        coords = regionCentroid.iloc[r]
        x_fig, y_fig = transData.transform(coords)

        # print(f'{region['LONGNAME']} {x_fig=:.2f} {y_fig=:.2f} {width=:.2f} {height=:.2f}')

        half = (r+1) // ((len(dfRegion)+1)/2)
        posSide = (1 - half, 1 - (r/(len(dfRegion)/2))*(1-half) + half*(r/(len(dfRegion)/2)-1))

        ax.annotate(
            region['SHORTNAME'],
            xytext=(coords), textcoords='data',
            xy=dfRegion.loc[region['LONGNAME']][['x_axes','y_axes']], xycoords='axes fraction',
            arrowprops={
                'arrowstyle': "-",
                'connectionstyle': "arc3"
                }
            )

        pie_ax = fig.add_axes([posData.loc[region['LONGNAME']]['x_fig'] - width/2, posData.loc[region['LONGNAME']]['y_fig'] - height/2, width, height], aspect='equal')
        pie_ax.pie(
            sumLTS[sumLTS['District']==region['LONGNAME']]['length'], 
            # labels=sumLTS[sumLTS['District']==region['LONGNAME']]['LTS'],
            startangle=90, colors=COLOR_SCALE
            )
        pie_ax.set_title(f'{region['LONGNAME']}\n{region['Councilor']}', y=1.0, pad=-4, fontdict={'size': 7})

def plot_layout_council():
    title = 'Bike Stress by Boston Council District'

    posLimitY = [0, 1.0]
    posSideX = [0.0, 0.9]
    sideCount = [4, 4]
    posData = pd.DataFrame(
        [
            ['District 1', posSideX[1], 4*(posLimitY[1]-posLimitY[0])/sideCount[1] + posLimitY[0]],
            ['District 2', posSideX[1], 2*(posLimitY[1]-posLimitY[0])/sideCount[1] + posLimitY[0]],
            ['District 3', posSideX[1], 3*(posLimitY[1]-posLimitY[0])/sideCount[1] + posLimitY[0]],
            ['District 4', posSideX[1], 1*(posLimitY[1]-posLimitY[0])/sideCount[1] + posLimitY[0]],
            ['District 5', 0.4, 0.15*(posLimitY[1]-posLimitY[0])],
            ['District 6', posSideX[0], 1*(posLimitY[1]-posLimitY[0])/sideCount[0] + posLimitY[0]],
            ['District 7', posSideX[0], 2*(posLimitY[1]-posLimitY[0])/sideCount[0] + posLimitY[0]],
            ['District 8', posSideX[0], 4*(posLimitY[1]-posLimitY[0])/sideCount[0] + posLimitY[0]],
            ['District 9', posSideX[0], 3*(posLimitY[1]-posLimitY[0])/sideCount[0] + posLimitY[0]],
        ], columns=['LONGNAME', 'x_axes', 'y_axes']
        ).set_index('LONGNAME')

if __name__ == '__main__':
    lts = load_lts_data()

    region = []
    for r in ['data_cities.geojson','data_neighborhoods.geojson','data_council_boston.json']:
        region.append(load_region(r, True))