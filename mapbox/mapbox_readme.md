# Recalculate GreaterBoston completely
py main.py process -cities Cambridge,Boston,Somerville,Brookline --rebuild --combine --plot

# Mapbox Tilesets
https://docs.mapbox.com/help/tutorials/get-started-mts-and-tilesets-cli/
https://github.com/mapbox/tilesets-cli

Note: Mapbox access token required 

Pricing: https://www.mapbox.com/pricing#tilesets
Billing period starts first day of month, ends last day of month. 
Limits are based on billing period.

## Create new tileset
tilesets upload-source skilcoyne stressmap plots/LTS.json
tilesets validate-recipe mapbox/recipe.json
tilesets create skilcoyne.stressmap_tiles --recipe mapbox/recipe.json --name "stress map"
tilesets publish skilcoyne.stressmap_tiles

## Verify info
List sources
    tilesets list-sources skilcoyne
Check source
    tilesets view-source skilcoyne stressmap

## Update data in tileset
tilesets view-source skilcoyne stressmap
tilesets upload-source skilcoyne stressmap plots/LTS.json --replace
tilesets view-source skilcoyne stressmap
tilesets publish skilcoyne.stressmap_tiles

## Update tileset recipe
https://docs.mapbox.com/mapbox-tiling-service/examples/natural-earth-data-roads/

tilesets validate-recipe mapbox/recipe.json
tilesets update-recipe skilcoyne.stressmap_tiles mapbox/recipe.json
tilesets publish skilcoyne.stressmap_tiles