https://docs.mapbox.com/help/tutorials/get-started-mts-and-tilesets-cli/

tilesets upload-source skilcoyne stressmap plots/LTS.json

tilesets create skilcoyne.stressmap_tiles --recipe mapbox/recipe.json --name "stress map"

tilesets publish skilcoyne.stressmap_tiles