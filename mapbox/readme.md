https://docs.mapbox.com/help/tutorials/get-started-mts-and-tilesets-cli/

tilesets delete skilcoyne.stressmap_tiles

tilesets upload-source skilcoyne stressmap plots/LTS.json

tilesets create skilcoyne.stressmap_tiles --recipe mapbox/recipe.json --name "stress map"

tilesets publish skilcoyne.stressmap_tiles


tilesets update-recipe skilcoyne.stressmap_tiles mapbox/recipe.json