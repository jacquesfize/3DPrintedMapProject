# 3D Printed Map project

This repo contains code and resources I used to create a _printable_ 3D map model.

(This document will be further detailed)

In a nutshell :

- I retrieve DEM data (in this case from the IGN)
- I load any kind of geometry data (here hiking trail)
- Rasterise the latter onto the DEM raster
- use `numpy-stl` and `freetype` to generate 3D text object
- Use `pyvista`, `gemgis`, and `numpy-stl` to transform DEM into a 3D mesh
- Use `numpy-stl` to save the final result

## Code Example

An example on how to produce 3D map with the current code

```python

from map import Map
import geopandas as gpd

map3D = Map("data.tif",scale_factor=0.5)

randos = gpd.read_file('randos.geojson')
randos = randos.explode()

# For now, geom must be done before text
map3D = map3D.add_geometry(randos, 10000)
map3D = map3D.add_text("placename", *(lat,lon), "fonts.ttf")


map3D.save("testtext+geom.stl")
```
