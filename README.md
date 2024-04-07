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

## Get Started

An example on how to produce 3D map with the current code

```python

from map import Map
import geopandas as gpd
from stl import mesh # numpy-stl

map3D = Map("data.tif",scale_factor=0.5)

randos = gpd.read_file('randos.geojson') # must be use the same projection as the raster
randos = randos.explode() # do not works with MultiLineString... explode() transform them into multiple LineString

# Extrude geometry on the raster
map3D = map3D.add_geometry(randos, 10000)
# Create a 3d object from a text and a given font file
map3D = map3D.add_text("placename", *(lat,lon), "fonts.ttf")
# Include mesh (use numpy-stl Mesh object) at coordinates lon=0 lat=0 (in the raster projection)
map3D = map3D.add_mesh(mesh.Mesh.from_file("test.stl"), 0, 0)

# To generate the final mesh
final_mesh = map3D.generate_mesh()

# To save the final mesh (the method call automatically calls the `generate_mesh()` method)
map3D.save("saveFile.stl")
```

## Final render example

![example](docs/images/plot_ecrins.png)

## Todo

- Be able to clip mesh generated and to print the clipped meshes
- Stabilize text extrusion
- Rename map.py to avoid confusion with `map` in python
- Generate a setup.py file
- Write a better documentation
