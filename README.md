# M4py : 3D Printed Map project

We propose a library m4py, a toolbox to generate 3D maps and combine them with 3D printed objects (text, mesh, etc).

## Get Started

### Setup

```shell
git clone https://github.com/jacquesfize/3DPrintedMapProject.git
cd 3DPrintedMapProject
pip install .
```

### Usage

First, import the Map object:

```python
from map import Map
```

To instatiate the Map object, we need the path to the raster file and the scale factor applied to the latter.

```python
map3D = Map("data.tif",scale_factor=0.5)
```

`Map` object are composed of:

- a raster `self.raster`
- a mesh of the 3D transformed raster `self.map_mesh`
- a list of geometries that will be extruded onto the raster `self.meshs`

If you just want the 3D model of the raster and save it:

```python
final_mesh = map3D.save_mesh("test.stl")
```

### Add text

To add text in your 3D map, use the `add_text` method like in the following:

```python
map3D = map3D.add_text("placename", *(lat,lon), "fonts.ttf")
```

If you want to place any mesh at non-geographic coordinates, use the `add_text` method like in the following:

```python
map3D = map3D.add_text("placename", *(x,y), "fonts.ttf",raster_coords=True)
```

### Add mesh

Similar to the `add_text` method, you can add any mesh at any geographic coordinates with the `add_mesh` method. For now, the mesh is defined by the `numpy-stl` `Mesh` object.

```python
map3D = map3D.add_mesh(mesh.Mesh.from_file("test.stl"), 0, 0)
```

An example on how to produce 3D map with the current code

### Extrude geometries

In addition to text and mesh, you can also extrude geometries in the 3D map. To do this, use the `add_geometry` method like in the following:

```python
geometry_dataframe = gpd.read_file('your_data.geojson')
map3D = map3D.add_geometry(geometry_dataframe, 10000)
```

### Plot your 3D map

To plot your 3D map, use the `plot` method like in the following:

```python
map3D.plot()
```

We use `pyvista` to plot the 3D object therefore if you want to customize the plot rendering, you can use the `plot` method like in `pyvista`:

```python
map3D.plot(cpos="xy")
```

![example](docs/images/plot_ecrins.png)

## 3D printed map example

This 3d map shows the refuges and hut in the _Parc National des Écrins_ (France) area.

![example](docs/images/test_refuges_ecrins.jpeg)

PS. Poor painting skills here🤦‍♂️

## Todo

- Be able to clip mesh generated and to print the clipped meshes
- Rename map.py to avoid confusion with `map` in python
- Write a better documentation
- drop numpy-stl for pyvista

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
