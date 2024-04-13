import atexit
from typing import Optional

import gemgis as gg
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from stl import mesh
from utils import extrude_geometries, extrude_text, load_raster_data, parse_Mesh_to_PolyData
from basic_mesh import generate_cube


class Map:
    def __init__(
        self,
        dem_filename: str = "",
        scale_factor: int = 1,
        data: dict = None,
        elevation_scale: float = 0.02,
    ):
        """
        Constructor for the Map class.

        Parameters
        ----------
        dem_filename : str, optional
            The path to the DEM file. Default is an empty string.
        scale_factor : int, optional
            The scale factor to apply to the DEM. Default is 1.
        data : dict, optional
            Dictionary containing the data to load. Default is None.
        elevation_scale : float, optional
            The scale factor to apply to the elevation values. Defaults to 0.02.
        Notes
        -----
        If `data` is not None, it calls `load_from_data(data)`.
        If `data` is None, it loads the raster data and applies the scale factor if specified.
        Registers `cleanup` to be called at exit.
        """
        if not data:
            self.raster, self.dem_band = load_raster_data(dem_filename, True)
            mem = rasterio.MemoryFile()
            mem = mem.open(**self.raster.profile.copy())
            mem.write(self.dem_band, 1)
            self.raster.close()
            self.raster = mem
            if scale_factor != 1:
                self.scale_raster(scale_factor)

            self.map_mesh = self.compute_dem_mesh(elevation_scale=elevation_scale)

            self.meshs = []
            self.elevation_scale = elevation_scale
        else:
            self.load_from_data(data)

        atexit.register(self.cleanup)

    def copy(self):
        return Map(
            data=dict(
                map_mesh=mesh.Mesh(self.map_mesh.data.copy()),
                raster=self.copy_raster(),
                meshs=[mesh.Mesh(mesh_.data.copy()) for mesh_ in self.meshs],
            )
        )

    def load_from_data(self, data: dict):
        """
        Load the Map from a dictionary containing its state.

        Parameters
        ----------
        data : dict
            Dictionary containing the state of the Map. It must contain the
            following keys:

            - map_mesh: A mesh.Mesh instance representing the map mesh.
            - raster: A rasterio.Dataset instance representing the raster.
            - meshs: A list of mesh.Mesh combined with the map mesh.

        Notes
        -----
        This method is used to load a Map from a dictionary containing its state.
        The state of the Map is composed of its map_mesh and raster.
        """
        self.map_mesh = data["map_mesh"]
        self.raster = data["raster"]
        self.dem_band = self.raster.read(1)
        self.meshs = data["meshs"]

    def copy_raster(self):
        """
        Copy the raster of the Map into a new rasterio.Dataset instance.

        Returns
        -------
        rasterio.Dataset
            A new rasterio.Dataset instance containing a copy of the Map's
            raster.
        """
        mem = rasterio.MemoryFile()
        mem = mem.open(**self.raster.profile.copy())
        mem.write(self.raster.read(1), 1)
        return mem

    def scale_raster(self, scale_factor: int = 1, new_filename: str = None):
        """
        Scales the raster by the given `scale_factor`.

        Parameters
        ----------
        scale_factor : int, optional
            The factor by which to scale the raster. Default is 1.
        new_filename : str, optional
            The filename to save the scaled raster to. If `None`, a temporary file is created.
            Default is `None`.

        Returns
        -------
        None

        """

        # scale raster
        band = self.raster.read(
            out_shape=(
                self.raster.count,
                int(self.raster.height * scale_factor),
                int(self.raster.width * scale_factor),
            ),
            resampling=Resampling.bilinear,
        )[0]

        # scale transform
        transform = self.raster.transform * self.raster.transform.scale(
            (self.raster.width / band.shape[-1]), (self.raster.height / band.shape[-2])
        )
        profile = self.raster.profile.copy()
        profile.update({"height": band.shape[0], "width": band.shape[1], "transform": transform})

        mem = rasterio.MemoryFile()
        mem = mem.open(**profile)
        mem.write(band, 1)

        self.raster.close()

        self.raster = mem
        self.dem_band = band

    def compute_dem_mesh(
        self, elevation_scale: float = 0.02, raster_band: np.ndarray = None
    ) -> mesh.Mesh:
        """
        Compute a mesh from the DEM (Digital Elevation Model).

        Parameters
        ----------
        elevation_scale : float, optional
            The scale factor to apply to the elevation values. Defaults to 0.02.
        Returns
        -------
        mesh.Mesh
            A mesh object representing the 3D mesh of the DEM.
        """
        band = self.raster.read(1) if not isinstance(raster_band, np.ndarray) else raster_band
        grid = (
            gg.visualization.create_dem_3d(
                dem=band * elevation_scale, extent=[0, band.shape[1], 0, band.shape[0]], res=1
            )
            .extract_geometry()
            .triangulate()
        )
        assert grid.is_all_triangles
        faces = grid.faces.reshape(-1, 4)[:, 1:4]
        points = grid.points
        map3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            map3d.vectors[i] = points[f]
        return map3d

    def add_text(
        self,
        text: str,
        x: float,
        y: float,
        fonts_file: str,
        inplace: bool = False,
        text_width: int = 100,
        not_world_coords=False,
        custom_altitude=None,
        **kwargs_extrude_text
    ) -> "Map":
        """
        Adds text to the map mesh.

        Parameters
        ----------
        text : str
            The text to be added.
        x : float
            The x-coordinate of the text in the world coordinates.
        y : float
            The y-coordinate of the text in the world coordinates.
        fonts_file : str
            The path to the font file.
        inplace : bool, optional
            If True, the text is added to the map mesh inplace, by default False.
        text_width : int, optional
            text width in the map mesh, by default 100
        Returns
        -------
        Map
            The updated map with the added text.
        """
        if not custom_altitude:
            altitude = list(self.raster.sample([(x, y)], 1))[0][0]
        else:
            altitude = custom_altitude
        text_3D = extrude_text(
            font_path=fonts_file, text=text, extrusion_height=30, **kwargs_extrude_text
        )
        tmax_x, tmax_y, tmax_z = text_3D.max_
        mmax_x, mmax_y, mmax_z = self.map_mesh.max_

        # translate text to the bottom left corner
        text_3D.translate(-text_3D.min_)
        text_3D.update_min()

        # scale text to a given width
        scale_factor = text_width / tmax_x
        text_3D.x *= scale_factor
        text_3D.y *= scale_factor

        # scale text to a given altitude
        altitude = int(
            (altitude / self.dem_band.max()) * self.map_mesh.max_[2]
        )  # scale altitude to max z position in the map mesh
        scale_factor_z = altitude / tmax_z
        text_3D.z *= scale_factor_z

        # get the x,y raster coords from the world coords
        if not not_world_coords:
            x, y = self.raster.index(x, y)
            print(1)
        # move the text to the given coordinates
        text_3D.translate([y, mmax_y - x, mmax_z / 15])

        if inplace:
            self.meshs.append(text_3D)
        else:
            new_Map = self.copy()
            new_Map.meshs.append(text_3D)
            return new_Map

    def add_mesh(
        self,
        mesh_to_add: mesh.Mesh,
        x: float,
        y: float,
        inplace: bool = False,
        not_world_coords=False,
        altitude_change_value=0,
    ) -> "Map":
        """
        Adds a mesh to the map mesh at a given world coordinate.

        Parameters
        ----------
        mesh_to_add : mesh.Mesh
            The mesh to be added.
        x : float
            The x-coordinate of the mesh in the world coordinates.
        y : float
            The y-coordinate of the mesh in the world coordinates.
        inplace : bool, optional
            If True, the mesh is added to the map mesh inplace, by default False.
        not_world_coords : bool, optional
            If True, the mesh is added to the map mesh in the raster coordinates, by default False

        Returns
        -------
        Map
            The updated Map with the added mesh.
        """
        altitude = list(self.raster.sample([(x, y)], 1))[0][0]
        altitude += altitude_change_value
        if not not_world_coords:
            x, y = self.raster.index(x, y)
        mmax_x, mmax_y, mmax_z = self.map_mesh.max_
        # scale text to a given altitude
        altitude = int((altitude / self.dem_band.max()) * mmax_z)
        mesh_to_add = mesh.Mesh(mesh_to_add.data.copy())
        mesh_to_add.translate(-mesh_to_add.min_)
        mesh_to_add.update_min()

        mesh_to_add.translate([y, mmax_y - x, altitude])

        if inplace:
            self.meshs.append(mesh_to_add)
        else:
            new_Map = self.copy()
            new_Map.meshs.append(mesh_to_add)
            return new_Map

    def add_geometry(
        self,
        geometry_dataframe: gpd.GeoDataFrame,
        extrusion_height: int,
        inplace: bool = False,
        **kwargs
    ) -> "Map":
        """
        Adds extruded geometries to the map mesh.

        Parameters
        ----------
        geometry_dataframe : geopandas.GeoDataFrame
            The geometries to be extruded
        extrusion_height : int
            The height of the extrusion
        inplace : bool, optional
            Whether to update the map mesh in place, by default False
        kwargs : dict
            Additional keyword arguments to be passed to `extrude_geometries`

        Returns
        -------
        Map
            A new Map instance if `inplace` is False, None otherwise
        """
        # assert geometry_dataframe.crs == self.raster.crs

        new_band = extrude_geometries(
            geometry_dataframe, self.raster, self.dem_band, extrusion_height, **kwargs
        )
        if inplace:
            self.map_mesh = self.compute_dem_mesh(
                raster_band=new_band, elevation_scale=self.elevation_scale
            )
        else:
            new_Map = self.copy()
            new_Map.map_mesh = self.compute_dem_mesh(
                raster_band=new_band, elevation_scale=self.elevation_scale
            )
            return new_Map

    def add_plateau(self, height: int, inplace: bool = False) -> Optional["Map"]:
        """
        Adds a plateau mesh to the map.

        Parameters
        ----------
        height : int
            The height of the plateau in millimeters.
        inplace : bool, optional
            Whether to update the map mesh in place, by default False.

        Returns
        -------
        Optional[Map]
            A new Map instance if `inplace` is False, None otherwise.
        """
        # create a cube as a plateau mesh
        plateau = generate_cube(int(self.map_mesh.max_[0]))

        # scale the plateau mesh to desired height
        desired_height = 5  # mm
        plateau.z *= desired_height / plateau.z.max()

        # scale the plateau mesh to match the map mesh y dimension
        plateau.y *= self.map_mesh.max_[1] / plateau.y.max()

        # translate the plateau mesh to the desired height
        plateau.translate([0, 0, -height])

        if inplace:
            # append the plateau mesh to the map meshes if inplace is True
            self.meshs.append(plateau)
        else:
            # create a new map instance, append the plateau mesh to its meshes, and return it
            new_Map = self.copy()
            new_Map.meshs.append(plateau)
            return new_Map

    def cleanup(self):
        """
        Close the raster file.
        """
        self.raster.close()

    def generate_mesh(self) -> mesh.Mesh:
        """
        Generate a mesh by concatenating the meshes of the map and its plates.

        Returns
        -------
        mesh.Mesh
            A new mesh containing the data of all the meshes of the map and its plates.
        """
        return mesh.Mesh(
            np.concatenate(
                [*(mesh_.data.copy() for mesh_ in self.meshs), self.map_mesh.data.copy()]
            )
        )

    def toPolyData(self):
        """
        Parse the mesh to a pyvista PolyData object.
        """
        return parse_Mesh_to_PolyData(self.generate_mesh())

    def save(self, filename: str):
        """
        Save the map mesh to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save to
        """
        self.toPolyData().save(filename)

    def clean_meshs(self):
        """
        Method to clean the meshs list by setting it to an empty list.
        """
        self.meshs = []

    def plot(self, **kwargs):
        """
        Plot the map mesh.
        """
        self.toPolyData().plot(**kwargs)
