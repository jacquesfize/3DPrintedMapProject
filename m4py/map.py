import atexit
from typing import Any, List, Tuple

import geopandas as gpd
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from .models import Bounds, SIZE
from .utils import (
    extrude_geometries,
    extrude_text,
    load_raster_data,
    parse_faces_to_pyvista_faces_format,
)
import mapa
import pyvista as pv


class Map:
    """
    Class use to create a 3DMap object
    """

    map_mesh: pv.PolyData  # Contains the DEM (Digital Elevation Model) 3D mesh
    meshes: List[pv.PolyData]  # Contains the 3D mesh added to the map
    elevation_scale: float  # The scale factor to apply to the elevation values
    height_plateau: float  # The height of the plateau
    raster: rasterio.DatasetReader  # Raster dataset from which is extracted the DEM
    dem_band: (
        np.ndarray
    )  # The DEM (Digital Elevation Model) 2D array used to generate the `map_mesh`

    def __init__(
        self,
        dem_filename: str = "",
        scale_factor: int = 1,
        data: dict = None,
        elevation_scale: float = 0.02,
        height_plateau: float = 4,
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
        height_plateau : float, optional
            The height of the plateau. Default is 4 (is given in mm).
        Notes
        -----
        If `data` is not None, it calls `load_from_data(data)`.
        If `data` is None, it loads the raster data and applies the scale factor if specified.
        Registers `cleanup` to be called at exit.
        """
        if not data:
            self.meshes = []
            self.elevation_scale = elevation_scale
            self.height_plateau = height_plateau

            self.raster, self.dem_band = load_raster_data(dem_filename, True)
            mem = rasterio.MemoryFile()
            mem = mem.open(**self.raster.profile.copy())
            mem.write(self.dem_band, 1)
            self.raster.close()
            self.raster = mem
            if scale_factor != 1:
                self.scale_raster(scale_factor)

            self.map_mesh = self.compute_dem_mesh(elevation_scale=elevation_scale)

        else:
            self.load_from_data(data)

        self.bounds = Bounds(self.map_mesh)
        atexit.register(self.cleanup)

    def copy(self):
        return Map(
            data=dict(
                map_mesh=self.map_mesh.copy(),
                raster=self.copy_raster(),
                meshes=[mesh_.copy() for mesh_ in self.meshes],
                height_plateau=self.height_plateau,
                elevation_scale=self.elevation_scale,
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

            - map_mesh: A pv.PolyData instance representing the map mesh.
            - raster: A rasterio.Dataset instance representing the raster.
            - meshes: A list of pv.PolyData combined with the map mesh.

        Notes
        -----
        This method is used to load a Map from a dictionary containing its state.
        The state of the Map is composed of its map_mesh and raster.
        """
        self.map_mesh = data["map_mesh"]
        self.raster = data["raster"]
        self.dem_band = self.raster.read(1)
        self.meshes = data["meshes"]
        self.elevation_scale = data["elevation_scale"]
        self.height_plateau = data["height_plateau"]

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
        self, elevation_scale: float = 0.02, raster_band: np.ndarray = None, **mapa_kwargs
    ) -> pv.PolyData:
        """
        Compute a mesh from the DEM (Digital Elevation Model).

        Parameters
        ----------
        elevation_scale : float, optional
            The scale factor to apply to the elevation values. Defaults to 0.02.
        Returns
        -------
        pv.PolyData
            A mesh object representing the 3D mesh of the DEM.
        """
        band = self.raster.read(1) if not isinstance(raster_band, np.ndarray) else raster_band
        desired_size = mapa_kwargs.get("desired_size", SIZE(*band.shape))

        data_mesh = mapa.compute_all_triangles(
            band,
            desired_size=desired_size,
            elevation_scale=elevation_scale,
            z_scale=1,
            z_offset=self.height_plateau,
            **mapa_kwargs,
        ).reshape(-1, 3)

        faces = np.arange(data_mesh.shape[0], dtype=np.int32).reshape(-1, 3)
        faces = parse_faces_to_pyvista_faces_format(faces)
        map3D_object = pv.PolyData(data_mesh, faces)

        map3D_object.rotate_z(-90, inplace=True)
        map3D_object.translate(-map3D_object.points.min(axis=0), inplace=True)
        return map3D_object

    def add_text(
        self,
        text: str,
        x: float,
        y: float,
        fonts_file: str,
        inplace: bool = False,
        text_width: int = 100,
        raster_coords=False,
        custom_altitude=None,
        epsg_src=None,
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
            The width of the text in the map mesh, by default 100.
        raster_coords : bool, optional
            If True, the text is added to the map mesh in the raster coordinates, by default False.
        custom_altitude : float, optional
            The altitude at which the text should be added, by default None.
        epsg_src : str, optional
            The EPSG code of the coordinate system in which the x and y coordinates are given, by default None.
        **kwargs_extrude_text
            Additional keyword arguments to be passed to the extrude_text function.

        Returns
        -------
        Map
            The updated map with the added text.
        """
        if not custom_altitude:
            altitude = self.get_altitude((x, y), epsg_src=epsg_src)
        else:
            altitude = custom_altitude
        text_3D = extrude_text(
            font_path=fonts_file, text=text, extrusion_height=30, **kwargs_extrude_text
        )
        tmax_x, tmax_y, tmax_z = text_3D.points.max(axis=0)
        mmax_x, mmax_y, mmax_z = self.map_mesh.points.max(axis=0)

        # scale text to a given width
        scale_factor_xy = text_width / tmax_x

        # scale text to a given altitude
        altitude = int(
            (altitude / self.dem_band.max()) * mmax_z
        )  # scale altitude to max z position in the map mesh
        scale_factor_z = altitude / tmax_z

        text_3D.scale([scale_factor_xy, scale_factor_xy, scale_factor_z], inplace=True)
        # translate text to the bottom left corner
        text_3D.translate(-text_3D.points.min(axis=0), inplace=True)
        # get the x,y raster coords from the world coords
        if not raster_coords:
            x, y = self.realworld_to_raster_coords((x, y), epsg_src=epsg_src)
        # move the text to the given coordinates

        text_3D.translate([x, mmax_y - y, mmax_z / 15], inplace=True)

        if inplace:
            self.meshes.append(text_3D)
        else:
            new_Map = self.copy()
            new_Map.meshes.append(text_3D)
            return new_Map

    def add_mesh(
        self,
        mesh_to_add: pv.PolyData,
        x: float,
        y: float,
        z: float = 0,
        inplace: bool = False,
        raster_coords=False,
        altitude_change_value=0,
        epsg_src: str = None,
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
        z : float, optional
            The z-coordinate of the mesh in the world coordinates, by default 0. Ignored if raster_coords is False.
        inplace : bool, optional
            If True, the mesh is added to the map mesh inplace, by default False.
        raster_coords : bool, optional
            If True, the mesh is added to the map mesh in the raster coordinates, by default False.
        altitude_change_value : float, optional
            The altitude change value in the map mesh, by default 0.
        epsg_src : str, optional
            The EPSG code of the coordinate system in which the x and y coordinates are given, by default None.

        Returns
        -------
        Map
            The updated Map with the added mesh.
        """
        altitude = z
        if not raster_coords:
            altitude = self.get_altitude((x, y), epsg_src=epsg_src)
        altitude += altitude_change_value
        if not raster_coords:
            x, y = self.realworld_to_raster_coords((x, y), epsg_src=epsg_src)
        mmax_x, mmax_y, mmax_z = self.map_mesh.points.max(axis=0)
        # scale text to a given altitude
        mesh_to_add = mesh_to_add.copy()
        mesh_to_add.translate(-mesh_to_add.points.min(axis=0), inplace=True)

        mesh_to_add.translate([y, mmax_y - x, altitude], inplace=True)

        if inplace:
            self.meshes.append(mesh_to_add)
        else:
            new_Map = self.copy()
            new_Map.meshes.append(mesh_to_add)
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

    def reproject_coords(
        self, xy_coords: Tuple[float, float], epsg_src: str
    ) -> Tuple[float, float]:
        """
        Transforms the given coordinates from `epsg_src` to the CRS of the raster.

        Parameters
        ----------
        xy_coords : Tuple[float, float]
            The coordinates to transform (lat,lon).
        epsg_src : str
            The EPSG code of the source CRS.

        Returns
        -------
        Tuple[float, float]
            The reprojected coordinates.
        """
        transformer = Transformer.from_crs(epsg_src, self.raster.crs)
        return transformer.transform(xy_coords[1], xy_coords[0])

    def realworld_to_raster_coords(
        self, xy_coords: Tuple[float, float], epsg_src: str = None
    ) -> Tuple[float, float]:
        """
        Returns the raster coordinates based on real world coordinates.

        Parameters
        ----------
        xy_coords : Tuple[float, float]
            The real world coordinates of the point.
        epsg_src : str, optional
            The EPSG code of the coordinate reference system of the point.
            If `epsg_src` is given, `xy_coords` will be reprojected.
            Default is None.

        Returns
        -------
        Tuple[float, float]
            The raster coordinates of the point.
        """
        if epsg_src is not None:
            xy_coords = self.reproject_coords(xy_coords, epsg_src)
        return self.raster.index(*xy_coords)

    def get_altitude(self, xy_coords: Tuple[float, float], epsg_src: Any = None) -> float:
        """
        Returns the altitude (elevation) of a point.

        Parameters
        ----------
        xy_coords : Tuple[float, float]
            The real world coordinates of the point.
        epsg_src : str, optional
            The EPSG code of the coordinate reference system of the point.
            If `epsg_src` is given, `xy_coords` will be reprojected.
            Default is None.

        Returns
        -------
        float
            The altitude of the point.
        """
        if epsg_src:
            xy_coords = self.reproject_coords(xy_coords, epsg_src)
        altitude = list(self.raster.sample([xy_coords], 1))[0][0] + self.height_plateau
        return int((altitude / self.dem_band.max()) * self.map_mesh.points.max(axis=0)[2])

    def cleanup(self):
        """
        Close the raster file.
        """
        self.raster.close()

    def generate_mesh(self) -> pv.PolyData:
        """
        Generate a mesh by concatenating the meshes of the map and its plates.

        Returns
        -------
        pv.PolyData
            A new mesh containing the data of all the meshes of the map and its plates.
        """
        new_mesh: pv.PolyData = self.map_mesh.copy()
        return new_mesh.merge(self.meshes, inplace=True)

    def save(self, filename: str):
        """
        Save the map mesh to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save to
        """
        self.generate_mesh().save(filename)

    def clean_meshs(self):
        """
        Method to clean the meshs list by setting it to an empty list.
        """
        self.meshes = []

    def plot(self, **kwargs):
        """
        Plot the map mesh.
        """
        self.generate_mesh().plot(**kwargs)
