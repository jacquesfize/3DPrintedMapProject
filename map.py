import atexit

import gemgis as gg
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from stl import mesh
from utils import extrude_geometries, extrude_text, load_raster_data

np.bool = np.bool_  # workaround for numpy bug


class Map:

    def __init__(self, dem_filename: str = "", scale_factor: int = 1, data: dict = None):
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

            self.map_mesh = self.compute_dem_mesh()
        else:
            self.load_from_data(data)

        atexit.register(self.cleanup)

    def copy(self):
        return Map(
            data=dict(
                map_mesh=mesh.Mesh(self.map_mesh.data.copy()),
                raster=self.copy_raster(),
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

        Notes
        -----
        This method is used to load a Map from a dictionary containing its state.
        The state of the Map is composed of its map_mesh and raster.
        """
        self.map_mesh = data["map_mesh"]
        self.raster = data["raster"]
        self.dem_band = self.raster.read(1)

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
        self, text: str, x: float, y: float, fonts_file: str, inplace: bool = False
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

        Returns
        -------
        Map
            The updated map with the added text.
        """
        altitude = list(self.raster.sample([(x, y)], 1))[0][0]
        band = self.dem_band
        text_3D = extrude_text(
            font_path=fonts_file,
            text=text,
            extrusion_height=30,
        )
        tmax_x, tmax_y, tmax_z = text_3D.max_
        mmax_x, mmax_y, mmax_z = self.map_mesh.max_

        # translate text to the bottom left corner
        text_3D.translate(-text_3D.min_)
        text_3D.update_min()

        # scale text to a given width
        wanted_width = mmax_x / 15
        scale_factor = wanted_width / tmax_x
        text_3D.x *= scale_factor
        text_3D.y *= scale_factor

        # scale text to a given altitude
        altitude = int(
            (altitude / band.max()) * self.map_mesh.max_[2]
        )  # scale altitude to max z position in the map mesh
        scale_factor_z = altitude / tmax_z
        text_3D.z *= scale_factor_z

        # get the x,y raster coords from the world coords
        x, y = self.raster.index(x, y)
        # move the text to the given coordinates
        text_3D.translate([y, mmax_y - x, mmax_z / 15])

        if inplace:
            self.map_mesh = mesh.Mesh(
                np.concatenate([text_3D.data.copy(), self.map_mesh.data.copy()])
            )
        else:
            new_Map = self.copy()
            new_Map.map_mesh = mesh.Mesh(
                np.concatenate([text_3D.data.copy(), self.map_mesh.data.copy()])
            )
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
            self.map_mesh = self.compute_dem_mesh(raster_band=new_band)
        else:
            new_Map = self.copy()
            new_Map.map_mesh = self.compute_dem_mesh(raster_band=new_band)
            return new_Map

    def cleanup(self):
        """
        Close the raster file.
        """
        self.raster.close()

    def save(self, filename: str):
        """
        Save the map mesh to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save to
        """
        self.map_mesh.save(filename)