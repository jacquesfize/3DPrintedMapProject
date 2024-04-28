import os
from typing import Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import pyvista as pv
import rasterio
from freetype import *
from PIL import Image, ImageDraw, ImageFont
from rasterio import features
from shapely import Point, Polygon
from shapely.geometry.base import BaseGeometry
from skimage.measure import marching_cubes
from tqdm.notebook import tqdm


def load_raster_data(
    filename: str, return_band1: bool = False
):  # -> Union[Tuple[rasterio.DatasetReader, np.ndarray], rasterio.DatasetReader]:
    """
    Load raster data from a file.

    Parameters
    ----------
    filename : str
        The path to the raster file.
    return_band1 : bool, optional
        Whether to return only the first band of the raster data. Defaults to False.

    Returns
    -------
    tuple or rasterio.DatasetReader
        If `return_band1` is False, returns a tuple containing the raster dataset and the first band of the raster data. If `return_band1` is True, returns only the raster dataset.

    Raises
    ------
    AssertionError
        If the file does not exist.
    """
    if isinstance(filename, str):
        assert os.path.exists(filename)

    raster_dataset = rasterio.open(filename)
    if return_band1:
        return raster_dataset, raster_dataset.read(1)
    return raster_dataset


def parse_geometries_to_rasterize_format(
    raster_dataset: rasterio.DatasetReader,
    geometries: Iterable[BaseGeometry],
    extrusion_height: int = 100,
    buffer_size: int = 50,
) -> List[Tuple[Polygon, float]]:
    """Parse a sequence of geometries to a format suitable for rasterization.

    Parameters:
    ----------
    raster_dataset : rasterio.DatasetReader
        A raster dataset.
    geometries : Iterable[BaseGeometry]
        A sequence of geometries.
    extrusion_height : int, optional
        The height to extrude each geometry by. Defaults to 100.
    buffer_size : int, optional
        The size of the buffer to add to each geometry. Defaults to 50.

    Returns:
    -------
    List[Tuple[Polygon, float]]
        A list of tuples, where each tuple contains a polygon and its height.
    """
    # List to store the resulting shapes.
    shapes: List[Tuple[Polygon, float]] = []

    # Iterate over the geometries.
    for geom in tqdm(geometries):
        # Recover the x,y coordinates of the geometry in a list of tuples.
        coords_list: List[Tuple[float, float]] = list(zip(*geom.coords.xy))

        # Extract altitude values from the raster dataset for each coordinate.
        heights = [
            altitude[0] + extrusion_height for altitude in raster_dataset.sample(coords_list)
        ]

        # Extend the shapes list with a tuple for each coordinate, consisting of a
        # buffered point and its corresponding altitude.
        shapes.extend(
            [
                (Point(x, y).buffer(buffer_size), heights[ix])
                for ix, (x, y) in enumerate(coords_list)
            ]
        )

    return shapes


def extrude_geometries(
    gdf: gpd.GeoDataFrame,
    raster_dataset: rasterio.DatasetReader,
    raster_band: np.ndarray,
    in_place: bool = False,
    extrusion_height: int = 100,
    buffer_size: int = 50,
    geom_column: str = "geometry",
    transform=None,
    shape=None,
) -> np.ndarray:
    """Extrude geometries onto a raster band.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with geometries to be extruded.
    raster_dataset : DatasetReader
        Raster dataset.
    raster_band : numpy.ndarray
        NumPy array representing the raster band.
    in_place : bool, optional
        Modify the input raster band in place (default False).
    extrusion_height : int, optional
        Height (in meters) to extrude for each geometry by (default 100). Will be add to existing alititudes sample from the raster.
    buffer_size : int, optional
        Size of the buffer to add to each geometry (default 50).
    geometry_column : str, optional
        Name of the geometry column in the GeoDataFrame (default "geometry").

    Returns
    -------
    numpy.ndarray
        Modified or new raster band with extruded geometries.
    """

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(raster_band, np.ndarray)
    assert geom_column in gdf

    # Reproject coordinates if CRS are different
    if gdf != raster_dataset.crs:
        gdf = gdf.to_crs(raster_dataset.crs)

    out_band = raster_band if in_place else raster_band.copy()
    shapes = parse_geometries_to_rasterize_format(
        raster_dataset=raster_dataset,
        geometries=gdf[geom_column].values,
        extrusion_height=extrusion_height,
        buffer_size=buffer_size,
    )
    features.rasterize(
        shapes,
        out=out_band,
        out_shape=raster_dataset.shape if not shape else shape,
        transform=raster_dataset.transform if not transform else transform,
    )
    return out_band


def extrude_text(
    text: str,
    font_path: str,
    font_size: int = 40,
    extrusion_height: int = 100,
) -> pv.PolyData:
    """
    Extrudes a text into a 3D mesh.

    Parameters
    ----------
    font_path : str
        The path to the font file.
    text : str
        The text to extrude.
    space_between_char : int, optional
        The space between characters in pixels. Default is 10.
    space_size : int, optional
        The size of the space in pixels. Default is 40.
    char_size : int, optional
        The size of the characters in pixels. Default is 10000.
    extrusion_height : int, optional
        The height of the extrusion in meters. Default is 100.

    Returns
    -------
    pv.PolyData
        The extruded 3D mesh.
    """
    fnt = ImageFont.truetype(font_path, font_size)
    _, _, w, h = fnt.getbbox(text)
    margin = (w // 5, h // 5)
    text_image = Image.new("L", (w + margin[0] * 2, h + margin[1] * 2), 0)
    modifier = ImageDraw.Draw(text_image)
    modifier.text(margin, text, font=fnt, fill=255, stroke_width=3)
    letter_image_matrix = np.asarray(text_image)

    xyz = np.asarray([letter_image_matrix for _ in range(5)])
    xyz[-1] = 0
    xyz[0] = 0
    verts, faces, _, _ = marching_cubes(xyz, spacing=[extrusion_height / 3, 1, 1])

    faces = parse_faces_to_pyvista_faces_format(faces)
    text3D = pv.PolyData(verts, faces)
    text3D.rotate_y(90, inplace=True)
    text3D.rotate_z(180, inplace=True)
    text3D.rotate_x(180, inplace=True)
    return text3D


def clean_mesh(_mesh: pv.PolyData, radius: int = 1000) -> pv.PolyData:
    return _mesh.fill_holes(radius)


def scale_mesh(mesh_to_scale: pv.PolyData, desired_width: float) -> pv.PolyData:
    """
    Scales the given mesh to have the specified desired width.

    Parameters
    ----------
    mesh : mesh.Mesh
        The mesh to be scaled.
    desired_width : float
        The desired width of the output mesh.

    Returns
    -------
    pyvista.PolyData
        The scaled mesh.
    """
    new_mesh = mesh_to_scale.copy()

    new_mesh.translate(-new_mesh.points.min(axis=0))

    scale_factor = desired_width / new_mesh.points.max(axis=0)[0]
    new_mesh.scale([scale_factor, scale_factor, scale_factor], inplace=True)
    return new_mesh


def parse_faces_to_pyvista_faces_format(faces: np.ndarray) -> np.ndarray:
    """
    Convert a list of face indices to the format required by PyVista.

    Parameters
    ----------
    faces : array-like
        List of face indices.

    Returns
    -------
    numpy.ndarray
        The face indices in the format required by PyVista.
    """
    return np.hstack(
        [(np.ones(len(faces), dtype=np.int32) * 3).reshape(-1, 1), faces], dtype=np.int32
    )
