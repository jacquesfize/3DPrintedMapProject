import os
from typing import Any, Iterable, List, Tuple, Union

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
from stl import mesh
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


def extrude_text_in_raster(
    raster_dataset: rasterio.DatasetReader,
    band: np.ndarray,
    text_with_pos: List[Tuple[str, Tuple[float, float]]],
    args_image_font: Iterable[Any] = [],
    text_align: str = "left",
    extrusion_height: int = 100,
    upscale_factor: int = 1,
):
    """
    Extrudes text into a raster dataset.

    Parameters
    ----------
    raster_dataset : rasterio.DatasetReader
        The raster dataset to extrude the text into.
    band : np.ndarray
        The band of the raster dataset to extrude the text into.
    text_with_pos : List[Tuple[str, Tuple[float, float]]]
        A list of tuples containing the text to extrude and its position in the raster dataset.
    args_image_font : Iterable[Any], optional
        Optional arguments for creating the image font. Default is an empty iterable.
    text_align : str, optional
        The alignment of the text in the raster dataset. Default is "left".
    extrusion_height : int, optional
        The height of the extrusion in meters. Default is 100.

    Returns
    -------
    np.ndarray
        The new raster band with the extruded text.
    """
    image: Image = Image.fromarray(band)
    draw: ImageDraw = ImageDraw.Draw(image)
    font: ImageFont = ImageFont.truetype(*args_image_font)

    for text, (x, y), altitude in text_with_pos:
        draw.text(
            [i * upscale_factor for i in raster_dataset.index(x, y)],
            text,
            font=font,
            align=text_align,
            fill=altitude + extrusion_height,
        )

    return np.asarray(image)


def extrude_text(
    text: str,
    font_path: str,
    font_size: int = 40,
    extrusion_height: int = 100,
) -> mesh.Mesh:
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
    mesh.Mesh
        The extruded 3D mesh.
    """
    fnt = ImageFont.truetype(font_path, font_size)
    _, _, w, h = fnt.getbbox(text)
    margin = (w // 5, h // 5)
    text_image = Image.new("L", (w + margin[0] * 2, h + margin[1] * 2), 0)
    modifier = ImageDraw.Draw(text_image)
    modifier.text(margin, text, font=fnt, fill=255, stroke_width=4)
    letter_image_matrix = np.asarray(text_image)

    xyz = np.asarray([letter_image_matrix for _ in range(5)])
    xyz[-1] = 0
    xyz[0] = 0
    verts, faces, _, _ = marching_cubes(xyz, spacing=[extrusion_height / 3, 1, 1])
    obj3D = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        obj3D.vectors[i] = verts[f]

    obj3D.rotate([0, 0, 1], np.deg2rad(90))
    obj3D.rotate([1, 0, 0], np.deg2rad(90))
    obj3D.rotate([0, 0, 1], np.deg2rad(90))
    obj3D = clean_mesh(obj3D)
    return obj3D


def clean_mesh(_mesh: mesh.Mesh, radius: int = 1000) -> mesh.Mesh:
    mesh_pyvista = parse_Mesh_to_PolyData(_mesh)
    mesh_pyvista = mesh_pyvista.fill_holes(radius)
    return parsePolydata_to_Mesh(mesh_pyvista)


def scale_mesh(mesh_to_scale, desired_width):
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
    mesh.Mesh
        The scaled mesh.
    """
    new_mesh = mesh.Mesh(mesh_to_scale.data.copy())
    new_mesh.translate(-new_mesh.min_)
    new_mesh.update_min()

    scale_factor = desired_width / new_mesh.max_[0]
    new_mesh.x *= scale_factor
    new_mesh.y *= scale_factor
    new_mesh.z *= scale_factor

    return new_mesh


def parse_Mesh_to_PolyData(_mesh: mesh.Mesh) -> pv.PolyData:
    points = _mesh.vectors.reshape(-1, 3)
    faces = np.arange(points.shape[0], dtype=np.uint32).reshape(-1, 3)
    faces = np.hstack(
        [(np.ones(len(faces), dtype=np.uint32) * 3).reshape(-1, 1), faces], dtype=np.uint32
    )
    cloud = pv.PolyData(points, faces)
    return cloud


def parsePolydata_to_Mesh(_polydata: pv.PolyData) -> mesh.Mesh:
    _polydata = _polydata.triangulate()
    faces = _polydata.faces.reshape(-1, 4)[:, 1:4]
    points = _polydata.points
    map3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        map3d.vectors[i] = points[f]
    return map3d
