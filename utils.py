from typing import Any, Iterable, List, Tuple, Union
import rasterio
from rasterio import features
import geopandas as gpd
from shapely import Point, Polygon
from shapely.geometry.base import BaseGeometry
from tqdm.notebook import tqdm
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os


def load_raster_data(
    filename: str, return_band1: bool = False
) -> Union[Tuple[rasterio.DatasetReader, np.ndarray], rasterio.DatasetReader]:
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
            altitude[0] + extrusion_height
            for altitude in raster_dataset.sample(coords_list)
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
