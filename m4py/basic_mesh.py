import numpy as np
import pyvista as pv
from .utils import scale_mesh


def generate_cube(width: int = 1) -> pv.PolyData:
    """
    Generate a cube mesh.

    Parameters
    ----------
    width : int, optional
        Width of the cube, by default 1.

    Returns
    -------
   pyvista.PolyData
        The generated cube mesh.
    """
    

    return pv.Cube(x_length=width, y_length=width, z_length=width)


def generate_pyramide(width: int = 1) -> pv.PolyData:
    """
    Generate a pyramide
     mesh.

    Parameters
    ----------
    width : int, optional
        Width of the pyramide
        , by default 1.

    Returns
    -------
    pyvista.PolyData
        The generated pyramide
         mesh.
    """
    vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0.5, 0.5, 1]])
    pyramid_mesh = pv.Pyramid(vertices)
    return pyramid_mesh if width == 1 else pyramid_mesh.scale([width, width, width])


def generate_cylinder(num_points: int = 100, radius: float = 3, height: float = 5) -> pv.PolyData:
    """
    Generate a cylinder mesh.

    Parameters
    ----------
    num_points : int, optional
        Number of points on the circular base of the cylinder, by default 100.
    radius : float, optional
        Radius of the circular base of the cylinder, by default 3.
    height : float, optional
        Height of the cylinder, by default 5.

    Returns
    -------
    pyvista.PolyData
        The generated cylinder mesh.
    """
    return pv.Cylinder(direction=[0, 0, 1], radius=radius, height=height,resolution=num_points)


def generate_torus(ring_radius: float = 10, cross_section_radius: float = 5
                   ) -> pv.PolyData:
    """
    Generate a torus mesh.

    Parameters
    ----------
    ring_radius : float, optional
        Radius of the ring of the torus, by default 10.
    crossection_radius : float, optional
        Radius of the cross section of the torus, by default 5.

    Returns
    -------
    pyvista.PolyData
        The generated torus mesh.
    """
    return pv.ParametricTorus(ringradius=ring_radius, crosssectionradius=cross_section_radius)


