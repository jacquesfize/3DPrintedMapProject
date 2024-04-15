import numpy as np
from stl import mesh


def generate_cube(width: int = 1) -> mesh.Mesh:
    """
    Generate a cube mesh.

    Parameters
    ----------
    width : int, optional
        Width of the cube, by default 1.

    Returns
    -------
    mesh.Mesh
        The generated cube mesh.
    """
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    )

    if width != 1:
        vertices *= width

    faces = np.array(
        [
            [0, 3, 1],
            [1, 3, 2],
            [0, 4, 7],
            [0, 7, 3],
            [4, 5, 6],
            [4, 6, 7],
            [5, 1, 2],
            [5, 2, 6],
            [2, 3, 7],
            [2, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
        ]
    )

    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        cube.vectors[i] = vertices[f, :]

    return cube


def generate_pyramide(width: int = 1) -> mesh.Mesh:
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
    mesh.Mesh
        The generated pyramide
         mesh.
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]])

    faces = np.array([[0, 4, 2], [0, 1, 4], [1, 3, 4], [2, 4, 3], [0, 1, 3], [0, 2, 3]])

    pyramide = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        pyramide.vectors[i] = vertices[f, :]

    return pyramide


def generate_cylinder(num_points: int = 100, radius: float = 3, height: float = 5) -> mesh.Mesh:
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
    mesh.Mesh
        The generated cylinder mesh.
    """
    angles = np.linspace(0, 2 * np.pi, num_points)
    points_base = np.array(
        [[np.cos(angle) * radius, np.sin(angle) * radius, 0] for angle in angles]
    )
    points_top = np.array(
        [[np.cos(angle) * radius, np.sin(angle) * radius, height] for angle in angles]
    )

    vectors = []
    for i in range(points_base.shape[0]):
        if i + 1 < points_base.shape[0]:
            # side of the cylinder
            vectors.append([points_base[i], points_base[i + 1], points_top[i]])
            vectors.append([points_top[i + 1], points_top[i], points_base[i + 1]])
            # base and top of the cylinder
            vectors.append([points_base[i + 1], points_base[i], [0, 0, 0]])
            vectors.append([points_top[i + 1], points_top[i], [0, 0, height]])
    vectors = np.array(vectors)
    mesh_ = mesh.Mesh(np.zeros(vectors.shape[0], dtype=mesh.Mesh.dtype))
    mesh_.vectors = vectors
    return mesh_


def generate_torus(
    num_theta: int = 30, num_phi: int = 30, major_radius: float = 2, minor_radius: float = 1
) -> mesh.Mesh:
    """
    Generate a torus mesh.

    Parameters
    ----------
    num_theta : int, optional
        Number of points around the torus, by default 30.
    num_phi : int, optional
        Number of points along the torus, by default 30.
    major_radius : float, optional
        Major radius of the torus, by default 2.
    minor_radius : float, optional
        Minor radius of the torus, by default 1.

    Returns
    -------
    mesh.Mesh
        The generated torus mesh.
    """
    theta = np.linspace(0, 2 * np.pi, num_theta)
    phi = np.linspace(0, 2 * np.pi, num_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = (major_radius + minor_radius * np.cos(phi_grid)) * np.cos(theta_grid)
    y = (major_radius + minor_radius * np.cos(phi_grid)) * np.sin(theta_grid)
    z = minor_radius * np.sin(phi_grid)

    vertices = np.zeros((num_theta * num_phi, 3))
    faces = np.zeros((2 * num_theta * num_phi, 3), dtype=np.uint32)

    for i in range(num_phi):
        for j in range(num_theta):
            vertices[i * num_theta + j] = [x[i, j], y[i, j], z[i, j]]
            next_i = (i + 1) % num_phi
            next_j = (j + 1) % num_theta
            faces[2 * (i * num_theta + j)] = [
                i * num_theta + j,
                i * num_theta + next_j,
                next_i * num_theta + j,
            ]
            faces[2 * (i * num_theta + j) + 1] = [
                i * num_theta + next_j,
                next_i * num_theta + next_j,
                next_i * num_theta + j,
            ]

    torus_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        torus_mesh.vectors[i] = vertices[f, :]
    return torus_mesh
