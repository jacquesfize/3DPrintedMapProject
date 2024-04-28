import pyvista as pv

class SIZE:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Bounds:
    def __init__(self, mesh:pv.PolyData):
        self.mesh = mesh
    @property
    def x_min(self):
        return self.mesh.points[:,0].min()
    @property
    def y_min(self):
        return self.mesh.points[:,1].min()
    @property
    def z_min(self):
        return self.mesh.points[:,2].min()
    @property
    def x_max(self):
        return self.mesh.points[:,0].max()
    @property
    def y_max(self):
        return self.mesh.points[:,1].max()
    @property
    def z_max(self):
        return self.mesh.points[:,2].max()