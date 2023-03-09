import numpy as np


class Splat:
    def __init__(self, center, normal, radius):
        self.center = center
        self.normal = normal
        self.radius = radius
        self.radius_squared = radius * radius

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius


class World:
    def __init__(self):
        self.splats = []
        self.sink = None

    def add_splat(self, splat):
        self.splats.append(splat)

    def intersection(self):
        return False

    def construct_world_splat(self):
        self.center = np.ndarray((len(self.splats), 3))
        self.normal = np.ndarray((len(self.splats), 3))
        self.radius = np.ndarray(len(self.splats))
        self.radius_squared = np.ndarray(len(self.splats))

        for i, splat in enumerate(self.splats):
            self.center[i] = splat.center
            self.normal[i] = splat.normal
            self.radius[i] = splat.radius
            self.radius_squared[i] = splat.radius_squared