class Splat:
    def __init__(self, center, normal, radius, points):
        self.center = center
        self.normal = normal
        self.radius = radius
        self.points = points


class World:
    def __init__(self):
        self.splats = []

    def add_splat(self, splat):
        self.splats.append(splat)

    def intersection(self):
        return False