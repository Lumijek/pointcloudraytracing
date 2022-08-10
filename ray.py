from vec3 import Point3, Vec3


class Ray:
    def __init__(self, origin, direction):
        self.orig = origin
        self.dir = Vec3.unit_vector(direction)

    def origin(self):
        return self.orig

    def direction(self):
        return self.dir

    def at(self, t):
        return self.orig + self.dir * t
