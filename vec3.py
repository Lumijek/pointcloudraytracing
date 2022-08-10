import numpy as np
from math import fabs, sqrt
from random import uniform


class Vec3:
    def __init__(self, e0=0, e1=0, e2=0):
        if type(e0) == np.ndarray:
            self.e = e0
        elif type(e0) == list:
            self.e = np.asarray(e0)
        else:
            self.e = np.array([e0, e1, e2], dtype=np.double)

    def x(self):
        return self.e[0]

    def y(self):
        return self.e[1]

    def z(self):
        return self.e[2]

    def __repr__(self):
        return f"({self.e[0]}, {self.e[1]}, {self.e[2]})"

    def __neg__(self):
        return Vec3(*(-self.e))

    def __getitem__(self, index):
        return self.e[index]

    def __setitem__(self, index, value):
        self.e[index] = value

    def __add__(self, v):
        return Vec3(*(self.e + v.e))

    def __sub__(self, v):
        return Vec3(*(self.e - v.e))

    def __mul__(self, t):
        if isinstance(t, Vec3):
            return Vec3(*(self.e * t.e))

        return Vec3(*(self.e * t))

    def __truediv__(self, t):
        return Vec3(*(self.e / t))

    def __iadd__(self, v):
        self.e += v.e
        return self

    def __imul__(self, t):
        if isinstance(t, vec3):
            self.e *= t.e
            return self
        self.e *= t
        return self

    def __itruediv__(self, t):
        self.e /= t
        return self

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    def length(self):
        return np.sqrt(self.length_squared())

    def length_squared(self):
        return self.e @ self.e

    def dot(self, v):
        return self.e @ v.e

    def cross(self, v):
        return Vec3(*np.cross(self.e, v.e))

    def unit_vector(self):
        return self / self.length()

    def near_zero(self):
        s = 1e-8
        return fabs(self.e[0]) < s and fabs(self.e[1]) < s and fabs(self.e[2]) < s

    def reflect(self, n):
        return self - 2 * self.dot(n) * n

    def refract(self, n, etai_over_etat):
        cos_theta = min(-self.dot(n), 1)
        r_out_perp = etai_over_etat * (self + cos_theta * n)
        r_out_parallel = -sqrt(fabs(1 - r_out_perp.length_squared())) * n
        return r_out_perp + r_out_parallel

    def to_list(self):
        return list(self.e)

    @classmethod
    def rand(self, a=0, b=1):
        return Vec3(uniform(a, b), uniform(a, b), uniform(a, b))

    @classmethod
    def random_in_unit_sphere(self):
        while True:
            p = Vec3().rand(-1, 1)
            if p.length_squared() >= 1:
                continue
            return p

    @classmethod
    def random_unit_vector(self):
        return self.random_in_unit_sphere().unit_vector()

    @classmethod
    def random_in_hemisphere(self, normal):
        in_unit_sphere = Vec3().random_in_unit_sphere()
        if in_unit_sphere.dot(normal) > 0:
            return in_unit_sphere
        else:
            return -in_unit_sphere

    @classmethod
    def random_in_unit_disk(self):
        while True:
            p = Vec3(uniform(-1, 1), uniform(-1, 1), 0)
            if p.length_squared() >= 1:
                continue
            return p


Point3 = Vec3
Color = Vec3
