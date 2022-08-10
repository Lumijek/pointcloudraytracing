import numpy as np
from vec3 import Vec3
from triangle import Triangle


class World:
    def __init__(self, mesh):
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)

    def __len__(self):
        return len(self.triangles)

    def get_triangle(self, n):
        return Triangle(
            Vec3(self.vertices[self.triangles[n][0]]),
            Vec3(self.vertices[self.triangles[n][1]]),
            Vec3(self.vertices[self.triangles[n][2]]),
        )

    def hit(self, ray):
        closest_so_far = None
        closest_dist = np.inf
        for i in range(len(self.triangles)):
            triangle = self.get_triangle(i)
            a, b = triangle.collide_with_ray(ray)
            if a:
                if ray.at(b) < closest_dist:
                    closest_so_far = i
        return closest_so_far
