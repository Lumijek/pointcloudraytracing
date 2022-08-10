import open3d as o3d
from ray import Ray
from vec3 import Vec3, Color
import numpy as np
from line_profiler import LineProfiler


# single ray cast
def cast_ray(ray, scene, mesh, max_dist, depth, verts=None, lt2=None):
    if depth <= 0:
        verts.append(ray.at(30).to_list())
        return (verts, lt2)

    if verts == None:
        verts = []
    if lt2 == None: # 
        lt2 = []
    rays = o3d.core.Tensor(
        [ray.origin().to_list() + ray.direction().to_list()],
        dtype=o3d.core.Dtype.Float32,
    )
    ans = scene.cast_rays(rays)
    t = ans["t_hit"].numpy()
    if t != np.inf:
        norm = Vec3(ans["primitive_normals"].numpy()[0])
        if ray.direction().dot(norm) > 0:
            norm = -norm
        reflect = ray.direction().reflect(norm)
        reflected_ray = Ray(ray.at(t), reflect)
        reflected_ray.orig = reflected_ray.at(0.0001)
        lt2.append(ray.at(t).to_list())
        lt2.append((ray.at(t) + norm * 5).to_list())
        verts.append(ray.at(t).to_list())
        cast_ray(reflected_ray, scene, mesh, max_dist, depth - 1, verts, lt2)
    if t == np.inf:
        pass
        verts.append(ray.at(30).to_list())
    return (verts, lt2)


# multiple ray cast // TODO: need to make seperate vec3 for numpy cases
def cast_rays(rays, scene, max_dist, depth):
    pass


def create_lines(verts):
    lines = []
    if len(verts) == 2:
        return []
    for i in range(len(verts) - 1):
        lines.append([i, i + 1])
    return lines


def create_lineset(points, lines, c):
    if c == True:
        colors = [[1, 0, 0] for _ in range(len(lines))]

    else:
        colors = [[0, 0, 1] for _ in range(len(lines))]
        if len(colors) > 0:
            colors[-1] = [0, 1, 0]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def main():
    file_name = "pointclouds/dt_mesh.obj"
    mesh = o3d.io.read_triangle_mesh(file_name)

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    t_list = [mesh]
    for i in range(200):
        dire = Vec3.random_in_unit_sphere()
        r = Ray(Vec3(-160, 10, 0), dire)
        points, nms = cast_ray(r, scene, mesh_t, 1, 5, verts=[r.origin().to_list()])
        lines = create_lines(points)
        line_set = create_lineset(points, lines, False)
        t_list.append(line_set)
        li = [[i, i + 1] for i in range(0, len(points), 2)]
        fl = create_lineset(nms, li, True)
        t_list.append(fl)

    o3d.visualization.draw_geometries(t_list)


lp = LineProfiler()
lp_wrapper = lp(main)
lp.add_function(cast_ray)  # add additional function to profile
lp_wrapper()
lp.print_stats()
