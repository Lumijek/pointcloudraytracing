import open3d as o3d
from ray import Ray
from vec3 import Vec3, Color
import numpy as np
from scipy.spatial import KDTree, Voronoi
import scipy
from line_profiler import LineProfiler
from time import perf_counter
from splat import Splat, World
from pprint import pprint
import matplotlib.pyplot as plt
import sys
import pdb


def nprint(var):
    print(var)
    print("\n--------------------------------------------------------\n")


np.set_printoptions(threshold=3000)


def fitPLaneLTSQ(XYZ):
    [rows, cols] = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  # X
    G[:, 1] = XYZ[:, 1]  # Y
    Z = XYZ[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return normal


def signed_distance(pi, pj, ni):
    return ni.dot(pi - pj)


def generate_splat(pcd, pcd_tree, point, k, threshold, perc, points, activated):
    [k, idx, _] = pcd_tree.search_knn_vector_3d(point, k)
    found_points = points[idx]
    normal = fitPLaneLTSQ(found_points)
    h = (found_points[0] - found_points).dot(normal)
    f = np.abs(h) < threshold
    v = np.where(f == 0)[0]
    if v.size == 0:
        ind = k - 1
        trueh = h
    else:
        ind = v[0]
        trueh = h[:ind]

    center = found_points[0] + normal * ((np.amax(trueh) - np.amin(trueh)) / 2)
    radius = np.linalg.norm(
        (found_points[ind] - center) - normal.dot(found_points[ind] - center) * normal
    )

    [k, idx, _] = pcd_tree.search_radius_vector_3d(point, radius * perc)
    np.asarray(pcd.colors)[idx, :] = [0, 1, 1]  # Convert all relevant points to blue.
    s = np.size(activated[idx]) - np.count_nonzero(activated[idx])
    activated[idx] = True
    return Splat(center, normal, radius, points)


def create_splats(world, pcd, pcd_tree, k, threshold, perc, points):
    activated = np.full(len(points), False)
    available_indices = np.where(activated == 0)[0]
    i = 0
    c = 0
    while True:
        i += 1
        if i % 10000 == 0:
            break
            c += 1
            s = np.sum(activated)
            # print(s, i)
            if s > 100000:
                break
            if c == 200:
                break
        p = np.random.choice(available_indices)
        c_splat = generate_splat(
            pcd, pcd_tree, pcd.points[p], k, threshold, perc, points, activated
        )
        world.add_splat(c_splat)


def ray_splat_intersection(O, D, world):
    for splat in world.splats:
        center, normal, radius, radius_squared = (
            splat.center,
            splat.normal,
            splat.radius,
            splat.radius_squared,
        )
        denom = D.dot(normal)
        t = (center - O).dot(normal) / denom
        points = O + D * t[:, None]
        distances = center - points
        distances_squared = np.sum(np.square(distances), axis=1)
        distances_squared[distances_squared > radius_squared] = np.nan


def improved_ray_splat_intersection(O, D, world, depth):
    center = world.center
    normal = world.normal
    radius = world.radius
    radius_squared = world.radius_squared

    denoms = normal.dot(D.T)
    t = np.einsum("ijk, ik->ij", (center[:, None] - O), normal) / denoms
    t[t < 0] = np.inf
    points = O + np.einsum("ij, jl->ijl", t, D)
    distances = points - O
    distances = center[:, None] - points
    distances_squared = np.sum(np.square(distances), axis=2)
    distances_squared[distances_squared > radius_squared[:, None]] = np.inf
    min_args = np.argmin(distances_squared, axis=0)
    mins = np.amin(distances_squared, axis=0)
    hits = np.take_along_axis(t, min_args[:, None], axis=0).diagonal()
    true_points = O + D * hits[:, None]
    true_points[np.isinf(mins)] = np.inf
    ls = create_lineset(O, true_points, depth)
    return true_points, normal[min_args[min_args != 0]], hits[:, None], ls


def create_lineset(O, H, depth):
    points = []
    for i in range(O.shape[0]):
        if np.isinf(H[i]).all():
            continue
        points.append(O[i].tolist())
        points.append(H[i].tolist())

    lines = []
    for i in range(0, len(points) - 1, 2):
        lines.append([i, i + 1])

    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def cast_ray(world, O, D, depth, geometry):
    if depth == 0:
        return geometry
    new_O, normals, t, line_set = improved_ray_splat_intersection(O, D, world, depth)
    print(depth, line_set)
    geometry.append(line_set)
    ind = ~np.isinf(new_O).any(axis=1)
    O = new_O[ind]
    D = D[ind]
    t = t[ind]
    bad_normals = np.where(np.sum(D * normals, axis=1) > 0)[0]  # If ray and normal are on same side(dot product > 0) make the normal negative
    normals[bad_normals] = -normals[bad_normals]
    reflected = D - 2 * np.sum(D * normals, axis=1)[:, None] * normals
    O += reflected * 1
    cast_ray(world, O, reflected, depth - 1, geometry)
    return geometry


def main():
    np.random.seed(1)
    number_of_rays = 1
    O = np.zeros(number_of_rays * 3).reshape(number_of_rays, 3)
    D = np.random.rand(number_of_rays * 3).reshape(number_of_rays, 3)
    D[0] = np.array([[0.03905478, 0.16983042, 0.8781425]])
    D = D / np.linalg.norm(D, axis=1, keepdims=True)  # normalize directions
    threshold = 1
    perc = 0.8
    file_name = "pointclouds/san.ply"
    pcd = o3d.io.read_point_cloud(file_name)
    points = np.asarray(pcd.points)
    points += np.random.normal(0, 0.000001, points.shape)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    world = World()
    create_splats(world, pcd, pcd_tree, 100, threshold, perc, points)
    world.construct_world_splat()
    # ray_splat_intersection(O, D, world)
    c = []
    ls = cast_ray(world, O, D, 1, c)
    ls.append(pcd)
    o3d.visualization.draw_geometries(ls)



lp = LineProfiler()
lp_wrapper = lp(main)
lp.add_function(cast_ray)  # add additional function to profile
lp.add_function(improved_ray_splat_intersection)
lp_wrapper()
lp.print_stats()
