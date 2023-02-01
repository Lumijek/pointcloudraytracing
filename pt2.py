import open3d as o3d
import numpy as np
from line_profiler import LineProfiler
from time import perf_counter
from splat import *
import sys
from collections import defaultdict
import scipy
import math
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import sqrt, sin, cos, pi

M_FACTOR = 0.1
PERC = 0.3
THRESHOLD = 0.5
NUMBER_OF_POINTS = 250_000
NUMBER_OF_RAYS = 2000
DEPTH = 2
SPLAT_SIZE = 100


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


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
    #np.asarray(pcd.colors)[idx, :] = [0, 1, 1]  # Convert all relevant points to blue.
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
            c += 1
            s = np.sum(activated)
            print("Points covered:", s, ", Splats:", i)
            if s > NUMBER_OF_POINTS:
                break

        p = np.random.choice(available_indices)
        c_splat = generate_splat(
            pcd, pcd_tree, pcd.points[p], k, threshold, perc, points, activated
        )
        world.add_splat(c_splat)

def test_sink_hit(O, D, scene, geometry, returns):
    rays = o3d.core.Tensor(np.hstack((O, D)),
                           dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)
    hits = ans['t_hit'].numpy()
    sink_indexes = np.where(hits != np.inf)[0]
    non_sink_indexes = np.where(hits == np.inf)[0]
    O_new = O[sink_indexes]
    D_new = D[sink_indexes]
    t = hits[sink_indexes]
    if(t.size > 0):
        H = O_new + D_new * t[:, None]
        l = create_lineset(O_new, H, 100)
        geometry.append(l)
        returns[0] += t.size
        return O[non_sink_indexes], D[non_sink_indexes]
    else:
        return O, D

def improved_ray_splat_intersection(O, D, world, depth, scene, geometry, returns):

    center = world.center
    normal = world.normal
    radius = world.radius
    radius_squared = world.radius_squared
    if(depth != DEPTH):
        O, D = test_sink_hit(O, D, scene, geometry, returns) # hit line set 
    denoms = normal.dot(D.T)
    t = np.einsum("ijk, ik->ij", (center[:, None] - O), normal) / denoms
    t[t < 0] = np.inf
    points = O + np.einsum("ij, jl->ijl", t, D)
    distances = center[:, None] - points
    distances_squared = np.sum(np.square(distances), axis=2)
    distances_squared[distances_squared > radius_squared[:, None]] = np.inf

    k = np.where(distances_squared != np.inf)
    tk = t[k]
    n = k[1].tolist()
    d = defaultdict(lambda: len(d))
    ids = np.array([d[x] for x in n])

    try:
        l = np.max(ids) + 1
    except:
        return 1, 1, 1, 1, 1, 1

    origin_distances = points[k] - O[k[1]]
    k = np.column_stack((k[1], k[0], ids))
    true_points = np.full((l, 3), np.inf)
    normals = np.zeros((l, 3))
    hits = np.zeros(l)
    inds = np.zeros(l, dtype=int)

    for i in range(len(k)):
        r, s, cid = k[i]
        current_distance = true_points[cid] - O[r]
        current_distance_squared = np.dot(current_distance, current_distance)
        new_distance = origin_distances[i]
        if np.dot(new_distance, new_distance) < current_distance_squared:
            true_points[cid] = origin_distances[i] + O[r]
            normals[cid] = normal[s]
            hits[cid] = tk[i]
            inds[cid] = r
    ls = create_lineset(O[inds], true_points, depth)
    ns = create_lineset(center, center + normal, depth)
    return true_points, D[inds], normals, hits, ls, ns


def create_lineset(O, H, depth):
    points = []
    for i in range(O.shape[0]):
        points.append(O[i].tolist())
        points.append(H[i].tolist())

    lines = []
    for i in range(0, len(points) - 1, 2):
        lines.append([i, i + 1])

    if depth == 100:
        colors = [[0, 0, 0] for _ in range(len(lines))]
    else:
        colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def cast_ray(world, O, D, depth, geometry, scene, returns):
    if depth == 0:
        return geometry
    O, D, normals, t, line_set, normal_set = improved_ray_splat_intersection(O, D, world, depth, scene, geometry, returns)
    if type(O) == int:
        return geometry
    geometry.append(line_set)
    #geometry.append(normal_set)
    bad_normals = np.where(np.sum(D * normals, axis=1) > 0)[0]  # If ray and normal are on same side(dot product > 0) make the normal negative
    normals[bad_normals] = -normals[bad_normals]
    reflected = D - 2 * np.sum(D * normals, axis=1)[:, None] * normals
    O += reflected * M_FACTOR
    last_hit = create_lineset(O, O + reflected * 0.5, 100)
    cast_ray(world, O, reflected, depth - 1, geometry, scene, returns)
    return last_hit

def add_sink(world, geometries, radius=1.0, center=[-12, 2, 10]):
    world.sink = Sphere(center, radius)
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.4, 0.8, 0.3])

    sphere = o3d.t.geometry.TriangleMesh.from_legacy(mesh_sphere)
    scene = o3d.t.geometry.RaycastingScene()
    sphere_id = scene.add_triangles(sphere)
    geometries.append(mesh_sphere)
    return scene

def batch(other_points, ray_num, geometries):
    pass



def sunflower(x, n, r_fac, alpha=0, geodesic=False):
    phi = (1 + sqrt(5)) / 2  # golden ratio

    points = []
    angle_stride = 360 * phi if geodesic else 2 * pi / phi ** 2
    b = round(alpha * sqrt(n))  # number of boundary points
    for k in range(1, n + 1):
        r = radius(k, n, b)
        theta = k * angle_stride
        points.append([x, r_fac * r * cos(theta), r_fac * r * sin(theta)])
    return points

def radius(k, n, b):
    if k > n - b:
        return 1.0
    else:
        return sqrt(k - 0.5) / sqrt(n - (b + 1) / 2)

def main():
    center = [-1, 0, 0]
    axis = [0, 0, 1]
    theta = math.radians(90)

    pcd2 = o3d.geometry.PointCloud()
    file_name = "pointclouds/car_cart4.mat"
    other_points = scipy.io.loadmat(file_name)['car_cart']
    other_points += np.random.normal(0, 0.000001, other_points.shape)
    other_points = np.dot(rotation_matrix(axis, theta), other_points.T).T

    end_points = other_points[np.random.choice(len(other_points), size=NUMBER_OF_RAYS, replace=False)]
    cent = np.sum(other_points, axis=0) / other_points.shape[0]
    pcd2.points = o3d.utility.Vector3dVector(other_points)


    O = np.zeros(NUMBER_OF_RAYS * 3).reshape(NUMBER_OF_RAYS, 3)
    O.T[0] = 5.0
    O.T[1] = 0.0
    O.T[2] = 0.5
    D = np.array(sunflower(cent[0], NUMBER_OF_RAYS, 2,alpha=0, geodesic=False)) - O
    D = D / np.linalg.norm(D, axis=1, keepdims=True)  # normalize directions

    #D = np.array([0.00160762, -0.35566735, 0.58488357] - O[0])
    #D = np.tile(D, (NUMBER_OF_RAYS, 1))
    #D = D / np.linalg.norm(D, axis=1, keepdims=True)  # normalize directions

    '''
    D = -np.random.rand(NUMBER_OF_RAYS * 3).reshape(NUMBER_OF_RAYS, 3)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)  # normalize directions

    O = np.zeros(3).reshape(1, 3)
    O.T[1] = 5
    O.T[2] = 10
    D = np.array([[-0.82650114, -0.3520768, -0.43924685]])
    '''

    '''
    file_name = "pointclouds/atk_back.pcd"
    pcd = o3d.io.read_point_cloud(file_name)
    points = np.asarray(pcd.points)
    points += np.random.normal(0, 0.000001, points.shape)
    points = np.vstack((points, other_points))
    pcd.points = o3d.utility.Vector3dVector(points)

    '''
    geometries = []

    pcd_tree = o3d.geometry.KDTreeFlann(pcd2)

    world = World()

    scene = add_sink(world, geometries, center=O[1])
    create_splats(world, pcd2, pcd_tree, 100, THRESHOLD, PERC, other_points)
    world.construct_world_splat()

    batches = 4
    bs = NUMBER_OF_RAYS // batches
    returns = [0]
    for i in range(batches):
        lh = cast_ray(world, O[(i * bs): ((i + 1) * bs)], D[(i * bs): ((i + 1) * bs)], DEPTH, geometries, scene, returns)
    print(geometries)

    #geometries.append(pcd)
    geometries.append(pcd2)
    #geometries.append(lh)
    o3d.visualization.draw_geometries(geometries)
    #PRINT HYPERPARAMETERS AND STATS:
    print("\n\nHyperparameters: ")
    print("_____________________________")
    print("Perc:", PERC)
    print("Threshold:", THRESHOLD)
    print("Number of points:", NUMBER_OF_POINTS)
    print("Number of rays:", NUMBER_OF_RAYS)
    print("Depth:", DEPTH)
    print("Splat Size", SPLAT_SIZE)
    print("\nOther stats:")
    print("Total splats generated: ", len(world.splats))


main()