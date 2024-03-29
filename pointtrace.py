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
from math import sqrt, sin, cos, pi
import multiprocessing as mp
import resource
import atexit
from pprint import pprint
import signal


M_FACTOR = 0.05
PERC = 0.3
THRESHOLD = 0.2
NUMBER_OF_POINTS = 200_000
NUMBER_OF_RAYS = 100
DEPTH = 4
SPLAT_SIZE = 100
SINK_CENTER = [5.0001, 0.0001, 0.5001]
SINK_RADIUS = 1
BATCHES = 8
BATCH_SIZE = NUMBER_OF_RAYS // BATCHES

def exit_handler():
    f.close()

def cprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.exit(0)

class LineSet:
    def __init__(self, start, end, depth):
        self.start = start
        self.end = end
        self.depth = depth

# taken from stackoverflow
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
    activated[idx] = True
    return Splat(center, normal, radius)

def create_splats(world, pcd, pcd_tree, k, threshold, perc, points):
    activated = np.full(len(points), False)
    available_indices = np.where(activated == 0)[0]
    i = 0
    c = 0
    while True:
        i += 1
        if i % 1000 == 0:
            c += 1
            s = np.sum(activated)
            print("Points covered:", s, ", Splats:", i, flush=True)
            
            if s > NUMBER_OF_POINTS:
                break

        p = np.random.choice(available_indices)
        c_splat = generate_splat(
            pcd, pcd_tree, pcd.points[p], k, threshold, perc, points, activated
        )
        world.add_splat(c_splat)

def sphere_intersect(sphere_center, sphere_radius, origin, direction):
    oc = origin - sphere_center
    a = np.sum(direction * direction, axis=1)
    b = 2 * np.sum(oc * direction, axis=1)
    c = np.sum(oc * oc, axis=1) - sphere_radius * sphere_radius
    discriminant = b * b - 4 * a * c
    dist = np.where(discriminant >= 0, (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a), np.inf)
    return dist

def test_sink_hit(O, D, geometry, returns, depth, paths, sink_paths):
    hits = sphere_intersect(SINK_CENTER, SINK_RADIUS, O, D)
    sink_indexes = np.where(hits != np.inf)[0]
    non_sink_indexes = np.where(hits == np.inf)[0]
    good_paths = [paths[i] for i in sink_indexes]
    #print(paths)
    #good_paths = []
    O_new = O[sink_indexes]
    D_new = D[sink_indexes]
    t = hits[sink_indexes]
    if(t.size > 0):
        H = O_new + D_new * t[:, None]
        for i in range(len(good_paths)):
            p = good_paths[i]
            p[-2].append(t[i])
            p.insert(-1, [H[i]])
            p = p[:-1]
            sink_paths.append(p)
            sys.stdout.flush()
        l = create_lineset(O_new, H, 100)
        geometry.append(l)
        returns[0] += t.size
        return O[non_sink_indexes], D[non_sink_indexes]
    else:
        return O, D

def improved_ray_splat_intersection(O, D, world, depth, geometry, returns, paths, sink_paths):
    center = world.center
    normal = world.normal
    radius = world.radius
    radius_squared = world.radius_squared
    if(depth != DEPTH):
        O, D = test_sink_hit(O, D, geometry, returns, depth, paths.copy(), sink_paths) # hit line set 
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
        return 1, 1, 1, 1, 1, 1, 1
 
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
    ns = None#ns = create_lineset(center, center + normal, depth)
    return true_points, D[inds], normals, hits, ls, ns, inds


def create_lineset(O, H, depth, de=False):
    if de == False:
        return LineSet(O, H, depth)
    points = []
    for i in range(O.shape[0]):
        if (np.sum(np.square(O[i] - H[i])) > 100000): continue
        points.append(O[i].tolist())
        points.append(H[i].tolist())

    lines = []
    for i in range(0, len(points) - 1, 2):
        lines.append([i, i + 1])

    if depth == 100:
        colors = [[0, 0, 0] for _ in range(len(lines))]
    elif depth == 1000:
        colors = [[0 , 0.75, 1] for _ in range(len(lines))]
    else:
        colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def get_hit_paths(hits):
    O = []
    H = []
    for hit in hits:
        if len(hit) != 4:
            continue
        for i in range(len(hit) - 1):
            O.append(hit[i][0])
            H.append(hit[i + 1][0])
    return np.array(O), np.array(H)

def cast_ray(world, O, D, depth, geometry, returns, paths, sink_paths):
    if depth == 0:
        return geometry
    O, D, normals, t, line_set, normal_set, inds = improved_ray_splat_intersection(O, D, world, depth, geometry, returns, paths, sink_paths)
    if type(O) == int:
        return geometry
    geometry.append(line_set)
    #geometry.append(normal_set)
    bad_normals = np.where(np.sum(D * normals, axis=1) > 0)[0]  # If ray and normal are on same side(dot product > 0) make the normal negative
    normals[bad_normals] = -normals[bad_normals]
    reflected = D - 2 * np.sum(D * normals, axis=1)[:, None] * normals
    O += reflected * M_FACTOR
    last_hit = create_lineset(O, O + reflected * 0.5, 99)

    paths = [paths[i] for i in inds]
    for i, p in enumerate(paths):
        p[-2].append(t[i])
        p.insert(-1, [O[i], reflected[i]])

    cast_ray(world, O, reflected, depth - 1, geometry, returns, paths, sink_paths)
    return last_hit

def add_sink(world, geometries, radius=SINK_RADIUS, center=SINK_CENTER):
    world.sink = Sphere(center, radius)
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.4, 0.8, 0.3])

    geometries.append(mesh_sphere)

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

def add_floor(world):
    center = np.array([0, 0, -1])
    normal = np.array([0, 0, 1])
    radius = 1000
    floor = Splat(center, normal, radius, None)
    world.add_splat(floor)

def load_pcd(file_name, theta=0, axis=[0, 0, 1]):
    pcd = o3d.io.read_point_cloud(file_name)
    points = np.asarray(pcd.points)
    points = mat_file[list(mat_file.keys())[-1]]
    points += np.random.normal(0, 0.000001, points.shape)
    points = np.dot(rotation_matrix(axis, theta), points.T).T
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, points

def load_mat(file_name, theta=0, axis=[0, 0, 1]):
    pcd = o3d.geometry.PointCloud()
    mat_file = scipy.io.loadmat(file_name)
    points = mat_file[list(mat_file.keys())[-1]]
    points += np.random.normal(0, 0.000001, points.shape)
    points = np.dot(rotation_matrix(axis, theta), points.T).T
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, points

def load_pointcloud(file_name, theta=0, axis=[0, 0, 1]):
    if file_name.split(".")[-1] == "pcd":
        pcd, points = load_pcd(file_name, theta, axis)
    else:
        pcd, points = load_mat(file_name, theta, axis)

    return pcd, points

if __name__ == "__main__":
    global f
    f = open("ray_batch.txt", "a")
    atexit.register(exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGINT, exit_handler)
    center = [-1, 1, 1]
    axis = [0, 0, 1]
    theta = math.radians(90)

    file_name = "pointclouds/car_cart4.mat"
    pcd, other_points = load_pointcloud(file_name, theta)
    cent = np.sum(other_points, axis=0) / other_points.shape[0]
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    g1 = []

    for rays in range(4000, 50000, 1000):
        print(rays)
        f.write(f"Rays: {rays}\n\n")
        NUMBER_OF_RAYS = rays
        O = np.zeros(NUMBER_OF_RAYS * 3).reshape(NUMBER_OF_RAYS, 3)
        O.T[0] = SINK_CENTER[0]
        O.T[1] = SINK_CENTER[1]
        O.T[2] = SINK_CENTER[2]
        D = np.array(sunflower(cent[0], NUMBER_OF_RAYS, 2 ,alpha=0, geodesic=False)) - O
        #D = -np.random.rand(*O.shape) 
        D = D / np.linalg.norm(D, axis=1, keepdims=True)  # normalize directions

        for batches in range(10, 100):
            print(batches)
            BATCHES = batches
            BATCH_SIZE = NUMBER_OF_RAYS // BATCHES
            f.write(f"Batch Size: {BATCH_SIZE} ")

            g1 = []
            world = World()
            add_sink(world, g1, center=SINK_CENTER, radius=SINK_RADIUS)
            create_splats(world, pcd, pcd_tree, SPLAT_SIZE, THRESHOLD, PERC, other_points)
            world.construct_world_splat()
            f.write(f"Splats: {len(world.splats)} ")

            start = time.perf_counter()
            manager = mp.Manager()
            geometries = manager.list()
            returns = manager.list()
            returns.append(0)
            params = []
            sink_paths = manager.list()
            for i in range(BATCHES):
                paths = [["running"] for _ in range(BATCH_SIZE)]
                for ind in range(len(paths)):
                    paths[ind].insert(-1, [O[i * BATCH_SIZE + ind], D[i * BATCH_SIZE + ind]])

                params.append((world, O[(i * BATCH_SIZE): ((i + 1) * BATCH_SIZE)], D[(i * BATCH_SIZE): ((i + 1) * BATCH_SIZE)], DEPTH, geometries, returns, paths, sink_paths))

            with mp.Pool() as pool:
                ret = pool.starmap(cast_ray, params)

            end = time.perf_counter()
            time_taken = round(end-start, 4)
            f.write(f"Time: {time_taken}\n")
    f.close()
    geometries_true = []
    for ls in geometries:
        pass#geometries_true.append(create_lineset(ls.start, ls.end, ls.depth, de=True))
    geometries_true.append(pcd)
    geometries_true.append(g1[0])
    print("Number of hits:", returns[0])

    sink_paths = list(sink_paths)
    print(len(sink_paths))
    O, H = get_hit_paths(sink_paths)
    geometries_true.append(create_lineset(O, H, 1000, de=True))
    '''
    for s in world.splats:
        disk = o3d.geometry.TriangleMesh.create_cylinder(radius = s.radius, height = 0.001, resolution=3, split=1)
        disk.compute_vertex_normals()
        n1 = np.asarray(disk.triangle_normals[0])
        v = np.cross(n1, s.normal)

        R = disk.get_rotation_matrix_from_axis_angle(v)
        disk.rotate(R).translate(s.center)
        geometries_true.append(disk)
    '''
    o3d.visualization.draw_geometries(geometries_true)

