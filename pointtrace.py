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

@profile
def fitPLaneLTSQ(XYZ):
    [rows,cols] = XYZ.shape
    G = np.ones((rows,3))
    G[:,0] = XYZ[:,0]  #X
    G[:,1] = XYZ[:,1]  #Y
    Z = XYZ[:,2]
    (a,b,c),resid,rank,s = np.linalg.lstsq(G,Z, rcond=None) 
    normal = (a,b,-1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return normal

def signed_distance(pi, pj, ni):
    return ni.dot(pi - pj)

@profile
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
    radius = np.linalg.norm((found_points[ind] - center) - normal.dot(found_points[ind] - center) * normal)

    [k, idx, _] = pcd_tree.search_radius_vector_3d(point, radius * perc)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 1] # Convert all relevant points to blue.
    activated[idx] = True
    return Splat(center, normal, radius, points)

#@profile
def create_splats(world, pcd, pcd_tree, k, threshold, perc, points):
    activated = np.full(len(points), False)
    available_indices = np.where(activated == 0)[0]
    i = 0
    while True:#sum(available_indices) != len(points):
        i += 1
        if i % 10000 == 0:
            print(np.sum(activated), i)
        p = np.random.choice(available_indices)
        #print(p)
        c_splat = generate_splat(pcd, pcd_tree, pcd.points[p], k, threshold, perc, points, activated)
        world.add_splat(c_splat)


    
@profile
def main():
    file_name = "pointclouds/san.ply"
    p = 377292
    pcd = o3d.io.read_point_cloud(file_name)
    points = np.asarray(pcd.points)
    points += np.random.normal(0, 0.000001, points.shape)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    #center, normal = generate_splat(pcd, pcd_tree, pcd.points[p], 500, 1, 0.4, points, activated)
    world = World()
    create_splats(world, pcd, pcd_tree, 200, 1, 0.8, points)
    endl = (pcd.points[p] + normal * 500)
    ori = pcd.points[p].tolist()
    pons = [center, endl.tolist()]
    lins = [[0, 1]]
    cols = [[1, 0, 0]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pons)
    line_set.lines = o3d.utility.Vector2iVector(lins)
    line_set.colors = o3d.utility.Vector3dVector(cols)
    o3d.visualization.draw_geometries([pcd, line_set])


main()