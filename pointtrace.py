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
number_of_splats = [0]
number_of_points = [0]
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
    np.asarray(pcd.colors)[idx, :] = [0, 1, 1] # Convert all relevant points to blue.
    s = (np.size(activated[idx]) - np.count_nonzero(activated[idx]))
    number_of_points.append(number_of_points[-1] + s)
    activated[idx] = True
    return Splat(center, normal, radius, points)

#Plot
#X-axis - number of splats
#Y-axis - number of points covered
#@profile
def create_splats(world, pcd, pcd_tree, k, threshold, perc, points):
    activated = np.full(len(points), False)
    available_indices = np.where(activated == 0)[0]
    i = 0
    c = 0
    while True:#sum(available_indices) != len(points):
        i += 1
        if i % 10000 == 0:
            c += 1
            s = np.sum(activated)
            print(s, i)
            if s > 9600000:
                break
            if c == 200:
                break
        p = np.random.choice(available_indices)
        c_splat = generate_splat(pcd, pcd_tree, pcd.points[p], k, threshold, perc, points, activated)
        number_of_splats.append(i + 1)
        world.add_splat(c_splat)


def main(threshold, perc):
    global number_of_points
    global number_of_splats
    file_name = "pointclouds/san.ply"
    pcd = o3d.io.read_point_cloud(file_name)
    points = np.asarray(pcd.points)
    points += np.random.normal(0, 0.000001, points.shape)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    world = World()
    create_splats(world, pcd, pcd_tree, 100, threshold, perc, points)
    #m = o3d.geometry.TriangleMesh.create_cylinder(radius = 50, height = 1, resolution = 100)
    #plt.plot(number_of_splats, number_of_points)
    #plt.title(f"Splats vs Points (perc: {perc})")
    #plt.xlabel("Number of splats")
    #plt.ylabel("Number of points")
    #plt.savefig(f'perc{str(perc).replace(".", "-")}.png', bbox_inches='tight')
    #plt.clf()
    #number_of_splats = [0]
    #number_of_points = [0]
    o3d.visualization.draw_geometries([pcd])

main(0.1, 0.8)