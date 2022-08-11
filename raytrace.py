import open3d as o3d
from ray import Ray
from vec3 import Vec3, Color
import numpy as np
from line_profiler import LineProfiler


# single ray cast
def cast_ray(ray, scene, mesh, depth, sink_id, baseray=None, verts=None, lt2=None, hit_sink = []):
    if depth <= 0:
        verts.append(ray.at(30).to_list())
        return (verts, lt2, hit_sink)

    if verts == None:
        verts = []
    if lt2 == None: #List to draw normals for each hit
        lt2 = []
    if hit_sink == None:
        hit_sink = []

    rays = o3d.core.Tensor(
        [ray.origin().to_list() + ray.direction().to_list()],
        dtype=o3d.core.Dtype.Float32,
    )
    ans = scene.cast_rays(rays)
    t = ans["t_hit"].numpy()
    if t != np.inf:

        if ans["geometry_ids"].numpy()[0] == sink_id:
            hit_sink.append(baseray)
        norm = Vec3(ans["primitive_normals"].numpy()[0])
        if ray.direction().dot(norm) > 0:
            norm = -norm
        reflect = ray.direction().reflect(norm)
        reflected_ray = Ray(ray.at(t), reflect)
        reflected_ray.orig = reflected_ray.at(0.0001)
        lt2.append(ray.at(t).to_list())
        lt2.append((ray.at(t) + norm * 5).to_list())
        verts.append(ray.at(t).to_list())
        cast_ray(reflected_ray, scene, mesh, depth - 1, sink_id, baseray, verts, lt2, hit_sink)
    if t == np.inf:
        pass
        verts.append(ray.at(30).to_list())
    return (verts, lt2, hit_sink)


# multiple ray cast // TODO: need to make seperate vec3 for numpy cases
def cast_rays(rays, scene, max_dist, depth):
    pass


#Create lines based on vertexes
def create_lines(verts):
    lines = []
    if len(verts) == 2:
        return []
    for i in range(len(verts) - 1):
        lines.append([i, i + 1])
    return lines


#Create line set to visualize using the points and lines
def create_lineset(points, lines, c):
    if c == True:
        colors = [[1, 0, 0] for _ in range(len(lines))]

    else:
        colors = [[0, 0, 1] for _ in range(len(lines))]
        if len(colors) > 0:
            colors[-1] = [0, 1, 0] #The last ray drawn from an origin will be green

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_sink(scene, s_type, dim, pos):
    if isinstance(pos, Vec3):
        pos = list(pos.e)
    if s_type == "sphere":
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=dim).translate(pos)
        sphere_t = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
        sphere_id = scene.add_triangles(sphere_t)
        return sphere, sphere_id
    elif s_type == "box": #dim should be [x, y, z]
        box = o3d.geometry.TriangleMesh.create_box(dim[0], dim[1], dim[2]).translate(pos)
        box_t = o3d.t.geometry.TriangleMesh.from_legacy(box)
        box_id = scene.add_triangles(box_t)
        return box, box_id


def main():
    file_name = "pointclouds/dt_mesh.obj"
    mesh = o3d.io.read_triangle_mesh(file_name)

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    a = scene.add_triangles(mesh_t)
    box, box_id = create_sink(scene, "box", [40, 40, 40], Vec3(-210, 200, 50))

    t_list = [mesh, box]

    #generate rays and normals to map
    for i in range(100000): 
        dire = Vec3.random_in_unit_sphere() #direction of ray
        r = Ray(Vec3(-160, 10, 0), dire) #First Vec3 is origin of the ray
        points, nms, sink = cast_ray(r, scene, mesh_t, 5, box_id, baseray = r,verts=[r.origin().to_list()]) #points is where the ray hits and nms is the normals
        lines = create_lines(points)
        line_set = create_lineset(points, lines, False)
        t_list.append(line_set)
        li = [[i, i + 1] for i in range(0, len(points), 2)]
        fl = create_lineset(nms, li, True)
        t_list.append(fl)

    '''
    TO VISUALIZE ALL THE RAYS THAT HIT THE SINK//DELETE CODE LATER
    '''
    n_list = [mesh, box]
    for j in range(len(sink)):
        points, nms, sink = cast_ray(sink[j], scene, mesh_t, 5, box_id, baseray = sink[j],verts=[sink[j].origin().to_list()]) #points is where the ray hits and nms is the normals
        lines = create_lines(points)
        line_set = create_lineset(points, lines, False)
        n_list.append(line_set)
        li = [[i, i + 1] for i in range(0, len(points), 2)]
        fl = create_lineset(nms, li, True)
        n_list.append(fl)

    o3d.visualization.draw_geometries(n_list)

lp = LineProfiler()
lp_wrapper = lp(main)
lp.add_function(cast_ray)  # add additional function to profile
lp_wrapper()
lp.print_stats()
