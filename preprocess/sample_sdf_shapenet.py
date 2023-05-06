import numpy as np
import trimesh
import os
import shutil
import argparse
import json
import open3d as o3d
import pyrender
import matplotlib.pyplot as plt
import cv2
from multiprocessing import Pool
from pysdf import SDF
from skimage.measure import marching_cubes


class sampler:
    def __init__(self, class_dir, args):
        self.class_dir = class_dir
        self.instance_list = os.listdir(class_dir)
        self.instance_list.sort()
        self.start = args.start
        self.end = args.end
        self.size = 1
        self.output_dir = args.write_dir
        assert self.end - self.start >= self.size
    
    def run_sampler(self):
        #counter = 0
        for i in self.instance_list[self.start:self.end]:
            try:
                self.sample_data(i, False)
            except:
                path = os.path.join(self.output_dir, i)
                if not os.path.exists(path):
                    shutil.rmtree(path)

    def normalize_vector(self, vector):
        return vector / (np.linalg.norm(vector))

    # https://learnopengl.com/Getting-started/Camera
    def get_pose(self, position_vector, front_vector, up_vector):
        m1 = np.eye(4)
        m2 = np.eye(4)
        
        # look at
        z = self.normalize_vector(front_vector)
        # up 
        y = self.normalize_vector(up_vector)
        # right
        x = self.normalize_vector(np.cross(y, z))
        # up
        y = self.normalize_vector(np.cross(z, x))
       
        m1[0, :3] = x
        m1[1, :3] = y
        m1[2, :3] = z
        m1[3, 3] = 1.0

        m2[0, 0] = m2[1, 1] = m2[2, 2] = 1.0
        m2[:3, 3] = -position_vector
        m2[3, 3] = 1.0

        return np.linalg.inv(np.matmul(m1, m2))

    def render_image(self, py_mesh, instance):
        
        points = np.array([[1, 1, 1], 
                          [1, 1, -1], 
                          [1, -1, 1], 
                          [-1, 1, 1], 
                          [-1, -1, 1], 
                          [-1, 1, -1], 
                          [1, -1, -1], 
                          [-1, -1, -1]])
        
        py_mesh = pyrender.Mesh.from_trimesh(py_mesh, smooth=False)
        
        for i, pos in enumerate(points):
            scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[1., 1., 1.])
            camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.)
            light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.)

            scene.add(py_mesh, pose =  np.eye(4))
            scene.add(light, pose = np.eye(4))
            scene.add(light2, pose = np.array([[1., 0., 0., 0], 
                                               [0., 1., 0., 3], 
                                               [0., 0., 1., 0], 
                                               [0., 0., 0., 1.]]))
            scene.add(light2, pose = np.array([[1., 0., 0., 0], 
                                               [0., 1., 0., -3], 
                                               [0., 0., 1., 0], 
                                               [0., 0., 0., 1.]]))
            mat = self.get_pose(pos, pos, np.array([0., 1., 0.]))
            scene.add(camera, pose=mat)
            # render scene
            flags = pyrender.constants.RenderFlags.SKIP_CULL_FACES
            r = pyrender.OffscreenRenderer(128, 128)
            color, _ = r.render(scene, flags=flags)
            path = os.path.join(self.output_dir, instance)

            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, f"{i:02d}.png"), color)   

    def sample_data(self, instance, gui=False):
        obj_path = os.path.join(self.class_dir, instance)
        mesh_vertices, mesh_faces, obj_mesh = self.load_object(obj_path)
        if not obj_mesh.is_watertight:
            return False
        if gui:
            obj_mesh.show()
        # render image
        self.render_image(obj_mesh, instance)
        # sample points sdf
        xyz = self.sample_points(10000)
        if gui:
            pts = trimesh.points.PointCloud(xyz)
            pts.show()
        sdf_est = SDF(mesh_vertices, mesh_faces)
        sdf = sdf_est(xyz)
        query = np.concatenate((xyz, np.array([sdf]).reshape(-1, 1)), -1)
        path = os.path.join(self.output_dir, instance) 
        if not os.path.exists(path):
            os.makedirs(path)    
        np.save(os.path.join(path, 'xyzsdf.npy'), query) 
        #print(sdf)
        #self.marching_cube(mesh_vertices, mesh_faces)

    def marching_cube(self, mesh_vertices, mesh_faces, res=256):
        # sample xyz from grid
        lin_coords = np.linspace(-1.0, 1.0, res)
        x_coords, y_coords, z_coords = np.meshgrid(lin_coords, lin_coords, lin_coords)
        xyz = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(-1, 3)
        
        # get sdf
        sdf_est = SDF(mesh_vertices, mesh_faces)
        sdf = sdf_est(xyz) 

        # make implicit value
        threshold = 0
        implicit_values = sdf.reshape(res, res, res)
        implicit_values = np.where(implicit_values>0, 1, -1)
       
        # run marching cube
        spacing = 2.*1.0/(res-1)
        verts, faces, _, _ = marching_cubes(implicit_values, level=0.0, spacing=(spacing, spacing, spacing))
        #verts = verts - 1.0
        mesh = trimesh.Trimesh(vertices=verts, faces=faces) 
        mesh.show(flags={'cull': False})
        #mesh = o3d.geometry.TriangleMesh()
        #mesh.vertices = o3d.utility.Vector3dVector(verts)
        #mesh.triangles = o3d.utility.Vector3iVector(faces)
        #mesh.compute_vertex_normals()
        #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # https://stackoverflow.com/questions/5408276/
    # sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    def sample_points(self, points):
        phi = np.random.rand(points, 1)*np.pi*2
        costheta = np.random.rand(points, 1)*2-1
        u = np.random.rand(points, 1)
        theta = np.arccos(costheta)
        r = 1.0*np.cbrt(u)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        xyz = np.squeeze(np.stack((x, y, z), axis=1))
        return xyz
    
    # from NeurlODFscene_or_mesh
    def as_mesh(self, scene_or_mesh):
        """
        Convert a possible scene to a mesh.
        If conversion occurs, the returned mesh has only vertex and face data.
        """
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                mesh = None  # empty scene
            else:
                # we lose texture information here
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in scene_or_mesh.geometry.values()))
        else:
            assert(isinstance(scene_or_mesh, trimesh.Trimesh))
            mesh = scene_or_mesh
        return mesh


    def load_object(self, object_path):
        obj_file = os.path.join(object_path, "models", "model_normalized.obj")
        obj_mesh = trimesh.load_mesh(obj_file)
        obj_mesh = self.as_mesh(obj_mesh)  
        ## deepsdf normalization
        mesh_vertices = obj_mesh.vertices
        mesh_faces = obj_mesh.faces
        center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
        max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
        max_dist = max_dist * 1.03
        mesh_vertices = (mesh_vertices - center) / max_dist
        obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    
        return mesh_vertices, mesh_faces, obj_mesh

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet-dir", default="/gpfs/data/ssrinath/projects/stnerf/DGP/ShapeNetCore.v2/", help="The path to the ShapeNet V2 object directory")
    # 03001627:chair
    parser.add_argument("--subdir", default="03001627", type=str, help="The object id")
    parser.add_argument("--cpu", default=4, type=int, help="cpu pool size")
    parser.add_argument("--write-dir", default="/gpfs/data/ssrinath/projects/stnerf/DGP/ShapeNetDGP/")
    parser.add_argument("--start", default=0, type=int, help="start from file i")
    parser.add_argument("--end", default=10000, type=int, help="end at file i")
    args = parser.parse_args()

    # Find out which object we are dumping data for
    taxonomy_file = open(os.path.join(args.shapenet_dir, "taxonomy.json"), "r")
    taxonomy_str = taxonomy_file.read()
    taxonomy = json.loads(taxonomy_str)
    #print(taxonomy)

    for d in taxonomy:
        if d["synsetId"] == args.subdir:
            class_name = d["name"].split(",")[0]
    if class_name == None:
        print(f"Unable to find directory '{args.subdir}' in ShapeNetCore.v2")
    else:
        class_dir = os.path.join(args.shapenet_dir, args.subdir)
    
    #print(class_name)
    #print(class_dir)
    sampler = sampler(class_dir, args)    
    sampler.run_sampler()
