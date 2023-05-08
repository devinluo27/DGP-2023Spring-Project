import numpy as np
import trimesh
import os
import pyrender
import cv2
from pysdf import SDF
from skimage.measure import marching_cubes


def load_object(object_path, instance):
    obj_file = os.path.join(object_path, instance, "models", "model_normalized.obj")
    obj_mesh = trimesh.load_mesh(obj_file)
    obj_mesh = as_mesh(obj_mesh)
    ## deepsdf normalization
    mesh_vertices = obj_mesh.vertices
    mesh_faces = obj_mesh.faces
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
    max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
    max_dist = max_dist * 1.03
    mesh_vertices = (mesh_vertices - center) / max_dist
    obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    
    return mesh_vertices, mesh_faces, obj_mesh

def render_image(py_mesh, instance):
        
    points = np.array([[1.2, 1.2, 1.2], 
                      [1.2, 1.2, -1.2], 
                      [1.2, -1.2, 1.2], 
                      [-1.2, 1.2, 1.2], 
                      [1.2, -1.2, -1.2], 
                      [-1.2, 1.2, -1.2], 
                      [-1.2, -1.2, 1.2], 
                      [-1.2, -1.2, -1.2]])
    pos = points
        
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
        mat = get_pose(pos, pos, np.array([0., 1., 0.]))
        scene.add(camera, pose=mat)
        # render scene
        flags = pyrender.constants.RenderFlags.SKIP_CULL_FACES
        r = pyrender.OffscreenRenderer(1024, 1024)
        color, _ = r.render(scene, flags=flags)
        path = os.path.join("./images/", instance)

        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, f"{i:02d}.png"), color)   

def as_mesh(scene_or_mesh):
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

# https://learnopengl.com/Getting-started/Camera
def get_pose(position_vector, front_vector, up_vector):
    m1 = np.eye(4)
    m2 = np.eye(4)
        
    # look at
    z = normalize_vector(front_vector)
    # up 
    y = normalize_vector(up_vector)
    # right
    x = normalize_vector(np.cross(y, z))
    # up
    y = normalize_vector(np.cross(z, x))
       
    m1[0, :3] = x
    m1[1, :3] = y
    m1[2, :3] = z
    m1[3, 3] = 1.0

    m2[0, 0] = m2[1, 1] = m2[2, 2] = 1.0
    m2[:3, 3] = -position_vector
    m2[3, 3] = 1.0

    return np.linalg.inv(np.matmul(m1, m2))

def normalize_vector(vector):
    return vector / (np.linalg.norm(vector))

def marching_cube(mesh_vertices, mesh_faces, res=256):
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
    verts = verts - 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces) 
    #mesh.show(flags={'cull': False})
    return mesh


if __name__=="__main__":
    # plane
    #obj_path = "../../ShapeNetCore.v2/02691156/"
    #instance = "121b5c1c81aa77906b153e6e0582b3ac"
    #instance = "150cdc45dabde04f7f29c61065b4dc5a"
    #instance = "150fd58e55111034761c6d3861a25da2"
    # car
    #obj_path = "../../ShapeNetCore.v2/02958343/"
    #instance = "12097984d9c51437b84d944e8a1952a5"
    #instance = "21205bdb7ca9be1d977e464a4b82757d"
    #instance = "21fcf7b6cfcd2c7933d7c9e122eec9b6"
    #
    #obj_path = "../../ShapeNetCore.v2/03642806/"
    #instance = "125c93cbc6544bd1f9f50a550b8c1cce"
    #instance = "129237ff9e4f5d5c95b5eddcdeed8f43"
    #instance = "151fccb37c2d51b0b82fa571a1cdbf24"
    obj_path = "../../ShapeNetCore.v2/03001627/"
    #instance = "122a480cfcdd742650c626aa72455dae"
    #instance = "12a56b6d02a93c5c711beb49b60c734f"
    #instance = "123305d8ccc0dc6346918a1d9c256af3"
    instance = "1d99f74a7903b34bd56bda2fb2008f9d"
    #instance = "1d828c69106609f8cd783766d090e665"
    #instance = "1e276a016b664e424d678187b8261d95"
    #instance = "1ee92a9d78cccbda98d2e7dbe701ca48"
    #instance = "ed7b1be61b8e78ac5d8eba92952b9366"
    #instance = "1ee92a9d78cccbda98d2e7dbe701ca48"
    #instance = "a11592a10d32207fd2c7b63cf34a2108"
    mesh_vertices, mesh_faces, obj_mesh = load_object(obj_path, instance)

    #obj_mesh = marching_cube(mesh_vertices, mesh_faces, 128)    

    #obj_mesh.show()

    render_image(obj_mesh, instance)
