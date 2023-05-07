import trimesh
import numpy as np
from skimage.measure import marching_cubes

def marching_cube(net, res=256, write_path=None):
    # sample xyz from grid
    lin_coords = np.linspace(-1.0, 1.0, res)
    x_coords, y_coords, z_coords = np.meshgrid(lin_coords, lin_coords, lin_coords)
    xyz = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(-1, 3)
        
    # get sdf
    sdf = net(xyz) 
 
    # make implicit value
    threshold = 0
    implicit_values = sdf.reshape(res, res, res)
    implicit_values = np.where(implicit_values>0, 1, -1)
       
    # run marching cube
    spacing = 2.*1.0/(res-1)
    verts, faces, _, _ = marching_cubes(implicit_values, level=0.0, spacing=(spacing, spacing, spacing))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces) 
    mesh.show(flags={'cull': False})
    
    # stl file
    mesh.export(write_path)
