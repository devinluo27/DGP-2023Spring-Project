import numpy as np
import trimesh
import os
import pyrender
import cv2


# from nerf-navigation
def rotateAxis(degrees, axis):
    '''
    Function to rotate around given axis
    Input:
        degrees - scalar - Angle in degrees
        
        axis - scalar - options:
            0 - around x axis
            1 - around y axis
            2 - around z axis  
    
    Returns:
        Homogeneous rotation matrix
    '''

    radians = np.radians(degrees)

    if axis == 2: # z - axis

        rotation_mat = np.array([[np.cos(radians), -np.sin(radians),           0,          0],
                                 [np.sin(radians),  np.cos(radians),           0,          0],
                                 [              0,                0,           1,          0],
                                 [              0,                0,           0,          1]])

    elif axis == 1: # y - axis

        rotation_mat = np.array([[np.cos(radians),                0,  np.sin(radians),          0],
                                 [              0,                1,                0,          0],
                                 [-np.sin(radians),               0, np.cos(radians),          0],
                                 [              0,                0,                0,          1]])

    elif axis == 0: # x - axis


        rotation_mat = np.array([[             1,                0,                0,          0],
                                [              0,  np.cos(radians), -np.sin(radians),          0],
                                [              0,  np.sin(radians),  np.cos(radians),          0], 
                                [              0,                0,                0,          1]])
    
    return rotation_mat


if __name__=="__main__":
    # plane
    obj_path = "../../ShapeNetDGPV3/"
    #instance = "1d99f74a7903b34bd56bda2fb2008f9d"
    #instance = "1ee92a9d78cccbda98d2e7dbe701ca48"
    instance = "1d828c69106609f8cd783766d090e665"

    path = os.path.join(obj_path, instance)

    xyz = np.load(os.path.join(path, "xyzsdf.npy"))

    #print(xyz.shape)

    xyz = xyz[xyz[:, 3]>=0][:, :3]

    xyz[:, 2] = 1 - xyz[:, 2]
    
    xyz  = xyz @ rotateAxis(-45, 2)[:3, :3] @ rotateAxis(15, 0)[:3, :3]

    pts = trimesh.points.PointCloud(xyz)
    pts.show()
