import numpy as np
import open3d as o3d
from open3d import *


def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)


def save_pcd(filename, points):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)