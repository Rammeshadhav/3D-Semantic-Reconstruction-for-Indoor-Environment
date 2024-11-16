#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import csv
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
data_3D = np.loadtxt("points3D.txt")
xyz=data_3D[:,1:4].astype(float)
rgb=data_3D[:,4:7].astype(int)
points_3D = np.loadtxt("sparse_3D_points.txt")
rgb_3D = np.loadtxt("sparse_rgb.txt")
neigh = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean')
neigh.fit(points_3D)
index=0
for point in xyz:
    distances, indices = neigh.kneighbors([point])
    ind=indices[0,:]
    rgb_list =[rgb_3D[k] for k in ind]
    rgb_list = np.array(rgb_list)
    count = dict()
    for sub_arr in rgb_list:
        sub_arr_str = str(sub_arr.tolist())
        if sub_arr_str in count:
            count[sub_arr_str] += 1
        else:
            count[sub_arr_str] = 1

    most_freq = max(count, key=count.get)
    most_freq_arr = np.array(eval(most_freq))
    rgb[index,0]=most_freq_arr[0]
    rgb[index,1]=most_freq_arr[1]
    rgb[index,2]=most_freq_arr[2]
    index+=1
    

with open("Dense_3D_points.txt", 'w') as f:
        csv.writer(f, delimiter=' ').writerows(xyz)
        

with open("Dense_rgb.txt", 'w') as f:
        csv.writer(f, delimiter=' ').writerows(rgb)    


points_array=np.array(xyz)
rgb_array=np.array(rgb)
rgb_array=rgb_array/255
points_array=points_array.astype(float)
rgb_array=rgb_array.astype(float)
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(points_array)
pc.colors = o3d.utility.Vector3dVector(rgb_array)
pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20,max_nn=50))
o3d.io.write_point_cloud("Dense_Semantic_point_cloud.ply", pc)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)
# Save the triangle mesh
o3d.io.write_triangle_mesh('Total_dense_meshed.ply', mesh)