import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d
from pyntcloud import PyntCloud
import glob
import plyfile as ply
from tqdm import tqdm


def combine (Nfiles, main_dir):
    # find all scan codes
    left_lidar_codes = glob.glob(os.path.join(main_dir, 'LIDAR_FRONT_LEFT', '*.ply'))
    left_lidar_codes = [f.split('/')[-1].split('_')[-1].split('.')[0] for f in left_lidar_codes]
    left_lidar_codes.sort()
    left_lidar_codes = left_lidar_codes[:Nfiles]

    right_lidar_codes = glob.glob(os.path.join(main_dir, 'LIDAR_FRONT_RIGHT', '*.ply'))
    right_lidar_codes = [f.split('/')[-1].split('_')[-1].split('.')[0] for f in right_lidar_codes]
    right_lidar_codes.sort()
    right_lidar_codes = right_lidar_codes[:Nfiles]

    top_lidar_codes = glob.glob(os.path.join(main_dir, 'LIDAR_TOP', '*.ply'))
    top_lidar_codes = [f.split('/')[-1].split('_')[-1].split('.')[0] for f in top_lidar_codes]
    top_lidar_codes.sort()
    top_lidar_codes = top_lidar_codes[:Nfiles]

    # find intersection of scan code
    scan_codes = set(left_lidar_codes).intersection(set(right_lidar_codes)).intersection(set(top_lidar_codes))
    scan_codes = list(scan_codes)
    scan_codes.sort()


    # combine lidar data
    all_lidars = ['LIDAR_FRONT_LEFT', 'LIDAR_FRONT_RIGHT', 'LIDAR_TOP']
    outdir = os.path.join(main_dir, 'LIDAR_COMBINED')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save combined point clouds and labels
    for code in tqdm(scan_codes):
        pcd_combined = o3d.geometry.PointCloud() 
        labels = []
        for lidar in all_lidars:
            file = os.path.join(main_dir, lidar, 'Town01_Opt_ClearSunset_' + code + '.ply')
            # load labels, i.e the  PlyProperty('ObjTag', 'uchar')), fourth property of the ply file
            plydata = ply.PlyData.read(file)
            labels += list(np.array(plydata.elements[0].data['ObjTag']))
            
            # load point clouds
            pc = o3d.io.read_point_cloud(file, format='ply')
            pcd_combined += pc

        # get labels
        labels = np.array(labels)
        #save labels
        np.save(os.path.join(outdir, 'Town01_Opt_ClearSunset_' + code + '_labels.npy'), labels)
        o3d.io.write_point_cloud(os.path.join(outdir, 'Town01_Opt_ClearSunset_' + code + '.ply'), pcd_combined)


def get_labels (path, save=False):
    plydata = ply.PlyData.read(path)
    labels = np.array(plydata.elements[0].data['ObjTag'])
    if save:
        np.save(path.split('.')[0] + '_labels.npy', labels)
    return labels

def plot_PC (path):
    import pyvista as pv
    pc = PyntCloud.from_file(path)
    points = np.asarray(pc.points)
    pv.plot(
        points,
        scalars=points[:, 2],
        render_points_as_spheres=True,
        point_size=5,
        show_scalar_bar=False,
        background='white',
        window_size=[300,300]
    )

def get_paths (main_dir):
    glob = os.path.join(main_dir, '*.ply')
    paths = glob.glob(glob)

    return paths

def get_images (
    path: str,
    theta_range:tuple = (-np.pi/2, np.pi/2), 
    phi_range:tuple = (0, 2*np.pi),
    pixel_size:tuple = (2, 2),
    ):
    """
    path: path to the ply file
    theta_range: range of theta to consider
    phi_range: range of phi to consider
    pixel_size: size of the pixel in degrees    

    returns:
    dictionary with 
        - key: label of the object
        - value: 2d representation of the object 
    """
    # load point cloud and labels
    pc = o3d.io.read_point_cloud(path, format='ply')
    labels = get_labels(path)
    all_points = np.asarray(pc.points)
    x, y, z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    
    # get spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)

    # theta = np.zeros(len(x))
    theta =  np.arctan(z , np.sqrt(x**2 + y**2))

    # we want a 360 range phi
    phi = np.zeros(len(x))
    xs = x[np.logical_and (x >= 0, y >= 0)]
    ys = y[np.logical_and (x >= 0, y >= 0)]
    phi[np.logical_and (x >= 0, y >= 0)] = np.arctan(ys, xs) # ranges 0, pi/2

    xs = x[x < 0]
    ys = y[x < 0]
    phi[x < 0] = np.arctan(ys, xs) + np.pi # ranges pi/2, pi

    xs = x[np.logical_and (x >= 0, y < 0)]
    ys = y[np.logical_and (x >= 0, y < 0)]
    phi[np.logical_and (x >= 0, y < 0)] = np.arctan(ys, xs) + 2*np.pi # ranges pi, 3pi/2


    # define the grid
    pixel_size_rad = (pixel_size[0]*np.pi/180, pixel_size[1]*np.pi/180)
    phi_grid = np.arange(phi_range[0], phi_range[1]+1, pixel_size_rad[0])
    theta_grid = np.arange(theta_range[0], theta_range[1]+1, pixel_size_rad[1])

    # get all the labels and image
    all_labels = np.unique(labels)
    result = dict()
    for l in all_labels:
        lmask = labels == l
        img = np.zeros((len(phi_grid), len(theta_grid)))
        for i, ph in enumerate(phi_grid):
            for j, th in enumerate(theta_grid):
                tmask = np.logical_and(theta>=th, theta<th+pixel_size[1])
                pmask = np.logical_and(phi>=ph, phi<ph+pixel_size[0])

                rs_pixel = r[tmask & pmask & lmask]

                if len(rs_pixel) > 0:
                    # print (f"pixel {i,j} has {len(rs_pixel)} points, min distance {np.min(rs_pixel)}")
                    min_index = np.argmin(rs_pixel)
                    img[i,j] = rs_pixel[min_index]
        result[l] = img

    return result
    

def reconstruct_3D(
        path_to_2d, 
        theta_range=(-np.pi/4, np.pi/4), 
        phi_range=(0, 2*np.pi), 
        pixel_size = (1,1), 
        save=False,
        save_path='reconstructed_PC'
        ):
    """
    path_to_2d: path to the 2d representation of the point cloud
    theta_range: range of theta to consider
    phi_range: range of phi to consider
    pixel_size: size of the pixel in degrees
    save: if true, saves the point cloud as a ply file
    """

    img = np.loadtxt(path_to_2d)

    pixel_size_rad = (pixel_size[0]*np.pi/180, pixel_size[1]*np.pi/180)
    phi_grid = np.arange(phi_range[0], phi_range[1]+1, pixel_size_rad[0])
    theta_grid = np.arange(theta_range[0], theta_range[1]+1, pixel_size_rad[1])
    # phi x theta
    # 3D reconstruction of the point cloud
    cartesian_coordinates = np.zeros((img.shape[0]*img.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = img[i, j]
            phi = phi_grid[i]
            theta = theta_grid[j]
            
            
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)

            cartesian_coordinates[i*img.shape[1] + j, 0] = x
            cartesian_coordinates[i*img.shape[1] + j, 1] = y
            cartesian_coordinates[i*img.shape[1] + j, 2] = z
    
    # create point cloud

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cartesian_coordinates)
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        o3d.io.write_point_cloud( os.path.join(save_path, path_to_2d.split('/')[-1].split('.')[0] + '.ply'), pcd)
    return pcd
