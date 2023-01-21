import argparse
import numpy as np
import open3d as o3d
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import CustomData
from models import IterativeBenchmark, icp
from metrics import compute_metrics, summary_metrics, print_metrics
from utils import npy2pcd, pcd2npy
import os

def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--root',help='the data path', default='dataset_final')
    parser.add_argument('--infer_npts', type=int, default=2048,
                        help='the points number of each pc for training')
    parser.add_argument('--in_dim', type=int, default=3,
                        help='3 for (x, y, z) or 6 for (x, y, z, nx, ny, nz)')
    parser.add_argument('--niters', type=int, default=8,
                        help='iteration nums in one model forward')
    parser.add_argument('--gn', action='store_true',
                        help='whether to use group normalization')
    parser.add_argument('--checkpoint', default='work_dirs/models/checkpoints/test_min_loss.pth',
                        help='the path to the trained checkpoint')
    parser.add_argument('--method', default='benchmark',
                        help='choice=[benchmark, icp]')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use the cuda')
    parser.add_argument('--show', action='store_true', default=False,
                        help='whether to visualize')
    parser.add_argument('--save_dir',  default='results', type=str,
                        help='whether to save')
    args = parser.parse_args()
    return args

from scipy.spatial import KDTree
def mean_point_distance_error(points, decoded_points):

    # Create a KDTree of the decoded points
    decoded_points_tree = KDTree(decoded_points)
    # Find the closest point in the decoded point cloud for each point in the original point cloud
    distances, closest_indices = decoded_points_tree.query(points)
    # Get the closest points
    closest_points = decoded_points[closest_indices]
    # Calculate the mean point distance error
    mpde = np.mean(np.linalg.norm(points - closest_points, axis=1))
    return mpde


def evaluate_benchmark(args, test_loader):
    model = IterativeBenchmark(in_dim=args.in_dim,
                               niters=args.niters,
                               gn=args.gn)
    if args.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    # 

    dura = []
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for i, (ref_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):
            if args.cuda:
                ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                                 gtR.cuda(), gtt.cuda()
            tic = time.time()
            R, t, pred_ref_cloud = model(src_cloud.permute(0, 2, 1).contiguous(),
                    ref_cloud.permute(0, 2, 1).contiguous())

            # mse 

            mse = torch.nn.MSELoss() 
            loss =mean_point_distance_error (pred_ref_cloud[0].cpu().detach().numpy(), ref_cloud.cpu().detach().numpy())
            losses.append(loss.item())

            toc = time.time()
            dura.append(toc - tic)
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

            if args.show:
                ref_cloud = torch.squeeze(ref_cloud).cpu().numpy()
                src_cloud = torch.squeeze(src_cloud).cpu().numpy()
                pred_ref_cloud = torch.squeeze(pred_ref_cloud[-1]).cpu().numpy()
                pcd1 = npy2pcd(ref_cloud, 0)
                pcd2 = npy2pcd(src_cloud, 1)
                pcd3 = npy2pcd(pred_ref_cloud, 2)
                o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

            

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic, losses


def evaluate_icp(args, test_loader):
    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for i, (ref_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):
        if args.cuda:
            ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                             gtR.cuda(), gtt.cuda()

        ref_cloud = torch.squeeze(ref_cloud).cpu().numpy()
        src_cloud = torch.squeeze(src_cloud).cpu().numpy()

        tic = time.time()
        R, t, pred_ref_cloud = icp(npy2pcd(src_cloud), npy2pcd(ref_cloud))
        toc = time.time()
        R = torch.from_numpy(np.expand_dims(R, 0)).to(gtR)
        t = torch.from_numpy(np.expand_dims(t, 0)).to(gtt)
        dura.append(toc - tic)

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

        if args.show:
            pcd2 = npy2pcd(src_cloud, 1)
            o3d.visualization.draw_geometries([pcd2])

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic

def save_metrics (args, dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic):
    with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
        f.write('dura: {}\n'.format(dura))
        f.write('r_mse: {}\n'.format(r_mse))
        f.write('r_mae: {}\n'.format(r_mae))
        f.write('t_mse: {}\n'.format(t_mse))
        f.write('t_mae: {}\n'.format(t_mae))
        f.write('r_isotropic: {}\n'.format(r_isotropic))
        f.write('t_isotropic: {}\n'.format(t_isotropic))

if __name__ == '__main__':
    seed = 222
    random.seed(seed)
    np.random.seed(seed)

    args = config_params()

    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_set = CustomData(args.root, args.infer_npts, False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if args.method == 'benchmark':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic, losses = \
            evaluate_benchmark(args, test_loader)
        print_metrics(args.method,
                      dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
        

        save_metrics(args, dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

        # save losses
        with open(os.path.join(args.save_dir, 'losses.txt'), 'w') as f:
            f.write('losses: {}\n'.format(losses))


    elif args.method == 'icp':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
            evaluate_icp(args, test_loader)
        print_metrics(args.method, dura, r_mse, r_mae, t_mse, t_mae, r_isotropic,
                      t_isotropic)
    else:
        raise NotImplementedError