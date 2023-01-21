import argparse
import numpy as np
import open3d
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import CustomData
from models import IterativeBenchmark
from loss import EMDLosspy
from metrics import compute_metrics, summary_metrics, print_train_info
from utils import time_calc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    ## dataset
    parser.add_argument('--root',help='the data path', default='dataset_final')
    parser.add_argument('--train_npts', type=int,  default=2048,
                        help='the points number of each pc for training')
    ## models training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gn', action='store_true',
                        help='whether to use group normalization')
    parser.add_argument('--epoches', type=int, default=20)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--in_dim', type=int, default=3,
                        help='3 for (x, y, z) or 6 for (x, y, z, nx, ny, nz)')
    parser.add_argument('--niters', type=int, default=8,
                        help='iteration nums in one model forward')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--milestones', type=list, default=[50, 250],
                        help='lr decays when epoch in milstones')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='lr decays to gamma * lr every decay epoch')
    # logs
    parser.add_argument('--saved_path', default='work_dirs/models',
                        help='the path to save training logs and checkpoints')
    parser.add_argument('--saved_frequency', type=int, default=1,
                        help='the frequency to save the logs and checkpoints')
    args = parser.parse_args()
    return args


def compute_loss(ref_cloud, pred_ref_clouds, loss_fn):
    losses = []
    discount_factor = 0.5
    for i in range(8):
        loss = loss_fn(ref_cloud[..., :3].contiguous(),
                       pred_ref_clouds[i][..., :3].contiguous())
        losses.append(discount_factor**(8 - i)*loss)
    return torch.sum(torch.stack(losses))


@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer):
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for ref_cloud, src_cloud, gtR, gtt in tqdm(train_loader):
        ref_cloud, src_cloud, gtR, gtt = ref_cloud.to(device), src_cloud.to(device), \
                                         gtR.to(device), gtt.to(device)
        optimizer.zero_grad()
        R, t, pred_ref_clouds = model(src_cloud.permute(0, 2, 1).contiguous(),
                                     ref_cloud.permute(0, 2, 1).contiguous())
        loss = compute_loss(ref_cloud, pred_ref_clouds, loss_fn)
        loss.backward()
        optimizer.step()

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        losses.append(loss.item())
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


@time_calc
def test_one_epoch(test_loader, model, loss_fn):
    model.eval()
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for ref_cloud, src_cloud, gtR, gtt in tqdm(test_loader):
            ref_cloud, src_cloud, gtR, gtt = ref_cloud.to(device), src_cloud.to(device), \
                                             gtR.to(device), gtt.to(device)
            R, t, pred_ref_clouds = model(src_cloud.permute(0, 2, 1).contiguous(),
                                         ref_cloud.permute(0, 2, 1).contiguous())
            loss = compute_loss(ref_cloud, pred_ref_clouds, loss_fn)
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)

            losses.append(loss.item())
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    model.train()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


def main():
    args = config_params()
    print(args)

    print ('Setting up data...')
    setup_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    summary_path = os.path.join(args.saved_path, 'summary')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(args.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    train_set = CustomData(args.root, args.train_npts)
    test_set = CustomData(args.root, args.train_npts, False)
    train_loader = DataLoader(train_set, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False,
                             num_workers=args.num_workers)

    # test dataset
    print ('test dataset', len(test_set))
    print ('train dataset', len(train_set))

    # load model if exists from checkpoints
    import glob
    model_paths = glob.glob(os.path.join(checkpoints_path, 'model_*.pth'))
    model_paths.sort()
    epochs = [int(path.split('_')[-1].split('.')[0]) for path in model_paths]
    start_epoch = max(epochs)+1 if len(epochs) > 0 else 0
    model = IterativeBenchmark(in_dim=args.in_dim, niters=args.niters, gn = args.gn)

    if len(model_paths) > 0:
        model_path = os.path.join(checkpoints_path, 'model_{}.pth'.format(start_epoch))
        print('load model from {}'.format(model_path))
        model.load_state_dict(torch.load(model_path))

        print('start from epoch {}'.format(start_epoch))
        max_epoch = args.epoches
        print ('max epoch', max_epoch)
    else: 
        max_epoch = args.epoches

    model = model.to(device)

    loss_fn = torch.nn.MSELoss()
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.milestones,
                                                     gamma=args.gamma,
                                                     last_epoch=-1)

    

    # writer = SummaryWriter(summary_path)
    writer ={'Loss/train': [], 'Loss/test': [], 'RError/train': [], 'RError/test': [],
                    'TError/train': [], 'TError/test':[] }

    test_min_loss, test_min_r_mse_error, test_min_rot_error = \
        float('inf'), float('inf'), float('inf')
    for epoch in range(start_epoch, max_epoch):
        print('=' * 20, epoch + 1, '=' * 20)
        train_results = train_one_epoch(train_loader, model, loss_fn, optimizer)
        print_train_info(train_results)
        test_results = test_one_epoch(test_loader, model, loss_fn)
        print_train_info(test_results)

        if epoch % args.saved_frequency == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_path,
                                                        'model_{}.pth'.format(epoch)))
            writer ['Loss/train'] += [train_results['loss']]
            writer ['Loss/test'] += [test_results['loss']]
            writer ['RError/train'] += [train_results['r_mse']]
            writer ['RError/test'] += [test_results['r_mse']]

        test_loss, test_r_error, test_rot_error = \
            test_results['loss'], test_results['r_mse'], test_results[
                'r_isotropic']
        if test_loss < test_min_loss:
            saved_path = os.path.join(checkpoints_path, "test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_loss = test_loss
        if test_r_error < test_min_r_mse_error:
            saved_path = os.path.join(checkpoints_path,
                                      "test_min_rmse_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_r_mse_error = test_r_error
        if test_rot_error < test_min_rot_error:
            saved_path = os.path.join(checkpoints_path,
                                      "test_min_rot_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_rot_error = test_rot_error
        scheduler.step()
    # save writer
    import pickle
    with open(os.path.join(summary_path, 'writer.pkl'), 'wb') as f:
        pickle.dump(writer, f)



if __name__ == '__main__':
    main()