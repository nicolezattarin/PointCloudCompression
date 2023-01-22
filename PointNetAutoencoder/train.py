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


# soppress user warnings
import warnings
warnings.filterwarnings("ignore")
def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    ## dataset
    parser.add_argument('--root',help='the data path', default='dataset_final')
    parser.add_argument('--load', type=bool, default=True,
                        help='whether to load the trained model')
    parser.add_argument('--train_npts', type=int,  default=4000,
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

import sys

@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer):
    losses = []
    compression_ratios = []
    for ref_cloud in tqdm(train_loader):
        ref_cloud = ref_cloud.to(device)

        optimizer.zero_grad()
        encoded, decoded = model(ref_cloud.permute(0,2,1).contiguous())

        loss = loss_fn(ref_cloud.permute(0,2,1).contiguous(), decoded)

        #product shapes normalized over batch_size
        encoding_size = encoded.shape[1]
        ref_cloud_size = ref_cloud.shape[1] * ref_cloud.shape[2]
        
        compression_ratio = encoding_size / ref_cloud_size
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        compression_ratios.append(compression_ratio)
    
    results = {
        'loss': np.mean(losses),
        'compression_ratio': np.mean(compression_ratios)
    }
    return results


@time_calc
def test_one_epoch(test_loader, model, loss_fn):
    model.eval()
    losses = []
    compression_ratios = []

    with torch.no_grad():
        for ref_cloud in tqdm(test_loader):
            ref_cloud = ref_cloud.to(device)
            encoded, decoded = model(ref_cloud.permute(0,2,1).contiguous())

            loss = loss_fn(ref_cloud.permute(0,2,1).contiguous(), decoded)

            #product shapes normalized over batch_size
            encoding_size = encoded.shape[1]
            ref_cloud_size = ref_cloud.shape[1] * ref_cloud.shape[2]
            
            compression_ratio = encoding_size / ref_cloud_size

            losses.append(loss.item())
            compression_ratios.append(compression_ratio) 
    model.train()
    results = {
        'loss': np.mean(losses),
        'compression_ratio': np.mean(compression_ratios)
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
    start_epoch = max(epochs) if len(epochs) > 0 else 0
    from autoencoder import Autoencoder
    # model = Autoencoder(in_dim=args.in_dim,  gn = args.gn)
    model = Autoencoder()
    if len(model_paths) > 0 and args.load:
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

    


    test_min_loss  = float('inf')
    import pandas as pd
    train_results_all = pd.DataFrame(columns=['loss', 'compratio'])
    test_results_all = pd.DataFrame(columns=['loss', 'compratio'])

    # save loss and compression ratio
    if not os.path.exists(os.path.join(checkpoints_path, 'results')):
        os.makedirs(os.path.join(checkpoints_path, 'results'))

    # load results if exists
    if os.path.exists(os.path.join(checkpoints_path, 'results', 'train_results_all.csv')):
        train_results_all = pd.read_csv(os.path.join(checkpoints_path, 'results', 'train_results_all.csv'))
        test_results_all = pd.read_csv(os.path.join(checkpoints_path, 'results', 'test_results_all.csv'))
        print('load results from {}'.format(os.path.join(checkpoints_path, 'results')))
        print('start from epoch {}'.format(start_epoch))
        max_epoch = args.epoches
        print ('max epoch', max_epoch)

    for epoch in range(start_epoch, max_epoch):
        print('=' * 20, epoch + 1, '=' * 20)
        train_results = train_one_epoch(train_loader, model, loss_fn, optimizer)
        test_results = test_one_epoch(test_loader, model, loss_fn)
        print('train loss: {:.4f}, train compratio: {:.4f}'.format(train_results['loss'], train_results['compression_ratio']))

        # save loss and compression ratio
        train_results_all = pd.concat([train_results_all, pd.DataFrame(train_results, index=[0])], ignore_index=True)
        test_results_all = pd.concat([test_results_all, pd.DataFrame(test_results, index=[0])], ignore_index=True)
        
        test_loss = test_results['loss']

        test_min_loss = min(test_min_loss, test_loss)
        if test_loss < test_min_loss:
            saved_path = os.path.join(checkpoints_path, "test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_loss = test_loss
        
        saved_path = os.path.join(checkpoints_path, "model_{}.pth".format(epoch))
        torch.save(model.state_dict(), saved_path)


        scheduler.step()

        train_results_all.to_csv(os.path.join(checkpoints_path, 'results', 'train_results_all.csv'), index=False)
        test_results_all.to_csv(os.path.join(checkpoints_path, 'results', 'test_results_all.csv'), index=False)


if __name__ == '__main__':
    main()