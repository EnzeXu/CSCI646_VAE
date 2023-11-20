"""main.py"""

import argparse

import numpy as np
import torch
import wandb

from solver_VAE import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    ## random seeds
    seed = args.seed
    use_cuda = torch.cuda.is_available() ## if have gpu or cpu
    if use_cuda:
        device = torch.device('cuda', args.gpu)
        # torch.manual_seed(args.seed)
        # torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
        # print("----using CPU now----")
    
    # np.random.seed(seed)
    net = Solver(args)
    
    if args.train:
        print("*** [start training]***")
        with wandb.init(project='CSCI646-VAE', name='Default'):
            net.train()
    else:
        net.viz_traverse()
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--load_ckp', default=False, type=str2bool, help='load checkpoint')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--limit', default=3, type=float, help='traverse limits')
    parser.add_argument('--KL_loss', default=20, type=float, help='KL_divergence')

    parser.add_argument('--z_dim', default=15, type=int, help='dimension of the representation z')

    parser.add_argument('--out_dim', default=10, type=int, help='dimension of predict y')
    # parser.add_argument('--beta', default=120, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--beta', default=2, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=10, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--alpha', default=10, type=float, help='alpha parameter for Z latent loss')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    # parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='VAE', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=50000, type=int, help='numer of iterations after which data is gathered for visdom')
    # parser.add_argument('--display_step', default=50000, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved')
    
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    
    args = parser.parse_args()
    
    main(args)
    
