"""utils.py"""

import argparse
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable
# import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def str2bool(v):
    # codes from : stackover

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, x, y):
    """Do same operation as np.where

    code from:
        //discuss.pytorch.org/
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        //stackoverflow.com/
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)


def _visual_tsne(img_z, y, viz_name):
    '''use tsne to visualize the results'''
    # print('emb z shape: ', y.shape)
    path = os.path.join('TSNE-Visual',viz_name)
    os.makedirs(path, exist_ok=True)

    tsne = TSNE(n_components=2, verbose=1, perplexity=30, random_state=213)
    tsne_result = tsne.fit_transform(img_z)
    # print('tsne_result shape', tsne_result.shape)
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x="tsne_1", y="tsne_2", hue='label',
                    palette=sns.color_palette("hls", 10),
                    data=tsne_result_df).set(title=viz_name)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(path + '/embedding.png')
    # plt.show()

