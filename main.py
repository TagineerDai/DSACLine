import torch
import torch.optim as optim
import torchvision.utils as vutils

import os
import time
import numpy
import warnings
import argparse

from skimage.io import imsave

from line_dataset import LineDataset
from line_nn import LineNN
from line_loss import LineLoss

from dsac import DSAC

parser = argparse.ArgumentParser(
    description='This script creates a toy problem of fitting line parameters (slope+intercept) to synthetic images '
                'showing line segments, noise and distracting circles. Two networks are trained in parallel and '
                'compared: DirectNN predicts the line parameters directly (two output neurons). PointNN predicts a '
                'number of 2D points to which the line parameters are subsequently fitted using differentiable RANSAC '
                '(DSAC). The script will produce a sequence of images that illustrate the training process for both '
                'networks.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--receptivefield', '-rf', type=int, default=65, choices=[65, 51, 37, 29, 15, 0],
                    help='receptive field size of the PointNN, i.e. one point prediction is made for each image patch '
                         'of this size, different receptive fields are achieved by different striding strategies, '
                         '0 means global, i.e. the full image, the DirectNN will always use 0 (global)')

parser.add_argument('--capacity', '-c', type=int, default=4,
                    help='controls the model capactiy of both networks (PointNN and DirectNN), it is a multiplicative '
                         'factor for the number of channels in each network layer')

parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                    help='number of line hypotheses sampled for each image')

parser.add_argument('--inlierthreshold', '-it', type=float, default=0.05,
                    help='threshold used in the soft inlier count. Its measured in relative image size '
                         '(1 = image width)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=0.5,
                    help='scaling factor for the soft inlier scores (controls the peakiness of '
                         'the hypothesis distribution)')

parser.add_argument('--inlierbeta', '-ib', type=float, default=100.0,
                    help='scaling factor within the sigmoid of the soft inlier count')

parser.add_argument('--learningrate', '-lr', type=float, default=0.001,
                    help='learning rate')

parser.add_argument('--lrstep', '-lrs', type=int, default=2500,
                    help='cut learning rate in half each x iterations')

parser.add_argument('--lrstepoffset', '-lro', type=int, default=30000,
                    help='keep initial learning rate for at least x iterations')

parser.add_argument('--batchsize', '-bs', type=int, default=32,
                    help='training batch size')

parser.add_argument('--trainiterations', '-ti', type=int, default=50,
                    help='number of training iterations (= parameter updates)')

parser.add_argument('--imagesize', '-is', type=int, default=64,
                    help='size of input images generated, images are square')

parser.add_argument('--storeinterval', '-si', type=int, default=1000,
                    help='store network weights and a prediction vizualisation every x training iterations')

parser.add_argument('--valsize', '-vs', type=int, default=9,
                    help='number of validation images used to vizualize predictions')

parser.add_argument('--valthresh', '-vt', type=float, default=5,
                    help='threshold on the line loss for vizualizing correctness of predictions')

parser.add_argument('--cpu', '-cpu', action='store_true',
                    help='execute networks on CPU. Note that (RANSAC) line fitting anyway runs on CPU')


def prepare_model(option):
    loss = LineLoss(option.imagesize)
    dsac = DSAC(option.hypotheses, option.inlierthreshold, option.inlierbeta, option.inlieralpha, loss)
    point_nn = LineNN(option.capacity, option.receptivefield)
    if not option.cpu:
        point_nn = point_nn.cuda()
    point_nn.train()
    opt_point_nn = optim.Adam(point_nn.parameters(), lr=opt.learningrate)
    lrs_point_nn = optim.lr_scheduler.StepLR(opt_point_nn, opt.lrstep, gamma=0.5)
    return point_nn, dsac, opt_point_nn, lrs_point_nn


def prepare_data(inputs, labels, option):
    # convert from numpy images to normalized torch arrays

    inputs = torch.from_numpy(inputs)
    labels = torch.from_numpy(labels)

    if not option.cpu:
        inputs = inputs.cuda()
    inputs.transpose_(1, 3).transpose_(2, 3)
    inputs = inputs - 0.5  # normalization

    return inputs, labels


if __name__ == "__main__":
    opt = parser.parse_args()
    dataset = LineDataset(opt.imagesize, opt.imagesize)
    point_nn, dsac, optimizer, lrsceduler = prepare_model(opt)

    for iteration in range(0, opt.trainiterations + 1):

        start_time = time.time()

        # generate data
        inputs, labels = dataset.sample_lines(opt.batchsize)
        inputs, labels = prepare_data(inputs, labels, opt)
        # robust fitting via DSAC
        point_prediction = point_nn(inputs)
        exp_loss, top_loss = dsac(point_prediction, labels)

        exp_loss.backward()  # calculate gradients (pytorch autograd)
        optimizer.step()  # update parameters
        optimizer.zero_grad()  # reset gradient buffer
        if iteration >= opt.lrstepoffset:
            lrsceduler.step()  # update learning rate schedule

        # wrap up
        end_time = time.time() - start_time
        print('Iteration: %6d, DSAC Expected Loss: %2.2f, DSAC Top Loss: %2.2f, Time: %.2fs'
              % (iteration, exp_loss, top_loss, end_time))

    print('Done without errors.')
