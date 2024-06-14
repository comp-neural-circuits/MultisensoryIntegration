#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
#style.use('seaborn')
import seaborn as sns
sns.set_theme()
#sns.set_palette("colorblind")

# defining things for the figures
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font',  size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes',  titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes',  labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend',  fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure',  titlesize=BIGGER_SIZE)  # fontsize of the figure title


def making_figures(path, W_thres, W_v1, W_s1, W_vec_v1, W_vec_s1, time_vec_list, time_points):

    fig = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplot(1, 2, 1)
    im = plt.imshow(W_v1, vmin=0, vmax=W_thres[1]/1, cmap='Reds')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.xlabel('V1')
    plt.ylabel('RL')
    plt.subplot(1, 2, 2)
    im = plt.imshow(W_s1, vmin=0, vmax=W_thres[1]/1, cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.xlabel('S1')
    plt.ylabel('RL')
    plt.savefig(path / 'sample2_RF.pdf')
    plt.close()

    fig = plt.subplots(5, 5, figsize=(12, 12))
    plt.suptitle('V1 refinement')
    for ii in range(round(time_points)):
        plt.subplot(5, 5, ii + 1)
        plt.imshow(W_vec_v1[ii][:][:], vmin=W_thres[0], vmax=W_thres[1], cmap='Reds')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Time = ' + str(time_vec_list[ii]/1000))
        plt.axis('off')
    plt.savefig(path / 'samples_v1.pdf')
    plt.close()

    fig = plt.subplots(5, 5, figsize=(12, 12))
    plt.suptitle('S1 refinement')
    for ii in range(round(time_points)):
        plt.subplot(5, 5, ii + 1)
        plt.imshow(W_vec_s1[ii][:][:], vmin=W_thres[0], vmax=W_thres[1], cmap='Blues')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Time = ' + str(time_vec_list[ii]/1000))
        plt.axis('off')
    plt.savefig(path / 'samples_s1.pdf')
    plt.close()

    np.savetxt(path / 'V1_RL.txt', W_v1, fmt="%.4f", delimiter=',')
    np.savetxt(path / 'S1_RL.txt', W_s1, fmt="%.4f", delimiter=',')
