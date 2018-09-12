#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
#from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gc
import os
from glob import glob
import sys

from . import athena_read as ar
from . import helpers

hpi = 1.5708
zero = 0.

def plot_mesh(fn='/home/mcoleman/BLayer/3d_test/M09_a/mesh_structure.dat', data=None, save=False, fig_fn=None):
    if data is None:
        ax0_r = []
        ax0_phi = []
        ax1_r = []
        ax1_th = []
        block = []
        with open(fn, 'r') as f:
            for line in f:
                line = line.strip()
                if (not line) or line[0] == '#':
                    if block:
                        block = np.array(block)
                        if (block[:,1] == hpi).any():
                            for i, point in enumerate(block):
                                if point[1] == hpi:
                                    ax0_r.append(point[0])
                                    ax0_phi.append(point[2])
                                    try:
                                        if block[i+1, 1] != hpi:
                                            ax0_r.append(np.nan)
                                            ax0_phi.append(np.nan)
                                    except IndexError:
                                        ax0_r.append(np.nan)
                                        ax0_phi.append(np.nan)
                        if (block[:,2] == zero).any():
                            for i, point in enumerate(block):
                                if point[2] == zero:
                                    ax1_r.append(point[0])
                                    ax1_th.append(point[1])
                                    try:
                                        if block[i+1, 2] != zero:
                                            ax1_r.append(np.nan)
                                            ax1_th.append(np.nan)
                                    except IndexError:
                                        ax1_r.append(np.nan)
                                        ax1_th.append(np.nan)
                    block = []
                else:
                    block.append(list(map(float, line.split(' '))))

        ax0_r = np.array(ax0_r)
        ax0_phi = np.array(ax0_phi)
        ax1_r = np.array(ax1_r)
        ax1_th = np.array(ax1_th)
    else:
        ax0_phi, ax0_r, ax1_th, ax1_r = data

    fig = plt.figure(figsize=(4,8))
    ax = plt.subplot(111)#, projection='polar')

    #ax = plt.subplot(121, projection='polar')
    #ax.plot(ax0_phi, ax0_r, 'k-', lw=1)

    #ax = plt.subplot(122, projection='polar')
    #ax.plot(ax1_th - .5 * np.pi, ax1_r, 'k-', lw=1)
    ax.plot(ax1_r * np.sin(ax1_th), ax1_r * np.cos(ax1_th), 'k-', lw=1)
    ax.set_aspect('equal', 'datalim')
    plt.xlim(0,None)

    if save or fig_fn:
        if fig_fn is None:
            fig_fn = '3d_grid.pdf'
        plt.savefig(fig_fn)
        plt.close()

    phi = ax0_phi[np.isfinite(ax0_phi)]
    print("n_phi", 2 * np.pi / phi[phi > 0].min())
    th = ax1_th[np.isfinite(ax1_th)]
    print("n_theta", np.pi / (th[th > hpi].min() - hpi))

    return ax0_phi, ax0_r, ax1_th, ax1_r

