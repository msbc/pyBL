import h5py
# from mayavi import mlab
import argparse
import numpy as np
import sympy as sp
import matplotlib as mpl
if __name__ == "__main__":
    mpl.use('agg')
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy.stats import scoreatpercentile as percentile
# from scipy.stats import linregress
# import scipy.signal
# import gc
import os
# from glob import glob
# import sys
# import traceback

try:
    from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel, Box1DKernel
except ImportError:
    from scipy.signal import fftconvolve
# from scipy.ndimage.filters import convolve1d
# import time
# import tarfile
# import subprocess

# from .. import athena_read as ar
# from ..delayed_read import athdf
# from .. import helpers
# from ..helpers import rolling_weighted_triangle_conv as running_mean
# from ..helpers import grad, mod_grad
# from ..parmap import parmap
from .defaults import rc


tau = np.pi * 2


class MidpointNormalize(mpl.colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a
    prescribed midpoint value

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        #super(MidpointNormalize, self).__init__(vmin, vmax, clip)
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

class oomfmt(mpl.ticker.ScalarFormatter):
    def __init__(self, oom, **kwargs):
        self._oom = oom
        super(oomfmt, self).__init__(**kwargs)
        self.orderOfMagnitude = self._oom

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._oom

def r_main(mach):
    tmp = 9 / mach
    out = min(1 + 1.5 * tmp, 1 + 1.5 * tmp ** 2)
    out = min(out, 3)
    return out


def line_plt(p1, p2, **popt):
    if p1[0] == p2[0]:
        dth = np.abs(p1[1] - p2[1])
        nth = int(np.ceil(max(6, (dth / np.pi) * 100)))
        th = np.linspace(p1[1], p2[1], nth)
        plt.plot(p1[0] * np.sin(th), p1[0] * np.cos(th), **popt)
    elif p1[1] == p2[1]:
        th = p1[1]
        r = np.array([p1[0], p2[0]])
        plt.plot(r * np.sin(th), r * np.cos(th), **popt)


def plot_mesh(fn='mesh_structure.dat', data=None, save=False, fig_fn=None):
    if data is None:
        ax1_r = []
        ax1_th = []
        block = []
        with open(fn, 'r') as f:
            for line in f:
                line = line.strip()
                if (not line) or line[0] == '#':
                    if block:
                        block = np.array(block)
                        for i, point in enumerate(block):
                            ax1_r.append(point[0])
                            ax1_th.append(point[1])
                        ax1_r.append(np.nan)
                        ax1_th.append(np.nan)
                    block = []
                else:
                    block.append(list(map(float, line.split(' '))))

        ax1_r = np.array(ax1_r)
        ax1_th = np.array(ax1_th)
    else:
        ax1_th, ax1_r = data

    fig = plt.figure(figsize=(4, 8))
    ax = plt.subplot(111)  # , projection='polar')

    # ax = plt.subplot(121, projection='polar')
    # ax.plot(ax0_phi, ax0_r, 'k-', lw=1)

    # ax = plt.subplot(122, projection='polar')
    # ax.plot(ax1_th - .5 * np.pi, ax1_r, 'k-', lw=1)
    if 0:
        ax.plot(ax1_r * np.sin(ax1_th), ax1_r * np.cos(ax1_th), 'k-', lw=1)
    else:
        for i in range(ax1_r.size - 1):
            line_plt((ax1_r[i], ax1_th[i]), (ax1_r[i + 1], ax1_th[i + 1]), c='k', lw=1)
    ax.set_aspect('equal', 'datalim')
    # plt.xlim(0,None)

    if save or fig_fn:
        if fig_fn is None:
            fig_fn = '3d_grid.pdf'
        plt.savefig(fig_fn)
        plt.close()

    return ax1_th, ax1_r


def boxcar(data, n, axis=None):
    csum = data.cumsum(axis=axis)
    tmp = np.roll(csum, n, axis=axis)
    tmp[:, :n, :] = 0
    return csum - tmp


def spiral(r, rp, cs):
    return -np.sign(r - rp) * (2 / np.sqrt(r) + r / rp ** 1.5 - 3 / np.sqrt(rp)) / cs


def and_neighbor(data, n=1, axis=0, pad=True):
    loc = (slice(None),) * axis
    base = data.copy()
    if pad:
        pad = data[loc + (-1,)][loc + (np.newaxis,)]
        base = np.concatenate((base,) + (pad,) * n, axis=axis)
    out = np.ones_like(base, dtype=bool)
    for i in range(-n, n + 1):
        out = np.logical_and(out, np.roll(base, n, axis=axis))
    return out[loc + (slice(0, data.shape[axis]),)]


def crudeDiff(t, data, axis=0, front=True):
    dt = np.diff(t)[(np.newaxis,) * axis + (slice(None),) + (np.newaxis,) * max(0, (
            len(data.shape) - axis - 1))]
    out = np.diff(data, axis=axis) / dt
    if front:
        loc = (slice(None),) * axis + ([0],)
        return np.concatenate((out[loc], out), axis=axis)
    else:
        loc = (slice(None),) * axis + ([-1],)
        return np.concatenate((out, out[loc]), axis=axis)


def lindblad_loc(ps, m=1):
    mtt = -2. / 3.
    corot = ps ** mtt
    return np.array([(m/(m-1))**mtt, 1, (m/(m+1))**mtt]) * corot


def smooth(data, width=64):
    try:
        len(width)
    except TypeError:
        width = width, 1
    kern = np.ones(width)
    kern /= kern.sum()
    try:
        return convolve_fft(data, kern, boundary='wrap', allow_huge=True)
    except NameError:
        return fftconvolve(data, kern, mode='same')


_smooth = smooth


def findAbsPath(fn, sim_path=None):
    if (sim_path is None) and (not os.path.isfile(fn)):
        for d in rc('dirs'):
            tmp = os.path.join(d, fn)
            if os.path.isfile(tmp):
                fn = tmp
                break
    elif sim_path:
        fn = os.path.join(sim_path, fn)
    return fn


def intr(dr, data, axis=-1):
    if axis < 0:
        axis += len(data.shape)
    loc = [np.newaxis] * axis + [slice(None)]
    return (data * dr).sum(axis=axis)


def parse_not_overwrite(overwrite, fn):
    if overwrite:
        return False
    try:
        if os.path.isfile(fn):
            return True
    except TypeError:
        pass
    return False

def lintrend(x, y, axis=0):
    d = y.ndim - x.ndim
    if d > 0:
        x = x[tuple([slice(None)] + d * [np.newaxis])]
    if d < 0:
        y = y[tuple([slice(None)] + (-d) * [np.newaxis])]
    xdx = (x * x).sum(axis=axis)
    xdy = (x * y).sum(axis=axis)
    denom = xdx - x.mean(axis=axis) * x.sum(axis=axis)
    m = (xdy - y.mean(axis=axis)) / denom
    b = (y.mean(axis=axis) * xdx - x.mean(axis=axis) * xdy)
    loc = [np.newaxis] + [slice(None)] * (x.ndim - 1)
    return m[loc], b[loc]

def phi_visible(_r, i, b=1):
    one = np.ones_like(_r * i)
    r = one * _r[:]
    out = np.zeros_like(r)
    a = one * b / np.cos(i)
    nanloc = np.where(one * i == .5 * np.pi)
    a[nanloc] = np.infty
    out[r >= a] = tau
    rloc = np.where(np.logical_and(r > b, r < a))
    rsq = r[rloc]**2
    _a = a[rloc]
    out[rloc] = np.pi + 2 * np.arctan(_a / b * np.sqrt((rsq - b**2) / (_a**2 - rsq)))
    out[nanloc] = np.pi + 2 * np.arctan(np.sqrt(r[nanloc]**2 - b**2) / b)
    return out
