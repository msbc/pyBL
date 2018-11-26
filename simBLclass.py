#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
#from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
#import h5py
#from mayavi import mlab
import numpy as np
import pandas
#import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import scoreatpercentile as percentile
from scipy.stats import linregress
#import math
import gc
#import psutil
import os
from glob import glob
import sys
import traceback
try:
    from astropy.convolution.convolve import convolve_fft
except ImportError:
    print('Warning, cannot load "convolve_fft" from astropy. Using scipy equivlent which uses zero padding.')
    from scipy.signal import fftconvolve
from scipy.signal import argrelextrema
import time
import tarfile

from . import athena_read as ar
from . import helpers
from .helpers import rolling_weighted_triangle_conv as running_mean
from .helpers import grad, mod_grad
from .parmap import parmap


_quiet = False
tau = 2 * np.pi

#mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb,amsmath}"]

_dirs = ['', '~/', '~/Dropbox/dev/pyBL', '/scratch/gpfs/sashaph/BLayer', '/perseus/scratch/gpfs/sashaph/BLayer',
         '~/BLayer', '~/BLayer/fft_tests', '~/archive', '~/data/bl', '~/data/pleiades_data/bl']
_dirs = list(map(os.path.expanduser, _dirs))
_dirs += [os.path.join(d, 'Mach8stampede') for d in _dirs]
_data_base = '/scratch/gpfs/sashaph/BLayer'
_pre = ['BL', 'disk', 'mock']
_i = map(str, range(1,5))
_ext = ['athdf', 'npy']
#_file_fmts = ['BL.out2.%5.5d.athdf', 'disk.out1.%5.5d.athdf']
_int_fmt = '%5.5d'
_file_fmts = ['.'.join([a, 'out' + b, _int_fmt, c]) for a in _pre for b in _i for c in _ext]

def line_plt(p1, p2, **popt):
    if p1[0] == p2[0]:
        dth = np.abs(p1[1] - p2[1])
        nth = int(np.ceil(max(6, (dth / np.pi) * 100)))
        #print(dth, nth)
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

    fig = plt.figure(figsize=(4,8))
    ax = plt.subplot(111)#, projection='polar')

    #ax = plt.subplot(121, projection='polar')
    #ax.plot(ax0_phi, ax0_r, 'k-', lw=1)

    #ax = plt.subplot(122, projection='polar')
    #ax.plot(ax1_th - .5 * np.pi, ax1_r, 'k-', lw=1)
    if 0:
        ax.plot(ax1_r * np.sin(ax1_th), ax1_r * np.cos(ax1_th), 'k-', lw=1)
    else:
        for i in range(ax1_r.size - 1):
            line_plt((ax1_r[i], ax1_th[i]), (ax1_r[i+1], ax1_th[i+1]), c='k', lw=1)
    ax.set_aspect('equal', 'datalim')
    #plt.xlim(0,None)

    if save or fig_fn:
        if fig_fn is None:
            fig_fn = '3d_grid.pdf'
        plt.savefig(fig_fn)
        plt.close()

    #phi = ax0_phi[np.isfinite(ax0_phi)]
    #print("n_phi", 2 * np.pi / phi[phi > 0].min())
    #th = ax1_th[np.isfinite(ax1_th)]
    #print("n_theta", np.pi / (th[th > hpi].min() - hpi))

    return ax1_th, ax1_r

def boxcar(data, n, axis=None):
    csum = data.cumsum(axis=axis)
    tmp = np.roll(csum, n, axis=axis)
    tmp[:,:n,:] = 0
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
    for i in xrange(-n, n+1):
        out = np.logical_and(out, np.roll(base, n, axis=axis))
    return out[loc + (slice(0,data.shape[axis]),)]

def crudeDiff(t, data, axis=0, front=True):
    dt = np.diff(t)[(np.newaxis,) * axis + (slice(None),) + (np.newaxis,) * max(0, (len(data.shape) - axis - 1))]
    out = np.diff(data, axis=axis) / dt
    if front:
        loc = (slice(None),) * axis + ([0],)
        return np.concatenate((out[loc], out), axis=axis)
    else:
        loc = (slice(None),) * axis + ([-1],)
        return np.concatenate((out, out[loc]), axis=axis)

def smooth(data, width=64):
    try:
        len(width)
    except TypeError:
        width = width, 1
    kern = np.ones(width)
    kern /= kern.sum()
    try:
        return convolve_fft(data, kern, boundary='wrap')
    except NameError:
        return fftconvolve(data, kern, mode='same')

_smooth = smooth

def _findAbsPath(fn, sim_path=None):
    if (sim_path is None) and (not os.path.isfile(fn)):
        for d in _dirs:
            tmp = os.path.join(d, fn)
            if os.path.isfile(tmp):
                fn = tmp
                break
    elif sim_path:
        fn = os.path.join(sim_path, fn)
    return fn

def _parse_file(fn, file_handle=None):
    ext = fn.split('.')[-1]
    if ext == 'athdf':
        return BLfile(fn, file_handle=file_handle)
    if ext == 'npy':
        return np.load(fn)[()]
    raise IOError('Cannot identify file type of "{0:}"'.format(fn))

def intr(dr, data, axis=-1):
    if axis == -1:
        axis += len(data.shape)
    loc = [np.newaxis] * axis + [slice(None)]
    return (data * dr).sum(axis=axis)

class BLfileBase(dict):
    def __init__(self, fn, sim_path=None, t=None, trim=True, data=None,
                 defvar=None, ai_data=None, sim=None, file_handle=None, x2_face=None):
        self.t = t
        self.sim = sim
        self._Qtrim = trim
        #if ai_data is None:
        #    ai_data = {}
        self._ai_data = ai_data
        if sim is not None:
            self.mach = sim.mach
        elif ai_data is not None:
            self.mach = 1. / ai_data['hydro']['iso_sound_speed']
        self.fn = _findAbsPath(fn, sim_path)
        if file_handle is None:
            file_handle = fn
        self._file_handle = file_handle
        if data is None:
            self.data = ar.athdf(file_handle, face_func_2=x2_face, return_levels=True)
        else:
            self.data = data
        self.x2_face = x2_face
        for i in self.data.keys():
            if i not in self:
                self[i] = None
            else:
                print('Warning: key {0:} already exists.'.format(i))
        #super(BLfile, self).__init__(self.fn)
        self.path = os.path.split(self.fn)[0]
        #print(data)
        #if data is not None:
        #    self.data.update(data)
        if self.t is None:
            for i in ['Time', 'time', 'T', 't']:
                if i in self.data:
                    self.t = self[i]
                    break
        if self.t is None:
            self.t = int(os.path.split(fn)[1].split('.')[2])
        if not 'Time' in self.data:
            self.data['Time'] = self.t
        self._prefix = '.'.join(os.path.split(fn)[-1].split('.')[:-1])
        #if sim_path and not self.t is None:
        #    self.name = os.path.split(sim_path)[-1] + ' {0:05d}'.format(self.t)
        #    self._prefix = os.path.split(sim_path)[-1] + '_{0:05d}'.format(self.t)
        #else:
        self.name = os.path.split(fn)[-1]
        self.t_str = '%.04g' % self.t
        self.r = self['x1f']
        self.dr = self.r[1:] - self.r[:-1]
        self.phi = self['x2f']
        self.rc = .5 * (self.r[:-1] + self.r[1:])
        self.phic = .5 * (self.phi[:-1] + self.phi[1:])
        self._grid_shape = self.phic.size, self.rc.size
        self._default_var = defvar

    def __repr__(self):
        path, fn = os.path.split(self.fn)
        meh, head = os.path.split(path)
        my_class = repr(self.__class__).split("'")[1].split('.')[-1]
        return '<{0:} {1:}>'.format(my_class, os.path.join(head, fn))

    @property
    def _defvar(self):
        try:
            tmp = self.sim.defvar
            if tmp:
                return tmp
        except AttributeError:
            pass
        for i in filter(None, [self._default_var, 'Rpseudo', 'pseudo', 'dens', 'FT-mag', 'FT-Re']):
            try:
                if not self[i] is None:
                    return i
            except KeyError:
                pass
        for i in self:
            try:
                if self[i].shape == self._grid_shape:
                    return i
            except KeyError:
                pass
        return None

    def _trim(self, data):
        data = np.atleast_1d(data)
        while 1 in data.shape:
            i = data.shape.index(1)
            loc = [slice(None)] * i + [0]
            loc += [slice(None)] * (len(data.shape) - len(loc))
            loc = tuple(loc)
            data = data[loc]
        return data

    def _parse_data(self, data):
        try:
            data.shape
            if self._Qtrim:
                return self._trim(data)
            return data
        except AttributeError:
            return self[data]

    def _special_keys(self, key):
        raise NotImplementedError

    def _parse_self(self, key):
        try:
            out = self._special_keys(key)
            if not out is None:
                return out
        except NotImplementedError:
            pass
        if hasattr(self, key):
            try:
                out = getattr(self, key)()
                if out.shape == self._grid_shape:
                    return out
            except (AttributeError, TypeError):
                pass
        raise KeyError('Unable to parse {0:}'.format(key))

    def __getitem__(self, key):
        try:
            out = super(BLfileBase, self).__getitem__(key)
            if out is None:
                out = self.data[key]
        except KeyError:
            out = self._parse_self(key)
        if self._Qtrim:
            return self._trim(out)
        return out

    def load_all(self):
        return self.data.load_all()

    def intr(self, data, axis=-1):
        data = self._parse_data(data)
        return intr(self.dr, data, axis=axis)

    def ddphi(self, data, axis=0):
        data = self._parse_data(data)
        return (np.roll(data, -1, axis=axis) - np.roll(data, 1, axis=axis)) / (self.phic[2] - self.phic[0])

    def rloc(self, r, subsample=None):
        rc = self.rc[::subsample]
        return np.abs(r - rc).argmin()

class BLfile(BLfileBase):
    def fft(self, data, axis=-2, mag=False):
        try:
            data.shape
        except AttributeError:
            data = self[data]
        out = np.fft.rfft(data, axis=axis)
        if mag:
            out = np.absolute(out)
        return out

    def plot2d(self, data=None, fn=None, save=False, subsample=False, title=None,
               name=None, ext='png', popt=None, cb=True, cbl=None, zerocent=None,
               vmin=None, vmax=None, cmap=None, cbopt=None, fig=None, fopt=None,
               ax=None, log=False, aspect=1, sdir=None, smooth=None,
               phi_shift=0, r_cut=None, phi_dot=0, ret_fn=False, rplot=1):
        '''Plot 2D sim data'''
        if fopt is None:
            fopt = {}
        if popt is None:
            popt = {}
        if cbopt is None:
            cbopt = {}
        r = self.r[np.newaxis, :]
        phi = self.phi[:,np.newaxis] + phi_shift
        if self.t and phi_dot:
            phi -= phi_dot * self.t
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        _popt = {}
        if data is None:
            data = self._defvar
        if hasattr(data, 'lower'):
            if data in ['pseudo', 'Rpseudo'] and r_cut is None:
                r_cut = .85
                if vmax is None and vmin is None:
                    vmin = 'smart'
            if data in ['vorticity', 'vortensity']:
                if r_cut is None:
                    r_cut = .9
                if zerocent is None:
                    zerocent = True
                if vmax is None and vmin is None:
                    vmax = 'smart'
            if data in ['vi', 've']:
                vmin='99%'
            if cbl is None:
                cbl = helpers.labeler(data)
                name = data
        data = self._parse_data(data)
        if type(data) != np.ndarray:
            raise TypeError('Data has type "{:}", not ndarray.'.format(type(data)))
        if not smooth is None:
            data = self.smooth(data, smooth)

        #parse/set options and defaults
        if subsample:
            loc = (slice(None,None,subsample),) * len(x.shape)
            x = x[loc]
            y = y[loc]
            data = data[loc]
        if log:
            _popt['norm'] = mpl.colors.LogNorm()
        # parse smart lim options
        tmp = {}

        try:
            if '%' == vmin[-1]:
                tmp['low'] = float(vmin[:-1])
                vmin = 'smart'
        except TypeError:
            pass
        try:
            if '%' == vmax[-1]:
                tmp['high'] = float(vmax[:-1])
                vmax = 'smart'
        except TypeError:
            pass
        if 'smart' in [vmin, vmax]:
            rloc = slice(None)
            if r_cut:
                if subsample:
                    raise NotImplementedError('r_cut not compatible with subsample')
                rloc = slice(self.rloc(r_cut), None)
            tmp = helpers.smartlim(data[:, rloc], **tmp)
            if vmin == 'smart':
                vmin = tmp[0]
            if vmax == 'smart':
                vmax = tmp[1]
        # check if zero centered data
        if zerocent is None and not log:
            zerocent = helpers.isZeroCent(data)
        if zerocent:
            if cmap is None:
                cmap = helpers.NCcmap
            if vmin is None and vmax is None:
                vmax = np.abs(data).max()
            elif vmin is None:
                vmin = -abs(vmax)
            else:
                vmax = abs(vmin)
            vmin = -vmax

        _popt.update(dict(cmap=cmap, vmin=vmin, vmax=vmax))
        _popt.update(popt)

        if fig is None and ax is None:
            fig = plt.figure(**fopt)
            if name is None:
                name = True
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        if aspect:
            ax.set_aspect(aspect)

        if name:
            if name in [True, 1]:
                name = ''
            if title is None:
                #title = self.name + ' ' + self.t_str + ' ' + name
                title = self.name + ' ' + name
                if self.sim:
                    title = self.sim.name + r' $t/2\pi = {0:g}$'.format(self.t / tau)
            if save and fn is None:
                fn = self._prefix + '_' + name + '_plot.' + ext

        #start plotting
        pcm = plt.pcolormesh(x, y, data, **_popt)
        if rplot:
            rplot = np.atleast_1d(rplot)
            for r in rplot:
                plt.plot(r * np.cos(self.phic), r * np.sin(self.phic), lw=1, c='1', ls=':')
        if title:
            plt.title(helpers.sanitize_lbl(title.format(**self.__dict__)))
        if cb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(pcm, cax=cax, **cbopt)
            cb.ax.yaxis.set_offset_position('left')
            if cbl:
                cb.set_label(cbl)

        plt.sca(ax)
        #save fig
        if save or fn:
            if fn is None:
                fn = self._prefix + '_plot.' + ext
            if not sdir is None:
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                fn = os.path.join(sdir, fn)
            plt.savefig(fn)
            plt.close()
        if ret_fn:
            return fn

        return pcm

    def smooth(self, data, width=64):
        return smooth(self._parse_data(data), width=width)

    def ddr(self, data):
        nm1 = len(data.shape) - 1
        loc = [slice(None)] * nm1
        left = loc + [slice(0,-1)]
        right = loc + [slice(1,None)]
        dr = (self.r[1:] - self.r[:1])[[np.newaxis] * nm1 + [slice(None)]]
        out = np.zeros_like(data)
        d = data[right] - data[left]
        out[left] = .5 * d
        out[right] += .5 * d
        out[loc + [0]] *= 2
        out[loc + [-1]] *= 2
        return out

    def drho_plot(self, ref=None):
        if ref is None:
            ref = self.sim.rho_ref
        drho = self.rhobar() - ref
        plt.plot(self.rc, drho)
        plt.xlabel('$r$')
        plt.ylabel(r'$\left<\rho({:.1f}\times 2\pi)\right>-\left<\rho(0)\right>$'.format(self.t / tau))

class BLaux(BLfile):

    def channel_map(self, var=None, save=False, fn=None, mmax=30, log=True,
                    fig=None, ax=None, aspect=None, fopt={}, popt={}, cbl=None,
                    title=None, vmin=1e-1, vmax=None, cb=True, cbopt={},
                    sdir=None, ext='pdf'):
        if var is None:
            var = self._defvar
        _popt = dict(vmin=vmin, vmax=vmax)
        if fig is None and ax is None:
            fig = plt.figure(**fopt)
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        if aspect:
            ax.set_aspect(aspect)
        if log:
            _popt['norm'] = mpl.colors.LogNorm()
        if title is None:
            title = self.name + ' ' + self.t_str
        _popt.update(popt)

        loc = (slice(None,mmax+1), slice(None))
        ft = (np.abs(self.fft(self._parse_data(var)))**2)[loc]
        one = np.ones((ft.shape[0] + 1, self.r.size))
        mode = (np.arange(ft.shape[0] + 1)[:, np.newaxis] - .5) * one
        r = self.r[np.newaxis, :] * one

        pcm = plt.pcolormesh(mode, r, ft, **_popt)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        if title:
            plt.title(helpers.sanitize_lbl(title))
        plt.xlabel(r'Mode ($m$)')
        plt.ylabel('Radius ($R$)')
        if cb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(pcm, cax=cax, **cbopt)
            cb.ax.yaxis.set_offset_position('left')
            if cbl is None:
                cbl = r'$\left|a_m\right|^2$'
            if cbl:
                cb.set_label(cbl)

        #save fig
        if save or fn:
            if fn is None:
                fn = self._prefix + '_channel_map.' + ext
                if not sdir is None:
                    if not os.path.isdir(sdir):
                        os.mkdir(sdir)
                    fn = os.path.join(sdir, fn)
            plt.savefig(fn)
            plt.close()

    def phase(self, data='pseudo', smooth=None, mod=False):
        try:
            data.shape
        except AttributeError:
            data = self._parse_data(data)
        if not smooth is None:
            data = self.smooth(data, smooth)
        modes = np.abs(self.fft(data)).argmax(axis=0)
        sqr = data**2
        loc = sqr.argmax(axis=0)
        phase = self.phic[loc]
        phase -= .5 * (1 - np.sign(data[loc,np.arange(data.shape[1])])) * np.pi
        phase %= tau
        if mod:
            phase %= tau / modes
        return phase

    def phase_plot(self, data='pseudo', fn=None, save=False, ext='pdf',
                   sdir=None, fig=None, ax=None, fopt={}, smooth=None, mod=False):
        data = self._parse_data(data)
        if len(data.shape) > 1:
            data = self.phase(data, smooth=smooth)
        if fig is None and ax is None:
            fig = plt.figure(**fopt)
        plt.plot(self.rc, data / np.pi)
        plt.ylim(0, 2)
        plt.xlabel('Radius')
        plt.ylabel(r'Phase$/\pi$')

        #save fig
        if save or fn:
            if fn is None:
                fn = self._prefix + '_phase_plot.' + ext
            if not sdir is None:
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                fn = os.path.join(sdir, fn)
            plt.savefig(fn)
            plt.close()

    def mode_phase(self, mmax=30):
        ft = self.fft('pseudo')[:mmax+1,:]
        phi = -np.angle(ft)
        a = np.abs(ft)
        #tmp = np.ma.array(a, mask=a - .5 * a.max(axis=0)[np.newaxis, :] > 0)
        #m = tmp.mean(axis=0)
        #s = tmp.std(axis=0)
        #loc = np.where(a - (m + 3 * s)[np.newaxis, :] > 0)
        #modes = np.array(sorted(set(loc[0])))
        #md = {modes[i]: i for i in range(len(modes))}
        #amp = np.ones(a.shape) * np.nan
        #ang = amp.copy()
        #amp[loc] = a[loc]
        #ang[loc] = phi[loc]
        #return amp, ang
        return a, phi

    def main_plots(self, save=True, ext='png', sdir=None):
        try:
            self.channel_map(save=save, ext=ext, sdir=os.path.join(sdir, 'channel_maps'))
        except KeyboardInterrupt:
            raise
        except:
            print('Unable to plot channel map for {0:}.'.format(self.name))
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print('-' * 60)
        for var in ['pseudo']:
            try:
                self.plot2d(var, save=save, ext=ext, sdir=os.path.join(sdir, '2d_plots'))
            except KeyboardInterrupt:
                raise
            except:
                print('Unable to plot {1:} for {0:}.'.format(self.name, var))
                print('-' * 60)
                traceback.print_exc(file=sys.stdout)
                print('-' * 60)


class BL3Dfile(BLfile):
    def curl(self, data):
        if hasattr(data, 'lower'):
            data = self[data+'1'], self[data+'2']
        x, y = data
        y *= self.rc[np.newaxis,:]
        return (grad(self.rc, y, axis=1) - self.ddphi(x)) / self.rc[np.newaxis,:]

    def draw_spiral(self, rp, phi0=0, opt=None):
        if opt is None:
            opt = {}
        if 'ls' not in opt:
            opt['ls'] = ':'
        if 'lw' not in opt:
            opt['lw'] = 1
        if 'c' not in opt:
            opt['c'] = '1'
        r = np.array([i for i in self.rc if i >= rp])
        phi = spiral(r, rp, 1./self.mach) + phi0
        print(phi)
        plt.plot(r * np.cos(phi), r * np.sin(phi), **opt)

class BLConsPrim(BL3Dfile):
    def rhobar(self):
        return self['dens'].mean(axis=0)

    def Mdot(self):
        return self['dens']*self['vel1']

    def v1v2(self):
        return self['vel1']*self['vel2']

    def vorticity(self, dvphi=False):
        if dvphi:
            out = self['vel2']
            out -=  self.rhoWeight(out)[np.newaxis,:]
            out = self.curl((self['vel1'], out))
        else:
            out = self.curl('vel')
        #out[:,0] = out[:, 1]
        #out[:,-1] = out[:,-2]
        return out

    def vortensity(self, dvphi=False):
        return self.vorticity(dvphi=dvphi) / self['dens']

    def vi(self):
        return self.rc[np.newaxis,:]**2 * self.vorticity(True)

    def ve(self):
        return self.rc[np.newaxis,:]**2 * self.vortensity(True)

    def rhoWeight(self, data):
        data = self._parse_data(data)
        dens = self['dens']
        return (dens * data).sum(axis=0) / dens.sum(axis=0)

    def __dv2(self):
        out = self['vel2']
        return out - out.mean(axis=0)[np.newaxis, :]

    def deltaVal(self, data):
        data = self._parse_data(data)
        return data - self.rhoWeight(data)

    def Rstress(self):
        return self['mom1'] * self.deltaVal('vel2')

    def CL(self):
        return self.rc**2 * (self['mom1'] * self['vel2']).mean(axis=0) * tau

    #BRS12 ApJ 760:22
    def CL20(self):
        return tau * self.rc**2 * self['dens'].mean(axis=0) * self.rhoWeight(self.deltaVal('vel1') * self.deltaVal('vel2'))

    def CL24(self):
        return tau * self.rc**2 * self['dens'].mean(axis=0) * self.rhoWeight(np.abs(self.deltaVal('vel1') * self.deltaVal('vel2')))

    def CL25(self, Op, M=None):
        if M is None:
            M = self.mach
        s = 1. / M
        dens = self['dens']
        Mdens = dens.mean(axis=0)
        return self.rc * s**3 * tau * np.mean((dens - Mdens[np.newaxis, :])**2, axis=0) / (dens.mean(axis=0) * (self.Ok() - Op))

    def CS(self):
        dens = self['dens']
        Mdens = dens.mean(axis=0)
        dv2 = self['vel2'] - self.Oloc() * self.rc
        return tau * self.rc**2 * Mdens * self.rhoWeight(dv2 * self['vel1'])

    def CA(self):
        dens = self['dens']
        Mdens = dens.mean(axis=0)
        return tau * self.rc**3 * Mdens * self.Oloc() * self.rhoWeight('vel1')

    def CLplot(self, Op=None, M=None, nm=6):
        cl = [self.CL(),self.CL24()]
        lbls = ['BRS12 Eqn %d' % i for i in [19,24]]
        if not Op is None:
            cl.append(self.CL25(Op, M))
        cl += [self.CS(), self.CA()]
        lbls += ['BRS13 $C_S$','BRS13 $C_A$']
        tmp = self.CSm()
        #s = tmp.sum(axis=0)
        #loc = sorted(range(tmp.shape[0] // 2), key=lambda x: 1/s[x])
        norm = self.intr(np.abs(tmp))
        loc = sorted(range(norm.shape[0]), key=lambda x: -norm[x])
        for i in loc[:nm]:
            cl.append(tmp[i])
            lbls.append('by mode {0:}'.format(i))
        for i in cl:
            plt.plot(self.rc, i)
        plt.plot(self.rc, tmp.sum(axis=0), 'k:')
        lbls.append('sum')
        plt.axhline(0, ls=':', lw=1, c='.5')
        plt.legend(lbls)
        plt.xlabel('R')
        plt.ylabel('$C_L$')

    def Ok(self):
        return self.rc**-1.5

    def Oloc(self):
        return self.rhoWeight('vel2') / self.rc

    def pseudo(self):
        err = 'Must be implemented in subclase of {0:}.'.format(self.__class__)
        raise NotImplementedError(err)

    def Rpseudo(self):
        return self.rc[np.newaxis, :] * self.pseudo()

    def CSm(self, delta=False):# dvm=None, dum=None, dSm=None, o=None):
        rho = self['dens']
        S = rho.mean(axis=0)[np.newaxis,:]
        if delta:
            norm = 1. / (self.phic.size - 1)
            dvm = self.fft(self.deltaVal('vel1')) * norm
            dum = self.fft(self.deltaVal('vel2')) * norm
        else:
            dvm = self.fft('vel1') / self.phic.size
            dum = self.fft('vel2') / self.phic.size
        r = self.rc[np.newaxis, :]
        return .5 * np.pi * r**2 * S * (np.conj(dvm) * dum + dvm * np.conj(dum))

    def CSmPlot(self, nm=5, norm=1):
        csm = np.real(self.CSm()) * norm
        csm[0] = 0
        cs = self.CS()
        norm = self.intr(np.abs(csm))
        modes = sorted(range(norm.shape[0]), key=lambda x: -norm[x])
        plt.figure()
        plt.plot(self.rc, cs, 'k-', label='$C_S$')
        for m in modes[:nm]:
            plt.plot(self.rc, csm[m], label=str(m))
        plt.plot(self.rc, csm[1:].sum(axis=0), c='.5', ls=':', label='sum')
        plt.legend()
        plt.axhline(0, c='.5', ls=':', lw=1)
        plt.xlim(self.r[0], self.r[-1])
        plt.xlabel('R')
        plt.ylabel('$C_S$')
        title = ''
        try:
            title = self.sim.name + ' '
        except:
            pass
        title += 't=$%g' % (self.t / tau) + r'\times 2 \pi$'
        plt.title(title)

    def kr(self, dSm):
        return self.ddr(dSm) / dSm

    def hs(self, S0):
        return -S0 / self.ddr(S0)

    def FFT_errors(self, ft_file, var='pseudo', vmin='smart', vmax='max'):
        afft = ft_file['FT-' + var]
        if len(var) == 2:
            if var[0] == 'v':
                var = 'vel' + var[-1]
        pfft = self.fft(var)[:afft.shape[0]]
        amp = np.abs(afft)
        n = self.rc.size * self.phic.size
        norm = np.median(np.abs(pfft) / amp)
        if norm > np.sqrt(n):
            norm = n
        elif int(norm + .5) == 1:
            norm = 1
        afft *= norm
        amp *= norm
        print("norm:", norm, "post-norm:", np.median(np.abs(pfft)/np.abs(afft)))
        dmag = (np.abs(pfft) - amp) / np.abs(pfft)
        dth = (np.angle(pfft) + np.angle(afft))# % tau

        fig = plt.figure(figsize=(8,8))
        ax0 = plt.subplot(221)
        data = abs(dmag)
        avmax = vmax
        avmin = vmin
        if 'smart' in [vmin, vmax]:
            tmp = helpers.smartlim(data)
        if vmin == 'smart':
            avmin = tmp[0]
        else:
            avmin = vmin
        if vmax == 'smart':
            avmax = tmp[1]
        elif vmax == 'max':
            avmax = data.max()
        else:
            avmax = vmax

        im0 = plt.imshow(data, vmin=avmin, vmax=avmax, interpolation='nearest', norm=mpl.colors.LogNorm())
        cb0 = plt.colorbar(im0, ax=ax0)
        plt.xlabel('r index')
        plt.ylabel('mode')
        cb0.set_label(r'$\left|\Delta|{\rm FFT}|/|{\rm FFT})|\right|$')

        ax1 = plt.subplot(222)
        data1 = abs(dth)
        bvmax = vmax
        bvmin = vmin
        if 'smart' in [vmin, vmax]:
            tmp = helpers.smartlim(data1)
        if vmin == 'smart':
            bvmin = tmp[0]
        else:
            bvmin = vmin
        if vmax == 'smart':
            bvmax = tmp[1]
        elif vmax == 'max':
            bvmax = data1.max()
        else:
            bvmax = vmax

        im1 = plt.imshow(data1, vmin=bvmin, vmax=bvmax, interpolation='nearest', norm=mpl.colors.LogNorm())
        cb1 = plt.colorbar(im1, ax=ax1)
        plt.xlabel('r index')
        plt.ylabel('mode')
        cb1.set_label(r'$\left|\Delta\theta\right|$')

        plt.subplot(223)
        data = amp
        # opt2['vmin'] = helpers.smartlim(data[loc])[0]
        # opt2['vmax'] = data[loc].max()
        im2 = plt.imshow(data, interpolation='nearest', norm=mpl.colors.LogNorm(),
                         vmin=helpers.smartlim(data)[0],
                         vmax=data.max())  # , vmin=helpers.smartlim(data[loc])[0], **opt2)
        cb2 = plt.colorbar(im2)
        plt.xlabel('r index')
        plt.ylabel('mode')
        cb2.set_label(r'$\left|{\rm FFT}\right|$')

        ax = plt.subplot(224)
        self.plot2d(var, ax=ax)

class BLcons(BLConsPrim):
    def _special_keys(self, key):
        if key[:3] == 'vel' and len(key) == 4:
            return self.vel(key[3])
        if key == 'dens**2':
            return self['dens']**2
        return None

    def vel(self, i):
        return self['mom{0:}'.format(i)] / self['dens']

    def pseudo(self):
        return self['mom1'] / np.sqrt(self['dens'])

class BLprim(BLConsPrim):
    def _special_keys(self, key):
        if key[:3] == 'mom' and len(key) == 4:
            return self.mom(key[3])
        if key == 'dens**2':
            return self['dens']**2
        return None

    def mom(self, i):
        return self['vel{0:}'.format(i)] * self['dens']

    def pseudo(self):
        return self['vel1'] * np.sqrt(self['dens'])

class BLFT(BLfile):
    def rhobar(self):
        return self['FT-dens'][0]

    def __init__(self, *args, **kwargs):
        super(BLFT, self).__init__(*args, **kwargs)
        #super().__init__(*args, **kwargs)
        #self.data = {}
        #self.data.update(self)

    def _trim(self, data):
        data = super(BLFT, self)._trim(data)
        #data = super()._trim(data)
        if len(data.shape) == 2:
            a = data.min(axis=1)
            b = data.max(axis=1)
            i = a.size - 1
            if 0:
                while a[i] == 0 and b[i] == 0 and i > 0:
                    i -= 1
                try:
                    i = max(i, self._ai_data['meshblock']['nx2'] - 1)
                except KeyError:
                    pass
            else:
                try:
                    i = self._ai_data['meshblock']['nx2'] - 1
                except TypeError:
                    i = 31
            data = data[slice(0, i + 1)].copy()
        return data

    def _special_keys(self, key):
        if key == 'FT':
            if 'FT-Re' in self.data:
              return (self['FT-Re'] - 1j * self['FT-Im']).astype('complex64')
        if key + '-Re' in self.data and key + '-Im' in self.data:
            return (self[key + '-Re'] - 1j * self[key + '-Im']).astype('complex64')
        if key == 'FT':
            if 'FT-pseudo-Re' in self.data:
                key = 'FT-pseudo'
            return (self[key + '-Re'] - 1j * self[key + '-Im']).astype('complex64')
        if key in ['mag', 'amp']:
            return np.abs(self['FT'])
        if key == 'angle':
            return np.angle(self['FT'])
        if 'FT-' + key + '-Re' in self.data:
            return self._special_keys('FT-' + key)
        return None

    def fluxes(self):
        u = self['FT-vel2']
        r2 = tau*(self.rc)[np.newaxis, :]**2
        out = {'CL': r2 * self['FT-CL'],
               'CA': r2 * u[0][np.newaxis, :] * self['FT-Mdot']}
        for i in ['Mdot', 'dens']:
            out[i] = np.real(self['FT-' + i][0])
        out['CS'] = out['CL'] - out['CA']
        v = self['FT-vel1']
        out['CSm'] = r2 * np.real(self['FT-dens'][0])[np.newaxis, :] * (np.conj(v) * u + np.conj(u) * v)
        out['drho'] = np.real(self['FT-dens'][0]) - self.sim.rho_ref
        out['vphi'] = np.real(u[0])
        out['vr'] = np.real(v[0])
        return out


############################
# End of BLfile subclasses #
############################

def loadBLfile(fn, **kwargs):
    fn = _findAbsPath(fn, kwargs.get('sim_path', None))
    data = _parse_file(fn, file_handle=kwargs.get('file_handle', None))
    ai_fn = kwargs.pop('athinput_fn', None)
    ai_data = kwargs.pop('ai_data', None)
    if ai_fn is None:
        path = kwargs.get('sim_path', '')
        tmp = glob(os.path.join('athinput.*'))
        if len(tmp) == 1:
            ai_fn = tmp[0]
    if ai_fn and ai_data is None:
        ai_data = ar.athinput(ai_fn)
    kind = None
    if not ai_data is None:
        kwargs['ai_data'] = ai_data
        tmp = os.path.split(fn)[-1].split('.')
        #print(tmp)
        if tmp[0] == ai_data['job']['problem_id']:
            outs = [i for i in ai_data.keys() if i[:6] == 'output']
            for out in outs:
                if ai_data[out].get('id', 'out' + out[6:]) == tmp[1]:
                    kind = ai_data[out].get('variable', None)
                    break
    else:
        if 'vel1' in data:
            kind = 'prim'
        if 'mom1' in data:
            kind = 'cons'
        if 'FT-Re' in data and 'FT-Im' in data:
            kind = 'FT'
    if kind == 'prim':
        return BLprim(fn, data=data, **kwargs)
    if kind == 'cons':
        return BLcons(fn, data=data, **kwargs)
    if kind in ['FT', 'FT-Range']:
        return BLFT(fn, data=data, **kwargs)
    #raise RuntimeError
    return BLfile(fn, data=data, **kwargs)

class _FileBuffer(object):
    def __init__(self, filenames, ghost=1, sim=None, var='FT'):
        self.filenames = filenames
        self.len = len(filenames)
        self.size = 2 * ghost + 1
        self.sim = sim
        self.ghost = ghost
        self.index = None
        self._var = var
        self._phase = None
        self._amp = None
        self._speed = None
        self._t = None
        self._offset = range(-ghost, ghost + 1)
        self.set_index(0)

    def _load(self, f):
        try:
            if int(f) == f:
                f = self.filenames[f]
        except ValueError:
            pass
        #print(f)
        bf = BLFT(f, sim=self.sim)
        return [bf.t, np.abs(bf[self._var]), np.angle(bf[self._var])]

    def _cap_index(self, i):
        return min(max(i, 0), self.len - 1)

    def set_index(self, i):
        i %= self.len
        self._t, self._amp, self._phase = zip(*[self._load(self._cap_index(i + j)) for j in self._offset])
        self._t = list(self._t)
        self._amp = list(self._amp)
        self._phase = list(self._phase)
        self._speed = [None] * self.size
        self._comp()
        self.index = i

    def _set(self, i):
        i %= self.len
        i0 = i - self.ghost
        tmp = self._load(self._cap_index(i0))
        self._t = [tmp[0]] * self.size
        self._amp = [tmp[1]] * self.size
        self._phase = [tmp[2]] * self.size
        self.index = i0
        while self.index < i:
            self._increment()

    def _comp(self, shift=False):
        # unwrap
        phi = np.array(self._phase)
        # coarse correct
        dphi = np.diff(phi, axis=0)
        tmp = np.pad(dphi, ((1, 0), (0, 0), (0, 0)), 'constant')
        tmp = np.minimum(np.floor(tmp / tau), 0)
        phi -= tmp.cumsum(axis=0) * tau
        del tmp
        # fine correct
        dphi = np.diff(phi, axis=0)
        t = np.array(self._t)
        dt = np.diff(t)
        fdphi = np.pad(dphi, ((0, 1), (0, 0), (0, 0)), 'constant')
        bdphi = np.pad(dphi, ((1, 0), (0, 0), (0, 0)), 'constant')
        fdt = np.pad(dt, (0, 1), 'constant')[:, np.newaxis, np.newaxis]
        bdt = np.pad(dt, (1, 0), 'constant')[:, np.newaxis, np.newaxis]
        shift = np.rint(np.maximum(fdphi * bdt / fdt - bdphi, 0) / tau)
        bshift = np.rint(np.roll(np.maximum(bdphi * fdt / bdt - fdphi, 0) / tau, 1, axis=0))
        shift[shift != bshift] = 0
        shift *= tau
        phi += shift.cumsum(axis=0)
        self._phase = list(phi)

        # speed
        self._speed = list(grad(t, phi))

    def _increment(self):
        if self.index >= self.len - 1:
            raise IndexError('Cannot increment past last time-step.')
        tmp = self._load(self.index + self.ghost + 1)
        self._t.append(tmp.pop(0))
        self._amp.append(tmp.pop(0))
        self._phase.append(tmp.pop(0))
        # coarse unwrap
        fdphi = self._phase[-1] - self._phase[-2]
        self._phase[-1] -= tau * np.minimum(np.floor(fdphi / tau), 0)
        # fine unwrap
        fdphi = self._phase[-1] - self._phase[-2]
        bdphi = self._phase[-2] - self._phase[-3]
        fdt = self._t[-1] - self._t[-2]
        bdt = self._t[-2] - self._t[-3]
        shift = np.rint(np.maximum(fdphi * bdt / fdt - bdphi, 0) / tau)
        bshift = np.rint(np.roll(np.maximum(bdphi * fdt / bdt - fdphi, 0) / tau, 1, axis=0))
        shift[shift != bshift] = 0
        shift *= tau
        self._phase[-2] += shift
        self._phase[-1] += shift

        # speed
        self._speed.append(None)
        dl = self._t[-2] - self._t[-3]
        dr = self._t[-1] - self._t[-2]
        Dinv = 1. / (dl + dr)
        rat = dr / dl
        self._speed[-2] =   (dl**-1 - dr**-1) * self._phase[-2] \
                          + Dinv / rat * self._phase[-1] \
                          - rat * Dinv * self._phase[-3]

        # update index and arrays
        self._phase.pop(0)
        self._t.pop(0)
        self._amp.pop(0)
        self._speed.pop(0)
        self.index += 1

    def increment(self):
        try:
            self._increment()
            return True
        except IndexError:
            return False

    @property
    def t(self):
        return self._t[self.ghost]

    @property
    def amp(self):
        return self._amp[self.ghost]

    @property
    def phase(self):
        return self._phase[self.ghost]

    @property
    def speed(self):
        return self._speed[self.ghost]

    @property
    def state(self):
        return (self.t, self.amp, self.phase, self.speed)

    def mean(self, data):
        return np.array([i for i in data if i is not None]).mean(axis=0)

    def std(self, data):
        return np.std(np.array([i for i in data if i is not None]), axis=0)


class IncrementalFFT(object):
    def __init__(self, filenames, sim=None, store_data=False, fine_out=None, coarse_out=None, var='FT', quiet=None):
        self.filenames = filenames
        self._ghost = 10
        self.buffer = _FileBuffer(filenames, ghost=self._ghost, sim=sim, var=var)
        self.sim = sim
        self._store_data = store_data
        if fine_out is None:
            fine_out = os.path.join(sim.path, 'FFT_fine_' + var + '.npy')
        if coarse_out is None:
            coarse_out = os.path.join(sim.path, 'FFT_coarse_' + var + '.npy')
        self.fine_out = fine_out
        self.coarse_out = coarse_out
        self._t = None
        self._amp = None
        self._phase = None
        self._speed = None
        if quiet is None:
            quiet = _quiet
        self.quiet = quiet
        self._cd = None

    def process(self, store_data=None):
        mode = 'wb'
        _ft = -1
        _ct = -1
        if os.path.isfile(self.fine_out):
            _ft = self.get_last_time(self.fine_out)
            mode = 'ab'
        if os.path.isfile(self.coarse_out):
            _ct = self.get_last_time(self.coarse_out)
            mode = 'ab'
        if store_data is None:
            store_data = self._store_data
        if store_data:
            self._t = []
            self._amp = []
            self._phase = []
            self._speed = []
            self._cd = {'t': [], 'amp': [], 'phase': [], 'speed': [], 'amp_std': [], 'phase_std': [], 'speed_std': []}
        test = True
        if not self.quiet:
            #print(self.coarse_out)
            print('Compiling FFT data.')
            helpers.update_progress(0)
        with open(self.fine_out, mode) as fine, open(self.coarse_out, mode) as coarse:
            while test:
                data = self.buffer.state
                if store_data:
                    self._t.append(data[0])
                    self._amp.append(data[1])
                    self._phase.append(data[2])
                    self._speed.append(data[3])
                if data[0] > _ft:
                    for i in data:
                        np.array(i).astype('float32').tofile(fine)
                name = os.path.split(self.buffer.filenames[self.buffer.index])[-1].split('.')
                if data[0] > _ct:
                    if name[1] == 'FT' and name[2][-1] == '0':
                        #print(self.buffer.index, data[0], self.buffer.filenames[self.buffer.index])
                        np.array(data[0]).astype('float32').tofile(coarse)
                        a = self.buffer._amp
                        if store_data:
                            self._cd['t'].append(data[0])
                            self._cd['amp'].append(self.buffer.mean(a))
                            self._cd['amp_std'].append(self.buffer.std(a))
                        self.buffer.mean(a).astype('float32').tofile(coarse)
                        self.buffer.std(a).astype('float32').tofile(coarse)
                        a = self.buffer._phase
                        if store_data:
                            self._cd['phase'].append(self.buffer.mean(a))
                            self._cd['phase_std'].append(self.buffer.std(a))
                        self.buffer.mean(a).astype('float32').tofile(coarse)
                        self.buffer.std(a).astype('float32').tofile(coarse)
                        a = self.buffer._speed
                        if store_data:
                            self._cd['speed'].append(self.buffer.mean(a))
                            self._cd['speed_std'].append(self.buffer.std(a))
                        self.buffer.mean(a).astype('float32').tofile(coarse)
                        self.buffer.std(a).astype('float32').tofile(coarse)
                test = self.buffer.increment()
                if not self.quiet:
                    helpers.update_progress(float(self.buffer.index) / self.buffer.len)
        helpers.update_progress(1)
        if store_data:
            for i in self._cd:
                self._cd[i] = np.array(i)

    def get_last_time(self, fn=None, nvar=None, nphi=None):
        if fn is None:
            fn = self.coarse_out
        if nvar is None:
            if fn == self.coarse_out:
                nvar = 6
            elif fn == self.fine_out:
                nvar = 3
            else:
                raise RuntimeError('"navr" cannot be determined.')
        first = BLFT(self.filenames[0], sim=self.sim)
        if nphi is None:
            nphi = self.sim.inputs['meshblock']['nx2']
        nr = first.rc.size
        del first
        with open(fn, 'r') as f:
            offset = - 4 * (nvar * nphi * nr + 1)
            f.seek(offset, os.SEEK_END)
            t = np.fromfile(f, 'float32', 1)[0]
        return t


class FTdataFile(object):
    def __init__(self, filename, nr=None, nphi=None, sim=None, coarse=True):
        if nr is None:
            nr = sim.rc.size
        if nphi is None:
            nphi = sim.inputs['meshblock']['nx2']
        self.filename = filename
        #print(filename)
        self.nr = nr
        self.nphi = nphi
        self.modes = np.arange(nphi)[np.newaxis,:,np.newaxis]
        self.sim = sim
        self.coarse = coarse
        self._t = None
        self._amp = None
        self._phase = None
        self._speed = None
        self._amp_std = None
        self._phase_std = None
        self._speed_std = None

    def existsQ(self):
        return os.path.isfile(self.filename)

    def updateQ(self):
        if not self.existsQ():
            return True
        if os.path.getsize(self.filename) == 0:
            return True
        if self.sim is not None:
            tmp = [0]
            files = [i for i in glob(os.path.join(self.sim.path, '*.athdf'))]
            tmp.extend([os.path.getmtime(i) for i in files])
            if max(tmp) > os.path.getmtime(self.filename):
                return True
        return False

    def generate(self):
        self.sim.gen_fft_file()

    def _read_data(self):
        if self.updateQ():
            self.generate()
        t = []
        amp = []
        phase = []
        speed = []
        nvar = 3
        if self.coarse:
            amp_std = []
            phase_std = []
            speed_std = []
            nvar += 3
        with open(self.filename, 'rb') as f:
            while True:
                try:
                    t.append(np.fromfile(f, 'float32', 1)[0])
                    data = list(np.reshape(np.fromfile(f, 'float32', nvar * self.nphi * self.nr), (nvar, self.nphi, self.nr)))
                    amp.append(data.pop(0))
                    if self.coarse: amp_std.append(data.pop(0))
                    phase.append(data.pop(0))
                    if self.coarse: phase_std.append(data.pop(0))
                    speed.append(data.pop(0))
                    if self.coarse: speed_std.append(data.pop(0))
                except (EOFError, IndexError, ValueError):
                    break
        self._t = np.array(t)
        self._amp = np.array(amp)
        self._phase = np.array(phase)
        self._speed = np.array(speed) / self.modes
        if self.coarse:
            self._amp_std = np.array(amp_std)
            self._phase_std = np.array(phase_std)
            self._speed_std = np.array(speed_std) / self.modes

    @property
    def t(self):
        if self._t is None:
            self._read_data()
        return self._t

    @property
    def amp(self):
        if self._amp is None:
            self._read_data()
        return self._amp

    @property
    def phase(self):
        if self._phase is None:
            self._read_data()
        return self._phase

    @property
    def speed(self):
        if self._speed is None:
            self._read_data()
        return self._speed

    @property
    def FT(self):
        return self.amp * np.exp(1j * self.phase)


class FFTset(object):
    def __init__(self, filenames, sim=None, athinput=None, fft_data=None,
                 fft_time=None):
        self.filenames = filenames
        self.sim = sim
        self._fft_data = fft_data
        self._fft_time = fft_time

    def _load_fft_data(self):
        data = []
        t = []
        for fn in self.filenames:
            if self.sim is None:
                f = BLFT(fn)
            else:
                f = self.sim.loadfile(fn)
            data.append(f['FT'])
            t.append(f.t)
        data = np.array(data)
        t = np.array(t)
        if self._fft_data is None:
            self._fft_data = data
        if self._fft_time is None:
            self._fft_time = t
        return data

    @property
    def data(self):
        if self._fft_data is None:
            return self._load_fft_data()
        return self._fft_data

    @property
    def time(self):
        if self._fft_time is None:
            self._load_fft_data()
        return self._fft_time

class CompositeFFTSet(object):
    def __init__(self, fftsets, fft_data=None, fft_time=None, phase_angle=None, phase_speed=None):
        self.fftsets = fftsets
        self._fft_time = fft_time
        self._fft_data = fft_data
        self._phase_angle = phase_angle
        self._phase_speed = phase_speed

    @property
    def data(self):
        if self._fft_data is None:
            return self._collect_fft_data()
        return self._fft_data

    @property
    def time(self):
        if self._fft_time is None:
            self._collect_fft_data()
        return self._fft_time

    def _collect_fft_data(self):
        ffts = sorted(self.fftsets, key=lambda x:x.time[1])
        nt = sum([i.time.size for i in ffts])
        shape = (nt,) + ffts[0].data.shape[1:]
        data = np.empty(shape, dtype='complex64')
        time = np.empty(nt)
        n = len(ffts)
        for i in range(n):
            data[i::n] = ffts[i].data
            time[i::n] = ffts[i].time
        while time[1] == 0:
            data = data[1:]
            time = time[1:]
        phase = self._unwrap(time, np.angle(data))
        if self._fft_data is None:
            self._fft_data = data
        if self._fft_time is None:
            self._fft_time = time
        if self._phase_angle is None:
            self._phase_angle = phase
        return time, data, phase

    def _unwrap(self, time=None, phase=None, limit=None, mNorm=False, fine_correct=False):
        if phase is None:
            phase = np.angle(self.data)
        else:
            phase = phase.copy()
        if time is None:
            time = self.time
        if limit is None:
            limit = 0
            # limit = - .1 / np.arange(phase.shape[1])[np.newaxis,:,np.newaxis]
        if mNorm:
            limit /= np.arange(phase.shape[1])[np.newaxis,:,np.newaxis]
        # coarse unwrap
        d = crudeDiff(time, phase, axis=0)
        shift = np.zeros_like(phase)
        shift[np.where(d < limit)] += tau
        #shift = np.roll(shift, -1, axis=0)
        #shift[-1] = 0
        self._shift = shift.copy()
        phase += shift.cumsum(axis=0)
        # fine-course unwrap
        if fine_correct:
            dphi = np.diff(phase, axis=0)
            fdphi = np.pad(dphi, ((0, 1), (0, 0), (0, 0)), 'constant')
            bdphi = np.pad(dphi, ((1, 0), (0, 0), (0, 0)), 'constant')
            dt = np.diff(time)
            dtmax = dt.max()
            fdt = np.pad(dt, (0, 1), 'constant')[:,np.newaxis,np.newaxis]
            bdt = np.pad(dt, (1, 0), 'constant')[:, np.newaxis, np.newaxis]
            ashift = np.maximum(fdphi * bdt / fdt - bdphi, 0) / tau
            bshift = np.maximum(bdphi * fdt / bdt - fdphi, 0) / tau
            shift = np.maximum(fdphi * bdt / fdt - bdphi + bdphi * fdt / bdt - fdphi, 0) * .5 / tau
            shift = np.rint(shift) * tau
            phase += shift.cumsum(axis=0)
        return phase

    def _fine_correct(self, time=None, phase=None):
        if phase is None:
            phase = self._unwrap()
        if time is None:
            time = self.time
        dphi = np.diff(phase, axis=0)
        fdphi = np.pad(dphi, ((0, 1), (0, 0), (0, 0)), 'constant')
        bdphi = np.pad(dphi, ((1, 0), (0, 0), (0, 0)), 'constant')
        dt = np.diff(time)
        dtmax = dt.max()
        fdt = np.pad(dt, (0, 1), 'constant')[:,np.newaxis,np.newaxis]
        bdt = np.pad(dt, (1, 0), 'constant')[:, np.newaxis, np.newaxis]
        shift = np.rint(np.maximum(fdphi * bdt / fdt - bdphi, 0) / tau)
        bshift = np.rint(np.roll(np.maximum(bdphi * fdt / bdt - fdphi, 0) / tau, 1, axis=0))
        shift[shift != bshift] = 0
        #shift = np.minimum(shift * tau, np.maximum(dtmax - fdphi, 0))
        shift *= tau
        self._fshift = shift.copy()
        return phase + shift.cumsum(axis=0)

    def phase(self):
        if self._phase_angle is None:
            self._phase_angle = self._unwrap()
        return self._phase_angle

    def prop_speed(self, phase=None):
        if self._phase_speed is None:
            if phase is None:
                phase = self._unwrap()
            m = np.arange(phase.shape[1])
            m = m[np.newaxis,:,np.newaxis]
            dphi = grad(self.time, phase)
            self._phase_speed = dphi / m
        return self._phase_speed


class BLsim(object):
    def __init__(self, path, fmts=None, coarse_data=None, fft_time=None,
                 athinput=None, mode_mask=None, main_modes=None,
                 phase_angle=None, rho_ref=None, mode_detect=None):
        if fmts is None:
            fmts = _file_fmts
        self._fmts = fmts
        self.name = os.path.split(os.path.abspath(path))[-1]
        path = os.path.expanduser(path)
        if path == self.name and not os.path.isdir(path):
            for d in _dirs:
                tmp = os.path.join(d, path)
                if os.path.isdir(tmp):
                    path = tmp
                    break
        if not os.path.isdir(path):
            raise IOError('Simulation directory "{0:}" not found.'.format(path))
        self.path = path
        if athinput is None:
            qry = os.path.join(path, 'athinput.*')
            tmp = glob(qry)
            if len(tmp) == 1:
                athinput = tmp[0]
            else:
                if 'athinput.bl' in tmp:
                    athinput = 'athinput.bl'
                else:
                    athinput = sorted(tmp, key=os.path.getmtime)[-1]
                print('Multiple athena inputs detected. Using "{0:}".'.format(athinput))
        if os.path.isfile(athinput):
            self.inputs = ar.athinput(athinput)
        elif os.path.isfile(os.path.join(self.path, athinput)):
            self.inputs = ar.athinput(os.path.join(self.path, athinput))
        else:
            self.inputs = {}
        self.fileDict = {}
        self.varDict = {}
        self.idDict = {}
        axes = []
        if not self.inputs:
            searches = [fmt.split('%')[0] + '*.' + fmt.split('d.')[-1] for fmt in fmts]
            raise NotImplementedError('Currently needs athinput.')
        else:
            self.mach = 1. / self.inputs['hydro']['iso_sound_speed']
            mesh = self.inputs['mesh']
            for i in [1, 2]:
                x = 'x' + str(i)
                nx = mesh['n' + x]
                Dx = mesh[x + 'max'] - mesh[x + 'min']
                if x + 'rat' in mesh:
                    rat = mesh[x + 'rat']
                    dx0 = (rat - 1.) / (rat**nx - 1.) * Dx
                    xf = np.ones(nx + 1) * mesh[x + 'min']
                    xf[1:] += (rat**np.arange(nx) * dx0).cumsum()
                else:
                    xf = (np.arange(nx + 1) * Dx / nx) + mesh[x + 'min']
                axes.append(xf)
            #raise RuntimeError
            self.r = axes[0].copy()
            self.dr = self.r[1:] - self.r[:-1]
            self.rc = .5 * self.r[1:] + .5 * self.r[:-1]
            self.phi = axes[1].copy()
            self.phic = .5 * self.phi[1:] + .5 * self.phi[:-1]
            outs = [i for i in self.inputs.keys() if i[:6] == 'output']
            a = self.inputs['job']['problem_id']
            c = '[0-9]*'
            varlist = list(filter(None, [self.inputs[i].get('variable') for i in outs]))
            self._coarse_data = None
            self._fine_data = None
            if os.path.isfile(os.path.join(self.path, 'fft.tar')):
                self._tar = tarfile.open(os.path.join(self.path,'fft.tar'), 'r|')
                tmp = [os.path.join(self.path, i) for i in ['cksum', 'hash']]
                tmp = [i for i in tmp if os.path.isfile(i)][0]
                with open(tmp) as f:
                    lines = [i.strip().split(' ')[-1] for i in f.readlines()]
                #lines = [os.path.join(self.path, i) for i in lines]
                self._tfiles = [i for i in lines if i]
            for out in outs:
                files = []
                b = self.inputs[out].get('id', 'out' + out[6:])
                searches = ['.'.join([a, b, c, ext]) for ext in _ext]
                for search in searches:
                    files += [os.path.split(i)[-1] for i in glob(os.path.join(path, search))]
                if b[:2] == 'FT':
                    try:
                        files += [i for i in self._tfiles if b == i.split('.')[1]]
                    except AttributeError:
                        pass
                    #print(b, len(files))
                self.fileDict[out] = sorted(files)
                var = self.inputs[out].get('variable')
                if varlist.count(var) == 1:
                    self.varDict[var] = out
                self.idDict[b] = out

        #for attr in ['r', 'phi', 'rc', 'phic']:
        #    setattr(self, attr, getattr(tmp, attr))
        self._rho_ref = rho_ref
        self._fft_data = None
        self._coarse_data = coarse_data
        self._fft_time = fft_time
        self._phase_angle = phase_angle
        self._mode_mask = mode_mask
        self._sigmas = [0,4]
        self._main_modes = main_modes
        self._mode_detect = None

        return None
        # End init

    @property
    def rho_ref(self):
        if self._rho_ref is None:
            try:
                self._rho_ref = np.real(self.loadfile('FT',0)['FT-dens'][0])
            except IOError:
                self._rho_ref = self.loadfile('cons',0)['rho'].mean(axis=0)
        return self._rho_ref

    def drho_plot(self, *args):
        try:
            args[0].drho_plot()
        except AttributeError:
            try:
                self.loadfile(*args).drho_plot()
            except AttributeError:
                self.loadfile('cons', args[0]).drho_plot()

    def draw_spiral(self, rp, phi0=0, opt=None):
        if opt is None:
            opt = {}
        if 'ls' not in opt:
            opt['ls'] = ':'
        if 'lw' not in opt:
            opt['lw'] = 1
        if 'c' not in opt:
            opt['c'] = '1'
        r = np.array([i for i in self.rc if i >= rp])
        phi = spiral(r, rp, 1./self.mach) + phi0
        plt.plot(r * np.cos(phi), r * np.sin(phi), **opt)

    def __repr__(self):
        return '<BLsim "{0:}">'.format(self.name)

    def _unwrap(self, time=None, phase=None, limit=None, mNorm=False):
        if phase is None:
            phase = np.angle(self.fft)
        if time is None:
            time = self.fft_time
        if limit is None:
            limit = 0
            #limit = - .1 / np.arange(phase.shape[1])[np.newaxis,:,np.newaxis]
        if mNorm:
            limit /= np.arange(phase.shape[1])[np.newaxis,:,np.newaxis]
        d = crudeDiff(time, phase, axis=0)
        shift = np.zeros_like(phase)
        shift[np.where(d < limit)] += tau
        return phase + shift.cumsum(axis=0)

    def fluxes(self, t0, tf, tnorm=tau, nsmooth=True, progress=True):
        if not tnorm or tnorm is True:
            tnorm = 1
        t0 *= tnorm
        tf *= tnorm
        n = 0
        ffts = [i for i in self.fileDict.keys()
                if self.inputs.get(i, {}).get('variable') == "FT-Range"]
        dt = self.inputs[ffts[0]]['dt']
        ffts = self.sortedFFT()
        i = int(2 * t0 // dt)
        i1 = int(min(2 * tf // dt, len(ffts) - 1))
        if nsmooth is True:
            nsmooth = int((i1 - i) // 10)
        if not nsmooth:
            nsmooth = 1
        print("ns:", nsmooth)
        out = {'dwdt': 0, 'drhodt': 0}
        tsb = None
        i0 = i
        while i <= i1:
            fn = ffts[i]
            ft = self.loadfile(os.path.split(fn)[-1])
            #ft = loadBLfile(fn, file_handle=fh, sim_path=os.path.abspath(self.path), sim=self, ai_data=self.inputs)
            #print('File:', ft)
            if n == 0:
                print('t0', ft.t / tnorm, t0 / tnorm, i, ft.fn)
                out['t0'] = ft.t / tnorm
                out['drho'] = np.real(ft['FT-dens'][0]) - self.rho_ref
                tsa = ft.t
                #print(ft['FT-vel2'][0])
            if progress:
                helpers.update_progress((i - i0) / (i1 - i0 + 1))
            if n < nsmooth:
                out['drhodt'] -= np.real(ft['FT-dens'][0])
                out['dwdt'] -= np.real(ft['FT-vel2'][0])
            elif i > i1 - nsmooth:
                out['drhodt'] += np.real(ft['FT-dens'][0])
                out['dwdt'] += np.real(ft['FT-vel2'][0])
                if tsb is None:
                    tsb = ft.t
            tmp = ft.fluxes()
            try:
                for k in keys:
                    out[k] += tmp[k]
            except NameError:
                keys = tmp.keys()
                out.update(tmp)
            n += 1
            i += 1
        if progress:
            helpers.update_progress(1)
        print('tf', ft.t / tnorm, tf / tnorm, i, ft.fn)
        print(n,i, i1)
        out['tf'] = ft.t / tnorm
        #out['dwdt'] += np.real(ft['FT-vel2'][0])
        #out['drhodt'] += np.real(ft['FT-dens'][0])
        #print(ft['FT-vel2'][0])
        #print(out['dw'])
        #out['dwdt'] /= tnorm * (out['tf'] - out['t0'] - t1) * self.rc * .5
        #out['drhodt'] /= tnorm * (out['tf'] - out['t0'] - t1) * self.rc * .5
        #out['dwdt'] /= tnorm * (out['tf'] - out['t0']) * self.rc
        out['dwdt'] /= (tsb - tsa) * self.rc * nsmooth
        out['drhodt'] /= (tsb - tsa) * self.rc * nsmooth
        print("t div", tnorm * (out['tf'] - out['t0']), dt * nsmooth, tsb - tsa)

        for k in keys:
            out[k] /= n
        return out

    def plot_fluxes(self, t0=None, tf=None, nm=5, data=None, figsize=None, save=False,
                    fn=None, ext='pdf', lopt=None, ff=1, sdir='', progress=True):
        if lopt is None:
            lopt = dict(handlelength=1, fontsize=8, handletextpad=.4, columnspacing=.7)
        if data is None:
            data = self.fluxes(t0, tf, progress=progress)
        csm = np.real(data['CSm'])
        csm[0] = 0
        cs = data['CS'][0]
        norm = self.intr(np.abs(csm))
        modes = sorted(range(norm.shape[0]), key=lambda x: -norm[x])

        if figsize is None:
            figsize = np.array((11,8.5)) * .8
        fig = plt.figure(figsize=figsize)
        gs = mpl.gridspec.GridSpec(3, 2, top=.93, left=.08, right=.98, bottom=.08, wspace=.15, hspace=.25)
        #fig, axs = plt.subplots(3, 2, figsize=figsize, top=.7)

        ax0 = plt.subplot(gs[0,0])
        plt.plot(self.rc, cs, 'k-', label='$C_S$')
        for m in modes[:nm]:
            plt.plot(self.rc, csm[m], label=str(m))
        plt.plot(self.rc, csm[1:].sum(axis=0), c='.5', ls=':', label='sum')
        plt.xlim(self.r[0], self.r[-1])
        #ylim = plt.ylim()
        plt.legend(ncol=nm + 2, **lopt)
        plt.axhline(0, c='.5', ls=':', lw=1)
        plt.axvline(1, c='.5', ls=':', lw=1)
        #plt.ylim(*ylim)
        #plt.xlabel('$R$')
        plt.ylabel('$C_S$')
        #plt.setp(ax0.get_xticklabels(), fontsize=6)

        ax = plt.subplot(gs[0,1])
        for i, m in enumerate(modes[:nm]):
            plt.plot(self.rc, csm[m], label=str(m), zorder=i+1)
        ylim = plt.ylim()
        plt.plot(self.rc, cs, 'k-', label='$C_S$', zorder=0)
        plt.plot(self.rc, csm[1:].sum(axis=0), c='.5', ls=':', label='sum', zorder=nm+2)
        plt.xlim(self.r[0], 1.4)
        plt.ylim(*ylim)
        #plt.legend(ncol=nm + 2, **lopt)
        plt.axhline(0, c='.5', ls=':', lw=1)
        plt.axvline(1, c='.5', ls=':', lw=1)
        #plt.ylim(*ylim)
        #plt.xlabel('$R$')
        plt.ylabel('$C_S$')
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.02))
        locs, labels = plt.yticks()
        dy = .2 * (locs[1]-locs[0])
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(dy))
        #plt.setp(ax0.get_xticklabels(), fontsize=6)


        ax = plt.subplot(gs[1,0], sharex=ax0)
        keys = [i for i in data.keys() if i[0] == 'C' and len(i) == 2]
        for k in keys:
            opt = {'label': '${0:}_{1:}$'.format(*k)}
            if k == 'CS':
                opt['c'] = 'k'
            plt.plot(self.rc, data[k][0], **opt)
        plt.legend(ncol=3, **lopt)
        plt.axhline(0, c='.5', ls=':', lw=1)
        plt.axvline(1, c='.5', ls=':', lw=1)
        plt.xlim(self.r[0], self.r[-1])
        #plt.xlabel('R')
        #plt.setp(ax.get_xticklabels(), visible=False)

        ax = plt.subplot(gs[1,1], sharex=ax0)
        ri = self.rloc(1)
        plt.plot(self.rc, data['drho'], label=r'$\delta\rho$')
        plt.plot(self.rc, data['vphi'] / self.rc, label=r'$\Omega$')
        op = self.rc**-3
        op +=  self.mach**-2 * grad(self.rc, data['dens']) / (data['dens'] * self.rc)
        op = np.sqrt(op)
        plt.plot(self.rc, op, label=r'$\Omega(P)$', ls='--')
        plt.plot(self.rc, self.rc**-1.5, label=r'$\Omega_{\rm k}$', lw=1, c='k', ls=':')
        plt.plot(self.rc, -1e3*data['vr']*self.mach, label=r'$-10^3v_r/c_s$')
        plt.legend(ncol=5, **lopt)
        plt.xlim(self.r[0], self.r[-1])
        ymax = data['drho'][ri:].max() * 1.05
        ymin = min(data['drho'][ri:].min() - .1 * ymax, 0)
        ylim = plt.ylim()
        plt.ylim(max(ylim[0], ymin), min(ylim[1], ymax))
        plt.axhline(0, c='.5', ls=':', lw=1)
        plt.axvline(1, c='.5', ls=':', lw=1)
        plt.xlabel('$R$')
        #ax.set_xticklabels([])
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.1))

        ax = plt.subplot(gs[2,0], sharex=ax0)
        plt.plot(self.rc, - data['Mdot'] * tau * self.rc, label=r'$\dot{M}$', c='k')
        ri = self.rloc(1.2)
        ri2 = self.rloc(2)
        norm = 1 / grad(self.rc, data['vphi'] * self.rc)
        ycs = norm * grad(self.rc, data['CS'][0])
        plt.plot(self.rc, ycs, label=r'$C_S$')
        ydw = norm * self.rc**3 * data['dens'] * data['dwdt'] * tau
        plt.plot(self.rc, ydw, label=r'$\partial_t \Omega$')
        plt.plot(self.rc, ycs + ydw, label=r'$C_S\! +\! \partial_t \Omega$', c='.5', ls=':')
        ydp = np.pi * self.rc**3.5 * grad(self.rc, data['drhodt']) * self.mach**-2
        ydp /= grad(self.rc, data['vphi'] * self.rc)
        plt.plot(self.rc, ydp, label=r'$\partial_t\partial_rP$')
        plt.axhline(0, c='.5', ls=':', lw=1)
        plt.axvline(1, c='.5', ls=':', lw=1)
        plt.legend(ncol=5, **lopt)
        ymax = np.maximum(ycs, data['Mdot'])
        ymax = ymax[ri:ri2].max() * 1.05
        ymin = np.minimum(ydw, data['Mdot'])
        ymin = min(ymin[ri:ri2].min() - .1 * ymax, 0)
        ylim = plt.ylim()
        ylim = plt.ylim(max(ylim[0], ymin), min(ylim[1], ymax))
        #print(ylim)
        yl = 2e-4
        #plt.ylim(-yl, yl)
        plt.xlim(self.r[0], self.r[-1])
        #plt.setp(ax.get_xticklabels(), visible=False)
        plt.xlabel('$R$')

        ax = plt.subplot(gs[2,1])
        plt.xticks([], [])
        plt.yticks([], [])
        one = self.rloc(1)
        mdot = data['Mdot'] * tau * self.rc
        info = [['C_S', data['CS'][0,one]],
               ['C_A', data['CA'][0,one]],
               ['C_L', data['CL'][0,one]],
               [r'\dot{M}', mdot[one]],
               ]
        info = ['$' + i[0] + '(R=1)=$'+ '{0:.3g}'.format(np.real(i[1])) for i in info]
        info.append(r'$\dot{M}(R_{\rm min})=$' + '{0:.3g}'.format(mdot[0]))
        info.append(r'$\dot{M}(R_{\rm max})=$' + '{0:.3g}'.format(mdot[-8:-2].mean()))
        integrand = data['drho'] * self.rc * self.dr
        info.append(r'$2\pi\int r\delta\rho dr=$' + '{0:.3g}'.format(tau*np.sum(integrand)))
        tmp = self.rc[self.rc<1][np.argmin(np.abs(data['drho'][self.rc<1]))]
        tmp = r'$2\pi\int_{'+'{:.2g}'.format(tmp)+r'}^4 r\delta\rho dr=$' + '{0:.3g}'.format(tau*np.sum(integrand[self.rloc(tmp):]))
        info.append(tmp)
        ncol = 2
        for i, s in enumerate(info):
            ix = i % ncol
            iy = i // ncol
            plt.text(.9 / ncol * ix + .05, .88 - .12 * iy, s)


        title = ''
        try:
            title = self.name + ' '
        except:
            pass
        title += '$t/ 2 \pi={t0:.1f}-{tf:.1f}$'.format(**data)
        fig.suptitle(title)
        #plt.tight_layout()
        #fig.tight_layout(rect=[0, 0.0, 1, 0.5])
        #fig.subplots_adjust(top=0.7)
        #plt.subplots_adjust(left=0.2, wspace=0.8, top=0.5)

        if save or fn:
            if fn is None:
                fn = '_flux_{:.1f}_{:.2f}.'.format(data['t0'], data['tf'] - data['t0'])
                fn = os.path.join(sdir, self.name + fn + ext)
                if sdir:
                    if not os.path.isdir(sdir):
                        os.mkdir(sdir)
            plt.savefig(fn)
            plt.close()

        return data

    def flux_series(self, t0=None, delta_t=100, dt0=50, save=True, sdir=None, **kwargs):
        if sdir is True:
            #sdir = self.name + '_fluxes'
            sdir = 'fluxes'
        if t0 is None:
            t0 = np.arange(0, self.fft_time[-1] / tau, dt0)
        for t in t0:
            self.plot_fluxes(t, t + delta_t, save=save, sdir=sdir, **kwargs)


    def sortedFFT(self):
        ffts = [out for out in self.fileDict.keys()
                if self.inputs.get(out, {}).get('variable') == "FT-Range"]
        ffts.sort(key=lambda x: self.inputs.get(out, {}).get('start_time', 0))
        n = max([len(self.fileDict[i]) for i in ffts])
        out = []
        for i in range(n):
            for series in ffts:
                try:
                    out.append(self.fileDict[series][i])
                except IndexError:
                    pass
        return [os.path.join(self.path, i) for i in out]

    def gen_fft_file(self, var='FT'):
        handler = IncrementalFFT(self.sortedFFT(), var=var, sim=self)
        handler.process()

    def load_fft_data(self, fine=False, var='FT'):
        _type = 'coarse'
        if fine:
            _type = 'fine'
        fn = os.path.join(self.path, 'FFT_' + _type + '_' + var + '.npy')
        if not os.path.isfile(fn):
            # TODO: store data
            self.gen_fft_file(var=var)
        data = FTdataFile(fn, sim=self)
        if _type == 'coarse':
            self._coarse_data = data
        else:
            self._fine_data = data
        return data

    @property
    def coarse_data(self):
        if self._coarse_data is None:
            self.load_fft_data()
        return self._coarse_data

    @property
    def fine_data(self):
        if self._fine_data is None:
            self.load_fft_data(fine=True)
        return self._fine_data

    def __FFT_composite(self):
        ffts = [out for out in self.fileDict.keys()
                if self.inputs.get(out, {}).get('variable') == "FT-Range"]
        return CompositeFFTSet([FFTset(self.fileDict[i], sim=self) for i in ffts])

    def __collect_fft_data(self):
        ffts = [out for out in self.fileDict.keys()
                if self.inputs.get(out,{}).get('variable') == "FT-Range"]
        ffts = sorted([FFTset(self.fileDict[i], sim=self) for i in ffts], key=lambda x:x.time[1])
        nt = sum([i.time.size for i in ffts])
        shape = (nt,) + ffts[0].data.shape[1:]
        data = np.empty(shape, dtype='complex64')
        time = np.empty(nt)
        n = len(ffts)
        for i in range(n):
            data[i::n] = ffts[i].data
            time[i::n] = ffts[i].time
        while time[1] == 0:
            data = data[1:]
            time = time[1:]
        phase = self._unwrap(time, np.angle(data))
        if self._fft_data is None:
            self._fft_data = data
        if self._fft_time is None:
            self._fft_time = time
        if self._phase_angle is None:
            self._phase_angle = phase
        return time, data, phase

    def __load_fft_data(self):
        data = []
        t = []
        try:
            files = self.files('FT-Range')
        except ValueError:
            f = 'output3'
            if self.inputs[f]['id'] == 'FT':
                files = self.files(f)
            else:
                raise NotImplementedError
        for fn in files:
            f = self.loadfile(fn)
            data.append(f['FT'])
            t.append(f.t)
        data = np.array(data)
        t = np.array(t)
        if self._fft_data is None:
            self._fft_data = data
        if self._fft_time is None:
            self._fft_time = t
        return data

    def verify_ft(self, orbit=100):
        bf = self.loadfile('cons', orbit)
        ft = self.loadfile('FT', orbit*10)
        varlist = ['pseudo', 'vel1', 'vel2', 'dens', 'dens**2', 'CL', 'v1v2', 'Mdot', 'vortensity']
        fig = plt.figure()
        for var in varlist:
            if var == 'CL':
                a = (bf['dens']*bf.v1v2()).mean(axis=0)
            else:
                a = bf[var].mean(axis=0)
            b = ft['FT-' + var + '-Re'][0]
            if var == 'vortensity':
                a[0] = b[0]
                a[-1] = b[-1]
            plt.plot(self.rc, (a-b)/np.maximum(np.abs(a),np.abs(b)), lw=1)
        plt.ylim(-5e-3, 5e-3)
        plt.legend(varlist)
        plt.title('Time/$2\pi={0:.1f}$'.format(bf.t / tau))

    def intr(self, data, axis=-1):
        return intr(self.dr, data, axis=axis)

    @property
    def __fft(self):
        if not self._fft_data is None:
            return self._fft_data
        return self._collect_fft_data()[1]

    @property
    def fft_data(self):
        if self._fine_data is None:
            return self.coarse_data
        return self.fine_data

    @property
    def fft(self):
        return self.fft_data.FT

    @property
    def amp(self):
        return self.fft_data.amp

    @property
    def phase(self):
        return self.fft_data.phase

    @property
    def speed(self):
        return self.fft_data.speed

    @property
    def fft_time(self):
        return self.fft_data.t

    @property
    def filenames(self):
        return sum(self.fileDict.values(), [])

    def files(self, key=None):
        if key is None:
            return self.filenames
        if key in self.idDict:
            return self.fileDict[self.idDict[key]]
        if key in self.fileDict:
            return self.fileDict[key]
        if key in self.varDict:
            return self.fileDict[self.varDict[key]]
        raise ValueError('Cannot find "{0:}" files.'.format(key))

    def _readable(self, t):
        try:
            out = self.loadfile(t)
        except IOError:
            return False
        if out is None:
            out = False
        return out

    def __gen_fft_time(self):
        t = None
        if self._fft_data is None:
            self._collect_fft_data()
            t = self._fft_time
        if t is None:
            if 'FT-Range' in self.varDict:
                dt = self.inputs[self.varDict['FT-Range']]['dt']
                t = np.arange(len(self.files('FT-Range'))) * dt
            else:
                raise NotImplementedError
            if self._fft_time is None:
                self._fft_time = t
        return t

    @property
    def __fft_time(self):
        if not self._fft_time is None:
            return self._fft_time
        return self._gen_fft_time()

    @property
    def __phase_angle(self):
        if not self._phase_angle is None:
            return self._phase_angle
        if self._fft_data is None:
            return self._collect_fft_data()[2]
        self._phase_angle = self._unwrap()
        return self._phase_angle

    def rloc(self, r, subsample=None):
        rc = self.rc[::subsample]
        return np.abs(r - rc).argmin()

    def tloc(self, t):
        return np.abs(t - self.fft_time).argmin()

    def extract_tar(self):
        print('Extract')
        self._tar.extractall(self.path)
        self._tfiles = None

    def loadfile(self, fn, index=None):
        if not index is None:
            fn = self.files(fn)[index]
        if fn in self.filenames:
            fh = None
            tmp = [fn, os.path.split(fn)[-1], os.path.join(self.path, fn)]
            tmp = np.array(map(os.path.exists, tmp))
            if not tmp.any():
                if fn in self._tfiles:
                    self.extract_tar()
            return loadBLfile(fn, file_handle=fh, sim_path=os.path.abspath(self.path), sim=self, ai_data=self.inputs)

    def __mode_mask(self, data=None, info=False, amin=None):
        if self._mode_mask is None:
            if amin is None:
                amin = 1e-3 / self.rc.size / self.phic.size
            amp = np.abs(self.fft)
            #Max = amp.max(axis=1)[:, np.newaxis, :]
            mask = np.zeros_like(amp, dtype=bool)
            one = np.ones_like(amp)
            mask[:,0,:] = True # always mask out m=0
            tmp = amp.copy()
            tmp[:,0,:] = 0
            mask = np.logical_or(mask, tmp == tmp.max(axis=1)[:, np.newaxis, :])
            del(tmp)
            _mask = mask.copy()
            keep_going = 1
            #mask = np.logical_or(mask, amp - .5 * Max > 0)
            # now recursively mask out signal
            mask = np.logical_or(mask, amp - percentile(amp, 95, axis=1)[:, np.newaxis, :]> 0)
            while keep_going:
                mamp = np.ma.array(amp, mask=mask)
                if keep_going == 1:
                    mask = _mask.copy()
                #M = mamp.max(axis=1).data[:, np.newaxis, :]
                m = mamp.mean(axis=1).data[:, np.newaxis, :] # mean of masked data
                s = mamp.std(axis=1).data[:, np.newaxis, :] # standard deviation
                tmp = np.logical_and(amp > amin, m + self._sigmas[-1] * s - amp < 0) # signal
                tmp = np.logical_or(_mask, tmp)
                if (mask == tmp).all():
                    keep_going = False
                else:
                    keep_going += 1
                mask = tmp.copy()
                if keep_going > 20:
                    print('Break')
                    break
            if info:
                mamp = np.ma.array(amp, mask=mask)
                m = mamp.mean(axis=1).data
                s = mamp.std(axis=1).data
                _info = {'mean': m, 'std': s}
            mask = np.logical_not(mask)
            mask[:,0,:] = True
            #mask = and_neighbor(mask, n=1, axis=0)
            #mask = and_neighbor(mask, n=1, axis=2)
            self._mode_mask = mask
        else:
            mask = self._mode_mask
            if info:
                tmp = np.logical_not(mask)
                tmp[:,0,:] = True
                amp = np.abs(self.fft)
                mamp = np.ma.array(amp, mask=tmp)
                m = mamp.mean(axis=1).data
                s = mamp.std(axis=1).data
                _info = {'mean': m, 'std': s}
        if not data is None:
            out = data.copy()
            out[np.where(mask)] = np.nan
            #out[np.where(mask)] = data[np.where(mask)]
            if info:
                _info['data'] = out
            else:
                return out
        if info:
            _info['mask'] = mask
            return _info
        return mask

    def __mode_phase(self):
        DeprecationWarning('This function (mode_phase) is deprecated.')
        out = np.array([np.abs(self.fft), self.phase_angle])
        return out

    def __prop_speed(self, dt=None, ir=None):
        if dt is None:
            for var in ['FT-Range', 'FT']:
                if var in self.varDict:
                    dt = self.inputs[self.varDict[var]]['dt']
                    break
        if dt is None:
            dt = self.inputs['output3']['dt']
        m = np.arange(self.fft.shape[1])
        if ir is None:
            data = self.phase_angle
            m = m[np.newaxis,:,np.newaxis]
        else:
            data = self.phase_angle[:,:,ir]
            m = m[np.newaxis,:]
        #dphi = np.gradient(data[1] / dt, axis=0)
        #dphi = mod_grad(data, axis=0, mod=tau)
        dphi = grad(self.fft_time, data)
        return dphi / m

    def mt_plot(self, r, fn=None, save=False, ext='pdf', sdir=None, fig=None, ax=None, fopt={}, vmin='smart',
                vmax='max', cb=True, cbl=None, popt={}, log=True, mmax=None):
        if mmax is None:
            mmax = self.fft.shape[1] - 1
        #ir = np.abs(self.rc - r).argmin()
        ir = self.rloc(r)
        r = self.rc[ir]
        _amp = np.abs(self.fft)
        data = _amp[:,:,ir].T

        if 'smart' in [vmin, vmax]:
            smart = helpers.smartlim(_amp[:,:mmax+1,:])
            if vmin == 'smart':
                vmin = smart[0]
            if vmax == 'smart':
                vmax = smart[1]
        if vmin == 'min':
            vmin = _amp[:,:mmax+1,:].min()
        if vmax == 'max':
            vmax = _amp[:,:mmax+1,:].max()
        if vmin == 'auto':
            vmin = max(data.min(), 1e-6)
        _opt = {'vmin': vmin, 'vmax': vmax, 'interpolation': 'nearest'}
        if log:
            _opt['norm'] =  mpl.colors.LogNorm()
        _opt.update(popt)

        if fig is None and ax is None:
            fig = plt.figure(**fopt)
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        fig = plt.gcf()

        plt.imshow(data, **_opt)
        plt.ylim(None, mmax)
        if self.name == 'test_run':
            plt.axvline(101.5, c='k', ls=':', lw=1)
        plt.xlabel('Time')
        plt.ylabel('Mode')
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:.2f}$'.format(r))
        #plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:}$'.format(helpers.eformat(r, math=False)))
        if cb:
            cb = plt.colorbar()
            if cbl is None:
                cbl = r'$\left|a_m\right|^2$'
            if cbl:
                cb.set_label(cbl)
        if save or fn:
            if log:
                log = 'log'
            else:
                log = 'lin'
            if fn is None:
                fn = self.name + '_mode-time_r={0:.2e}_{1:}.'.format(r, log) + ext
            fig.savefig(fn)
            plt.close()

    def main_modes(self, nm=None, skip_zero=True, rmax=1.7):
        if self._main_modes is None:
            fft = self.fft * self.rc[np.newaxis, np.newaxis, :]
            amp = self.amp.copy()
            amp[:, :, np.where(self.rc > 1.7)[0]] = 0
            if self.fft_time[-1] < 200:
                nt = self.fft_time.size
                a = self.intr(amp[nt//2:].sum(axis=0))
            else:
                a = self.intr(amp[self.tloc(100 * tau) - 1:].sum(axis=0))
            self._main_modes = sorted(range(a.size), key=lambda x: -a[x])
        modes = self._main_modes[:]
        if skip_zero:
            try:
                modes.remove(0)
            except ValueError:
                pass
        return modes[:nm]

    def _old_r_phase_plotter(self, r, data, ret_m=False, tloc=None, sort=True,
                         sdata=None, order=None):
        ir = self.rloc(r)
        #data = self.mode_phase()
        modes = []
        handles = []
        if len(data.shape) == 3:
            rdata = data[:,:,ir]
        else:
            rdata = data
        for i in xrange(data.shape[1]):
            tmp = rdata[:,i].copy()
            #tmp = tmp[tloc]
            j = 0
            # make sure data covers 5 adjacent times
            while np.isfinite(tmp).any() and j < 5:
                tmp = np.gradient(tmp)
                j += 1
            if np.isfinite(tmp).any():
                modes.append(i)
                handles.append(None)
        if order is None:
            order = modes[:]
            if sort:
                if sdata is None:
                    sdata = np.abs(self.fft[:,:,ir]).sum(axis=0)
                order = sorted(modes, key=lambda m: sdata[m])
        if len(modes) > 9:
            colors1 = plt.cm.viridis(np.linspace(0., 1, 128))
            colors2 = plt.cm.plasma(np.linspace(0, 1, 128))
            colors = np.vstack((colors1, colors2))
            mymap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
        else:
            mymap = plt.cm.viridis
        norm = 1. / (len(modes) - 1.)
        for m in order:
            i = modes.index(m)
            c = mymap(i * norm)
            #print(i,m,sdata[m])
            handles[i] = plt.plot(self.fft_time / tau, rdata[:,m], lw=1, c=c)[0]
        opt = {'loc': 0, 'frameon': True, 'handlelength': .7, 'prop': {'size':8}, 'ncol': 3}
        lbls = ['$%d$' % m for m in modes]
        leg = plt.legend(handles, lbls, **opt)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.xlabel(r'Time/$2\pi$')
        #plt.ylabel('Phase')
        plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:.2f}$'.format(r))
        if ret_m:
            return modes

    def r_phase(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        ir = self.rloc(r)
        # TODO: Fix mask
        mask = self.mode_mask()[:,:,ir]
        data = np.ma.array(self.phase[:,:,ir], mask=mask)
        m = self._r_phase_plotter(r, data, ret_m=ret_m)
        plt.ylabel('Phase')
        return m

    def __old_r_speed(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        ir = self.rloc(r)
        mask = self.mode_mask()[:,:,ir]
        speed = self.prop_speed(ir=ir)
        data = speed.copy()
        data[mask] = np.nan
        speed = np.ma.array(speed, mask=mask)
        sdata = speed.mean(axis=0) / speed.std(axis=0)
        opt = dict(ret_m=ret_m, sdata=sdata)
        m = self._old_r_phase_plotter(r, data, **opt)
        ylim = list(plt.ylim())
        ylim[0] = max(0, ylim[0])
        ylim[1] = min(1, ylim[1])
        plt.ylim(*ylim)
        plt.ylabel('Speed')
        return m

    def __old_r_amp(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        ir = self.rloc(r)
        mask = self.mode_mask()[:,:,ir]
        amp = np.abs(self.fft[:,:,ir])
        data = np.ma.array(amp, mask=mask)
        #data[mask] = np.nan
        m = self._old_r_phase_plotter(r, data, ret_m=ret_m)
        nt = self.fft_time.size
        m0 = data[:nt//5].max()
        m1 = data[nt//5:].max()
        plt.ylim(0, None)
        if m0 > 1.2 * m1:
            plt.ylim(None, 1.1 * m1)
        mask = np.logical_not(mask)
        mask[:,0] = True
        tmp = np.ma.array(amp, mask=mask)
        std = tmp.std(axis=1)
        mean = tmp.mean(axis=1)
        ls = '-'
        for i in self._sigmas:
            y = mean + i * std
            plt.plot(self.fft_time / tau, y, c='k', ls=ls, lw=1)
            ls = ':'
        plt.ylabel('Amplitude')
        return m

    def _r_phase_plotter(self, r, data, modes=None, nm=5, add_modes=None, std_plot=False,
                         ret_m=None, smooth=False, sw=20, std=None, rsmooth=None, fn=None,
                         save=None, ext='pdf', cout=None, add_max=None):
        ir = self.rloc(r)
        rslice = ir
        r = self.rc[ir]
        if modes is None:
            modes = self.main_modes(nm=nm)[::-1]
        if add_modes is not None:
            add_modes = list(np.atleast_1d(add_modes))
            modes.extend([m for m in add_modes if m not in modes][:add_max])
        modes = list(modes)
        handles = {}
        weight = None
        rweight = None
        if rsmooth:
            if rsmooth == -1:
                if r <= 1:
                    rsmooth = 5
                else:
                    rsmooth = 30
            if rsmooth < 0:
                raise ValueError('Keyword "rsmooth" cannot be negative.')
            rslice = slice(max(ir - rsmooth, 0), ir + rsmooth + 1)
            weight = std**-2
            rdata, rweight = np.average(data[:,modes,rslice], weights=weight[:,modes,rslice], axis=2, returned=True)
        else:
            if len(data.shape) == 3:
                rdata = data[:,modes,ir]
            else:
                rdata = data[:,modes]
            if std is not None:
                if len(std.shape) == 3:
                    std = std[:, modes, ir]
                else:
                    std = std[:, modes]
        for i, m in enumerate(modes):
            line = rdata[:,i]
            if smooth:
                if std is not None and rsmooth:
                    line = running_mean(rdata[:,i], rweight[:,i], sw)
                    #print(sw,line.size,self.fft_time.size)
                else:
                    if smooth in [True, 1]:
                        smooth = 'flat'
                    line = helpers.smooth(line, window=smooth, window_len=sw)
            handles[m] = plt.plot(self.fft_time / tau, line, lw=1)[0]
            if std is not None and std_plot:
                c = handles[m].get_color()
                plt.fill_between(self.fft_time / tau, line - std[:,m], line + std[:,m], color=c, alpha=.1)
        opt = {'loc': 0, 'frameon': True, 'handlelength': .7, 'prop': {'size':8}, 'ncol': 3}
        modes.sort()
        cd = {m: handles[m].get_color() for m in handles.keys()}
        handles = [handles[m] for m in modes]
        lbls = ['$%d$' % m for m in modes]
        leg = plt.legend(handles, lbls, **opt)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.xlabel(r'Time/$2\pi$')
        #plt.ylabel('Phase')
        ax = plt.gca()
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(25))
        plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:.2f}$'.format(r))
        plt.xlim(np.floor(self.fft_time[0] / tau), np.ceil(self.fft_time[-1] / tau))
        if cout:
            return cd

    def r_speed(self, r, fig=True, save=None, fn=None, ext='pdf', tmark=None, **kwarg):
        if fig is True:
            plt.figure()
        if not 'smooth' in kwarg:
            kwarg['smooth'] = 'flat'
            if not 'sw' in kwarg:
                kwarg['sw'] = 20
        ir = self.rloc(r)
        data = self.speed
        try:
            kwarg['std'] = self.fft_data._speed_std
            if 'rsmooth' not in kwarg:
                kwarg['rsmooth'] = -1
        except AttributeError:
            pass
        cd = self._r_phase_plotter(r, data, cout=True, **kwarg)
        ylim = list(plt.ylim())
        ylim[0] = 0
        ylim[1] = 1
        plt.ylim(*ylim)
        xlim = plt.xlim()
        if tmark is not None:
            tmark = np.atleast_1d(tmark)
            for t in tmark:
                c='k'
                try:
                    if len(t) > 1:
                        m = t[1]
                        t = t[0]
                        c = cd.get(int(m), 'k')
                except TypeError:
                    pass
                x = (t, t)
                y = (ylim[0], ylim[0] + .05 * (ylim[1] - ylim[0]))
                plt.plot(x, y, c=c)
            plt.xlim(*xlim)
            plt.ylim(*ylim)
        plt.ylabel('Speed')
        if fn and save is None:
            save = True
        if save:
            if fn is None:
                fn = helpers.sanitize_lbl(self.name) + '_r_speed.' + ext.lstrip('.')
            plt.savefig(fn)
            plt.close()
        return None

    def r_amp(self, r, fig=True, **kwarg):
        if not 'smooth' in kwarg:
            kwarg['smooth'] = 'flat'
            if not 'sw' in kwarg:
                kwarg['sw'] = 20
        if fig is True:
            plt.figure()
        ir = self.rloc(r)
        data = self.amp
        try:
            kwarg['std'] = self.fft_data._speed_std
            if 'rsmooth' not in kwarg:
                kwarg['rsmooth'] = -1
        except AttributeError:
            pass
        self._r_phase_plotter(r, data, **kwarg)
        nt = self.fft_time.size
        m0 = data[:nt//5, 1:, ir].max()
        m1 = data[nt//5:, 1:, ir].max()
        plt.ylim(0, None)
        if m0 > 1.2 * m1:
            plt.ylim(None, 1.1 * m1)
        plt.ylabel('Amplitude')
        return None

    def _t_phase_plotter(self, data, ret_m=False, sort=True, sdata=None,
                         modes=None, std=None, nm=None):
        if modes is None:
            modes = []
            for i in xrange(data.shape[0]):
                tmp = data[i,:].copy()
                #tmp = tmp[tloc]
                j = 0
                # make sure data covers 5 adjacent r
                while np.isfinite(tmp).any() and j < 5:
                    tmp = np.gradient(tmp)
                    j += 1
                if np.isfinite(tmp).any():
                    modes.append(i)
        order = modes[:]
        if sort:
            if sdata is None:
                sdata = data.mean(axis=1)
            order = sorted(modes, key=lambda m: sdata[m])
        if not nm is None:
            order = order[-nm:]
            modes = [m for m in modes if m in order]
        handles = [None] * len(modes)
        if len(modes) > 9:
            colors1 = plt.cm.viridis(np.linspace(0., 1, 128))
            colors2 = plt.cm.plasma(np.linspace(0, 1, 128))
            colors = np.vstack((colors1, colors2))
            mymap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
        else:
            mymap = plt.cm.viridis
        norm = 1. / (len(modes) - 1.)
        for m in order:
            i = modes.index(m)
            c = mymap(i * norm)
            #print(i,m,sdata[m])
            opt = dict(lw=1, c=c)
            handles[i] = plt.plot(self.rc, data[m,:], **opt)[0]
            if not std is None:
                plt.plot(self.rc, data[m,:] - std[m,:], ls=':', **opt)
                plt.plot(self.rc, data[m,:] + std[m,:], ls=':', **opt)
        opt = {'loc': 0, 'frameon': True, 'handlelength': .7, 'prop': {'size':8}, 'ncol': 3}
        lbls = ['$%d$' % m for m in modes]
        leg = plt.legend(handles, lbls, **opt)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.xlabel('Radius')
        if ret_m:
            return modes

    def t_amp(self, t='mean', ret_m=False, fig=True, tmin=2e2*tau, nm=5):
        if fig is True:
            plt.figure()
        if t == 'mean':
            data = np.abs(self.fft[self.tloc(tmin):]).mean(axis=0)
            title = '$t$ mean'
            sdata = data.mean(axis=1)
        else:
            it = self.tloc(t)
            data = np.abs(self.fft[it,:,:])
            title = '$t={0:.2f}$'.format(t)
            sdata = data.mean(axis=1)
        m = self._t_phase_plotter(data, ret_m=ret_m, nm=nm, sdata=sdata)
        plt.title(title)
        plt.ylabel('Amplitude')
        return m

    def t_speed(self, t='mean', ret_m=False, fig=True, nm=5, tmin=2e2*tau):
        if fig is True:
            plt.figure()
        speed = self.speed()
        std = None
        if t == 'mean':
            amp = np.abs(self.fft[self.tloc(tmin):])
            data = np.average(speed[self.tloc(tmin):], axis=0, weights=amp)
            std = np.sqrt(np.average((speed[self.tloc(tmin):] - data[np.newaxis,:,:])**2, axis=0, weights=amp))
            sdata = amp.mean(axis=(0,2))
            title = '$t$ mean'
        else:
            it = self.tloc(t)
            data = speed[it]
            sdata = np.abs(self.fft[it]).mean(axis=1)
            title = '$t={0:.2f}$'.format(t)
        m = self._t_phase_plotter(data, ret_m=ret_m, std=std, sdata=sdata, nm=nm)
        plt.title(title)
        plt.ylabel('Speed')
        ylim = list(plt.ylim())
        ylim[0] = max(0, ylim[0])
        ylim[1] = min(1, ylim[1])
        plt.ylim(*ylim)
        return m

    def plot2d(self, data, *args, **kwargs):
        phi_dot = kwargs.pop('phi_dot', [0])
        pop_title = False
        if not type(data) == list:
            data = [data]
        sdir = False
        out = []
        if len(phi_dot) > 0 and kwargs.get('sdir', None) is None:
            sdir = True
        if len(args) == 1:
            bf = args[1]
        else:
            pre = args[0]
            tmp = []
            for i in args[1:]:
                try:
                    tmp.extend(i)
                except TypeError:
                    tmp.append(i)
            if len(tmp) == 1 and not tmp[0] is None:
                bf = self.loadfile(pre, tmp[0])
            else:
                try:
                    tmp[1] += 1
                except (IndexError, TypeError):
                    pass
                for fn in self.files(pre)[slice(*tmp)]:
                    bf = self.loadfile(fn)
                    for pd in phi_dot:
                        kwargs['phi_dot'] = pd
                        if sdir is True:
                            kwargs['sdir'] = '%g' % pd
                        if not 'title' in kwargs:
                            kwargs['title'] = r'$\Omega_p = {0:g},\, t/2\pi = {1:07.2f}$'.format(pd, bf.t/tau)
                            pop_title = True
                        for d in data:
                            out.append(bf.plot2d(d, **kwargs))
                        if pop_title:
                            kwargs.pop('title')
                        plt.close()
                return out
        if not hasattr(bf, 'plot2d'):
            bf = self.loadfile(bf)
        for pd in phi_dot:
            kwargs['phi_dot'] = pd
            if sdir is True:
                kwargs['sdir'] = '%g' % phi_dot
            for d in data:
                out.append(bf.plot2d(d, **kwargs))
        if len(out) == 1:
            return out[0]
        return out

    def speed_shift(self, phi_dot=.1*np.arange(10), data=None, base='cons', t0=None, t1=None,
                    mkmov=False, add_phi_dot=None, dpi=300, **kwargs):
        if data is None:
            data = ['Rpseudo', 'vorticity', 'vortensity', 'vi', 've']
        phi_dot = np.atleast_1d(phi_dot)
        if not add_phi_dot is None:
            phi_dot = np.concatenate([phi_dot, np.atleast_1d(add_phi_dot)])
        kwargs['phi_dot'] = phi_dot
        kwargs['ret_fn'] = True
        if dpi:
            tmp = kwargs.get('fopt', None)
            if tmp is None:
                kwargs['fopt'] = {'dpi': dpi}
            elif not 'dpi' in tmp:
                kwargs['fopt']['dpi'] = dpi
        if not 'save' in kwargs:
            kwargs['save'] = True
            if not 'ext' in kwargs:
                kwargs['ext'] = 'png'
        fns = self.plot2d(data, base, t0, t1, **kwargs)
        if mkmov:
            raise NotImplementedError('Need to reimplement for multiple phi-dots.')
            helpers.mkmov(fnames="")
        return None

    def diagnostic(self, rs=[-1, 1.2], save=False, fn=None, ext='png', figsize=None,
                   sdir=None, subsample=None, sz=4, xmax=2.5, dpi=300, modes=None,
                   add_modes=None, tmark=None, add_max=None):
        self.amp #make sure data is loaded
        #self.mode_mask()
        rs = np.atleast_1d(rs)
        nr = rs.size
        nx = nr + 1
        ny = 2
        if figsize is None:
            figsize = (nx * sz, ny * sz)
        if dpi:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        else:
            fig = plt.figure(figsize=figsize)
        gs = mpl.gridspec.GridSpec(ny, nx, top=.9, bottom=.1, hspace=.3)

        ropt = dict(modes=modes, add_modes=add_modes, add_max=add_max, fig=False)
        for i, r in enumerate(rs):
            if r == -1:
                r = .5 + .5 * self.rc[0]
            ax = plt.subplot(gs[0,i])
            plt.sca(ax)
            self.r_amp(r, **ropt)

            ax = plt.subplot(gs[1,i])
            plt.sca(ax)
            self.r_speed(r, tmark=tmark, **ropt)

        ax = plt.subplot(gs[0,nr])
        f = self.loadfile(self.files('cons')[-1])
        f.plot2d('Rpseudo', ax=ax, vmin='smart', cbl=r'$rv_r\sqrt{\rho}$', subsample=subsample,
                 title=r'$t/2\pi={:.2f}$'.format(f.t / tau))
        plt.xlim(-xmax, xmax)
        plt.ylim(-xmax, xmax)
        fig.suptitle('Diagnostic for ' + helpers.sanitize_lbl(self.name))

        ax = plt.subplot(gs[-1,-1])
        #Get rid of ticks and axes
        spines = [ax.spines[j] for j in ax.spines.keys()]
        for spine in spines :
            spine.set_color('none')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        # the time is now
        now = time.asctime() + ' ' + time.tzname[time.localtime().tm_isdst]
        ax.text(.5, 1, helpers.sanitize_lbl(self.name) + '\n' + now, ha='center', va='top')
        info = {}
        pars = []
        def _add(key, val):
            info[key] = val
            pars.append(key)
        # populate
        #_add('Name', self.name)
        _add(r'$\mathcal{M}$', self.mach)
        _add('$N_r$', self.rc.size)
        _add(r'$N_\phi$', self.phic.size)
        _add('$r$', '[{:.3g}, {:.3g}]'.format(self.r[0], self.r[-1]))
        _add('Seed', self.inputs['problem'].get('seed', 'random'))
        _add('Amp', '{:.3g}'.format(self.inputs['problem'].get('seedAmp', .01)))
        # print the stuff in a grid
        j = 0
        ncol = 3
        for par in pars :
            try :
                txt = '{0:s}: {1:g}'.format(par, float(info[par]))
            except (TypeError, ValueError) :
                txt = '{0:s}: {1:}'.format(par, info[par])
            ax.text(.00 + .4 * (j % ncol), 1. - .5 * .12 * (j // ncol + 3), txt)
            j += 1
        if save or fn:
            if fn is None:
                fn = self.name + '_diag.' + ext
            if sdir:
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                fn = os.path.join(sdir, fn)
            plt.savefig(fn)
            plt.close()
            return fn
        return None

    def gatherVort(self):
        out = {i: [] for i in ['vorticity', 'vortensity', 'dvorticity', 'dvortensity']}
        i = 0
        for f in self.files('cons'):
            bf = self.loadfile(f)
            tmp = bf.vorticity()
            out['vorticity'].append(tmp.mean(axis=0))
            tmp /= bf['dens']
            out['vortensity'].append(tmp.mean(axis=0))
            tmp = bf.vorticity(True)
            out['dvorticity'].append(tmp.mean(axis=0))
            tmp /= bf['dens']
            out['dvortensity'].append(tmp.mean(axis=0))
            if not i % 10:
                print(i)
            i += 1
        return {i: np.array(out[i]) for i in out.keys()}

    def stVort(self, data=None, var=None, vmin=None, vmax='smart', zerocent=None, log=False, cmap=None, popt=None,
               r_cut=None, fig=None, ax=None, fopt=None, interpolation='nearest'):
        if fopt is None:
            fopt = {}
        if popt is None:
            popt = {}
        _popt = {}
        if data is None:
            fn = os.path.join(self.path, "v_st.p")
            if os.path.isfile(fn):
                import pickle
                data = pickle.load(open(fn, "rb" ))
                #pickle.dump(data, fn, "wb"))
            else:
                data = self.gatherVort()
        if type(data) == dict:
            if var is None:
                var = 'dvortensity'
            data = data[var]
        nt = data.shape[0]
        #ext = [[0, nt], []]
        if fig is None and ax is None:
            fig = plt.figure(**fopt)
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()

        tmp = {}
        try:
            if '%' == vmin[-1]:
                tmp['low'] = float(vmin[:-1])
                vmin = 'smart'
        except TypeError:
            pass
        try:
            if '%' == vmax[-1]:
                tmp['high'] = float(vmax[:-1])
                vmax = 'smart'
        except TypeError:
            pass
        if 'smart' in [vmin, vmax]:
            rloc = slice(None)
            if r_cut:
                rloc = slice(self.rloc(r_cut), None)
            tmp = helpers.smartlim(data[:, rloc], **tmp)
            if vmin == 'smart':
                vmin = tmp[0]
            if vmax == 'smart':
                vmax = tmp[1]
        # check if zero centered data
        if zerocent is None and not log:
            zerocent = helpers.isZeroCent(data)
        if zerocent:
            if cmap is None:
                cmap = helpers.NCcmap
            if vmin is None and vmax is None:
                vmax = np.abs(data).max()
            elif vmin is None:
                vmin = -abs(vmax)
            else:
                vmax = abs(vmin)
            vmin = -vmax

        _popt.update(dict(cmap=cmap, vmin=vmin, vmax=vmax))#, interpolation=interpolation))
        _popt.update(popt)

        #plt.imshow(data.T, **_popt)
        x = np.arange(nt)[:,np.newaxis] * np.ones_like(data)
        y = self.rc[np.newaxis,:] * np.ones_like(data)
        pcm = plt.pcolormesh(x, y, data, **_popt)
        plt.colorbar()

        plt.xlabel('$t / 2\pi$')
        plt.ylabel('$r$')

    def tVort(self, data, t, save=False):
        plt.plot(self.rc, data['dvortensity'][t])
        plt.plot(self.rc, data['dvorticity'][t])
        plt.legend([r'$\delta\omega_z/\rho$', r'$\delta\omega_z$'])
        plt.xlabel('$r$')
        plt.title(r'$t/2\pi={0:d}$'.format(t))
        plt.axes().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.1))
        if save:
            fn = 'vort_t-{0:04d}.pdf'.format(t)
            plt.savefig(fn)
            plt.close()


    def _mr_plot(self, t, data, std, log=False, norm=None, dt=5, dr=.01, ext='pdf', fig=None, ax=None, save=False,
                 fn=None, cbl=None, skip_m0=None, speed=False, popt=None, title=None, amp=False, sdir=None):
        tslice = slice(t - dt, t + dt + 1)
        weights = np.minimum(np.nan_to_num(std[tslice]), 1e99) ** -2
        _data, weight = np.average(data[tslice], weights=weights, axis=0, returned=True)

        rbins = np.arange(self.rc[0] - self.rc[0] % dr, self.rc[-1], dr)
        bins = []
        for r in rbins:
            loc = np.where(np.logical_and(r <= self.rc, self.rc < r + dr))[0]
            bins.append(np.average(_data[:,loc], weights=weight[:,loc], axis=1))
        bins = np.array(bins).T
        tmp = bins[np.isfinite(bins)]
        bmax = tmp.max()
        bmin = tmp.min()

        if title is None:
            title = helpers.sanitize_lbl(self.name) + " $t/2\pi={0:g}\pm{1:g}$".format(t, dt)


        if log and norm is None:
            norm = mpl.colors.LogNorm()
        extent = [rbins[0], rbins[-1], -.5, data.shape[1] - .5]

        if speed:
            if skip_m0 is None:
                skip_m0 = True
            bins[bins > 1] = np.nan
            bins[bins < 0] = np.nan
            if 'vmax' not in popt and bmax > .9:
                popt['vmax'] = .9
            if 'vmin' not in popt and bmin < .2:
                popt['vmin'] = .2
        if skip_m0:
            extent[2] = .5
            bins = bins[1:,:]

        if amp:
            if 'vmax' not in popt and bmin > .1:
                popt['vmax'] = .1
            if 'vmin' not in popt and bmax < 1e-5:
                popt['vmin'] = .1e-5

        if popt is None:
            popt = {}
        _opt = dict(extent=extent, norm=norm, interpolation='nearest')
        _opt.update(popt)

        if fig is None and ax is None:
            fig = plt.figure()
        if ax is None:
            ax = plt.gca()

        im = ax.imshow(bins, **_opt)
        cb = plt.colorbar(im)
        if cbl:
            cb.set_label(cbl)
        plt.xlabel('$r$')
        plt.ylabel('$m$')
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        if title:
            plt.title(title)
        plt.axvline(1, lw=1, c='k', ls=':')

        if fn and save is None:
            save = True
        if save:
            if sdir is None:
                sdir = ''
            if fn is None:
                fn = helpers.sanitize_lbl(self.name) + '_mr.' + ext.lstrip('.')
                fn = os.path.join(sdir, fn)
            if sdir:
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
            fig.savefig(fn)
            plt.close(fig)

    def mr_speed(self, ts, log=False, norm=None, dt=5, dr=.01, ext='pdf', fig=None, ax=None, save=False, fn=None,
                 cbl=None, sdir=None, **kwargs):
        if sdir is True:
            sdir = 'mr_speed'
        if sdir:
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
        tlist = np.atleast_1d(ts)
        for t in tlist:
            if fn is None:
                _fn = helpers.sanitize_lbl(self.name) + '_mr_speed_{0:05d}.'.format(t) + ext.lstrip('.')
                _fn = os.path.join(sdir, _fn)
            else:
              _fn = fn
            if cbl is None:
                cbl = r'$\Omega_{\rm p}$'
            opt = dict(log=log, norm=norm, dt=dt, dr=dr, ext=ext, fig=fig, ax=ax, save=save, fn=_fn, cbl=cbl, speed=True,
                       popt=kwargs)
            self._mr_plot(t, self.speed, self.fft_data._speed_std, **opt)

    def mr_amp(self, ts, log=True, norm=None, dt=5, dr=.01, ext='pdf', fig=None, ax=None, save=False, fn=None,
                 cbl=None, **kwargs):
        tlist = np.atleast_1d(ts)
        for t in tlist:
            if fn is None:
                _fn = helpers.sanitize_lbl(self.name) + '_mr_amp_{0:05d}.'.format(t) + ext.lstrip('.')
            else:
              _fn = fn
            if cbl is None:
                cbl = r'$|A_m|$'
            opt = dict(log=log, norm=norm, dt=dt, dr=dr, ext=ext, fig=fig, ax=ax, save=save, fn=_fn, cbl=cbl, amp=True,
                       popt=kwargs)
            self._mr_plot(t, self.amp, self.fft_data._amp_std, **opt)

    def my_fft_plots(self, save=True, quiet=False, diag=True, sdir=None):
        if diag:
            self.diagnostic(save=save, ext='png')
        self.mr_speed(range(100, int(self.fft_time[-1] / tau + .5) + 10, 100), save=1, sdir=sdir)
        if not quiet:
            print('Consider using the following:')
            print('    sim.speed_plots(modes)')
            print('    sim.get_speed(m, t0)')

    def speed_plots(self, modes, rin=-1, rout=1.2, save=True, tmark=None):
        if rin == -1:
            rin = self.rc[0] * .5 + .5
        if save:
            self.r_speed(rin, modes=modes, tmark=tmark, fn=self.name + '_rin.pdf')
            self.r_speed(rout, modes=modes, tmark=tmark, fn=self.name + '_rout.pdf')
        else:
            self.r_speed(rin, modes=modes, tmark=tmark)
            self.r_speed(rout, modes=modes, tmark=tmark)

    def get_speed(self, m, t0, dt=50, r=-1, dr=10, fmt='.3f'):
        if r == -1:
            r = self.rc[0] * .5 + .5
        rl = self.rloc(r)
        loc = [slice(t0, t0 + dt), m, slice(rl, rl + dr)]
        out = np.average(self.speed[loc], weights=self.fft_data._speed_std[loc]**-2)
        if fmt:
            print(('{0:' + fmt + '}').format(out))
        return out

    def mk_maps(self, var_list=['Rpseudo', 've'], dt=25, base_dir=None, file='cons', popt=None):
        if popt is None:
            popt = {}
        var_list = np.atleast_1d(var_list)
        fopt={'dpi': 300, 'figsize': (6,6)}
        #path = self.name + '_maps'
        path = 'maps'
        if base_dir is not None:
            path = os.path.join(base_dir, path)
        i = 0
        if not os.path.isdir(path):
            os.makedirs(path)
        while True:
            if i - 1 <= self.fft_time[-1] / tau:
                print('Map of t/orb={:d}'.format(i))
            try:
                for var in var_list:
                    if var is not None:
                        var = str(var)
                    if not os.path.isdir(os.path.join(path, var)):
                        os.makedirs(os.path.join(path, var))
                    self.loadfile(file, i).plot2d(var, sdir=os.path.join(path, var), save=True, fopt=fopt, **popt)
            except IndexError:
                break
            i += dt

    def mode_detect(self, r=None, save=True, fn=None, dt=10, nbin=3, emax=1e-4, smax=2e-4, dr=5,
                    data_only=False, dw=.05, overlap=10, nskip=3, out_mult=2):
        if (not data_only) and (self._mode_detect is not None):
            return self._mode_detect
        if r is None:
            r = [.5 + .5 * self.rc[0], 1.2]
        r = np.atleast_1d(r)
        ris = map(self.rloc, r)
        dr = max(0, int(dr))
        tlist = [0]
        t = self.fft_time
        while tlist[-1] < t[-1]:
            tlist.append(tlist[-1] + dt * tau)
        tlist = np.array(tlist)
        ti = [0]
        ti += [t[t < tlist[i+1]].argmax() for i in range(len(tlist) - 1)]
        tslice = [slice(ti[i], ti[i+1]+1) for i in range(len(tlist) - 1)]
        fits = np.empty((len(ris), len(ti) - 1, self.speed.shape[1], 6))
        weights = np.minimum(np.nan_to_num(self.fft_data._speed_std), 1e99) ** -2
        for i, ri in enumerate(ris):
            s, w = np.average(self.speed[:,:,ri-dr:ri+dr+1], weights=weights[:,:,ri-dr:ri+dr+1], axis=2, returned=True)
            for j in range(len(ti) - 1):
                for m in range(self.speed.shape[1]):
                    fits[i,j,m,0] = np.average(s[tslice[j], m], weights=w[tslice[j], m])
                    fits[i,j,m,1:] = linregress(t[tslice[j]], s[tslice[j], m])
        r = np.array([self.rc[i] for i in ris])
        mult = np.ones_like(r)
        mult[r > 1] = out_mult
        mult = mult[:, np.newaxis, np.newaxis]
        mask = np.logical_and(fits[:,:,:,0] < 1, fits[:,:,:,0] > 0)
        mask = np.logical_and(mask, np.abs(fits[:,:,:,1]) < smax * mult)
        mask = np.logical_and(mask, np.abs(fits[:,:,:,5]) < emax * mult)
        if nskip:
            mask[:,:nskip,:] = 0
        run = np.maximum(boxcar(mask, nbin, axis=1), boxcar(mask[:,::-1,:], nbin, axis=1)[:,::-1,:])
        data = dict(t=tlist, r=r, fits=fits, mask=np.logical_not(mask), run=run, tlist=tlist)
        if data_only:
            return data
        self._mode_detect = modeData(data, sim=self, dw=dw, overlap=overlap, nbin=nbin)
        return self._mode_detect

    def main_plots(self, maps=False, fluxes=True, working_dir=None, quiet=False):
        if working_dir is True:
            working_dir = self.name + '_plots'
        if not working_dir:
            working_dir = os.getcwd()
        if working_dir:
            if not os.path.isdir(working_dir):
                os.mkdir(working_dir)
        pwd = os.getcwd()
        try:
            os.chdir(working_dir)
            if not quiet: print('    Mode detect')
            md = self.mode_detect()
            md.write()
            md.plot(save=True)
            gmodes = list({int(m[0]) for m in md.g_modes()})
            t = [(.5 * (m[1] + m[2]) / tau, m[0]) for m in md.g_modes()]
            if not quiet: print('    Diagnostic')
            self.diagnostic(save=True, add_modes=gmodes, add_max=1, tmark=t[:])
            if not quiet: print('    My fft')
            self.my_fft_plots(diag=False, quiet=True, sdir=True)
            if gmodes:
                if not quiet: print('    Speed plots')
                self.speed_plots(gmodes, tmark=t[:])
            if maps:
                if not quiet: print('    Maps')
                self.mk_maps()
            if fluxes:
                if not quiet: print('    Flux Series')
                self.flux_series(sdir=True, progress=(not quiet))
        finally:
            os.chdir(pwd)

    def parse_func(self, func, *args, **kwargs):
        return getattr(self, func)(*args, **kwargs)


class modeData(object):
    def __init__(self, data, sim=None, dw=.05, overlap=10, nbin=3):
        self.sim = sim
        self.dw = dw
        self.nbin = nbin
        self._dt = overlap * tau
        mask = np.logical_not(data['mask'])
        run = data['run']
        r = data['r']
        fits = data['fits']
        tlist = data['tlist']
        self.r = r
        out = [[] for i in r]
        for z in zip(*np.where(run == nbin)):
            w = fits[z[0], z[1], z[2], 0]
            t0 = z[1]
            t1 = t0
            tmp = [i for i in out[z[0]] if (i[0] == z[2]) and (i[1] <= tlist[t0] <= i[2])]
            #if not tmp:
            if (not tmp) and (t0 < fits.shape[1] - 1):
                while (run[z[0], t1 + 1, z[2]] == nbin) and (abs(fits[z[0], t1 + 1, z[2], 0] - w) < dw * w):
                    t1 += 1
                    w = fits[z[0], t0:t1, z[2], 0].mean()
                    if t1 >= run.shape[1] - 1:
                        break
                if t1 - t0 >= nbin:
                    t0 = tlist[t0]
                    t1 = tlist[t1+1]
                    out[z[0]].append([z[2], t0, t1, w])
        self.mode_data = [np.array(i) for i in out]

    def filter(self, data=None):
        if data is None:
            try:
                return self._filter[:]
            except AttributeError:
                self._filter = [self.filter(i) for i in self.md]
                return self._filter[:]
        tmp = sorted(data, key=lambda x:x[0] - 1e-6 * (x[2] - x[1]))
        out = []
        for mode in tmp:
            similar = [i for i in out if (mode[0] == i[0]) and (abs(mode[3] - i[3]) < self.dw * i[3])]
            if not similar:
                out.append(mode)
        return np.array(out)

    def write(self, fn=None):
        if fn is None:
            fn = self.sim.name + '_modes.csv'
        rs = ["R = "+repr(i) for i in self.r]
        rs.append('Global Modes')
        data = self.filter()
        data.append(self.filter(self.g_modes()))
        out = ['# m, t_start, t_end, speed','']
        for i, r in enumerate(rs):
            out.append('# ' + r)
            out += ['{0:d}, {1:.2f}, {2:.2}, {3:.3f}'.format(int(j[0]), *j[1:]) for j in data[i]]
            out.append('')
        with open(fn, 'w') as f:
            f.write('\n'.join(out))
        return

    def plot(self, save=False, fn=None, ext='pdf', inc_global=True):
        markers = 'o','+','x','.'
        lbls = []
        for i, r in enumerate(self.r):
            plt.scatter(self.filter()[i][:,0], self.filter()[i][:,3], marker=markers[i])
            lbls.append(r'$r={0:.2f}$'.format(r))
        if inc_global:
            data = self.g_modes()
            plt.scatter(data[:,0], data[:,3], marker='*')
            lbls.append('Global')
        plt.legend(lbls)
        plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        plt.xlabel('mode')
        plt.ylabel(r'$\Omega_{\rm p}$')
        if save or fn:
            if fn is None:
                fn = self.sim.name + '_dispersion.' + ext
            plt.savefig(fn)
            plt.close()
        return


    def __getitem__(self, item):
        return self.mode_data[item]

    @property
    def md(self):
        return self.mode_data

    def g_modes(self):
        try:
            return self._g_modes
        except AttributeError:
            out = []
            for mode in self[0]:
                keep = True
                t0 = mode[1]
                t1 = mode[2]
                for r in self[1:]:
                    try:
                        mlist = [i for i in r if (i[0] == mode[0]) and (abs(i[3] - mode[3]) < self.dw * mode[3])]
                    except IndexError:
                        mlist = []
                    while mlist:
                        if min(mlist[0][2], mode[2]) - max(mlist[0][1], mode[1]) < self._dt:
                            mlist.pop(0)
                        else:
                            break
                    if not mlist:
                        keep = False
                        break
                    t0 = max(t0, mlist[0][1])
                    t1 = min(t1, mlist[0][2])
                if keep:
                    out.append([mode[0], t0, t1, mode[3]])
            self._g_modes = self.filter(np.array(out))
        return self._g_modes






class auxBLsim(BLsim):
    def mode_plot(self, data=None, cb=True, title=None, cbl=None, vmin=0,
                  vmax=20, main_plots=False, mpopt={}, save=False, fn=None,
                  ext='pdf', sdir=None, fig=None, fopt={}, ax=None):
        _mpopt = {'ext':ext, 'sdir':sdir}
        _mpopt.update(mpopt)
        if data is None:
            if self._mode_phase is None or main_plots:
                data = self.mode_phase(main_plots=main_plots, mpopt=_mpopt)[0]
            else:
                data = self.mode_phase()[0].argmax(axis=1)
        one = np.ones((self.times.size + 1, self.r.size))
        r = self.r[np.newaxis, :] * one
        t = np.concatenate((self.times, np.array([self.times[-1] + 1.])))[:,np.newaxis] * one
        t -= .5
        cmap = plt.get_cmap(lut = vmax - vmin + 1)
        if fig is None and ax is None:
            fig = plt.figure(**fopt)
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        fig = plt.gcf()
        pcm = plt.pcolormesh(t, r, data, vmin=vmin, vmax=vmax, cmap=cmap)
        if self.name == 'test_run':
            plt.axvline(101.5, c='k', ls=':', lw=1)
        ax = plt.gca()
        if title is None:
            title = self.name + ' dominant mode'
        if title:
            plt.title(helpers.sanitize_lbl(title))
        plt.xlabel('Time')
        plt.ylabel('Radius')
        if cb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(pcm, cax=cax)
            cb.ax.yaxis.set_offset_position('left')
            labels = np.arange(0, vmax + 1, 5)
            #scale = (vmax - vmin) / (vmax - vmin + 1)
            loc = (labels + .5 ) * (vmax - vmin) / (vmax - vmin + 1)
            cb.set_ticks(loc)
            cb.set_ticklabels(labels)
            #cb.ax.minorticks_on()
            #cb.ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
            #cb.ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            if cbl:
                cb.set_label(cbl)
        if save or fn:
            if fn is None:
                fn = self.name + '_ST_mode.' + ext
            fig.savefig(fn)
            plt.close()

    def mt_plot(self, r, fn=None, mmax=30, save=False, ext='pdf', sdir=None,
                fig=None, ax=None, fopt={}, vmin='smart', vmax='max', cb=True,
                cbl=None, popt={}, log=True):
        ir = np.abs(self.rc - r).argmin()
        r = self.rc[ir]
        _amp = self.mode_phase()[0]
        data = _amp[:,:,ir].T

        if 'smart' in [vmin, vmax]:
            smart = helpers.smartlim(_amp[:,:mmax+1,:])
            if vmin == 'smart':
                vmin = smart[0]
            if vmax == 'smart':
                vmax = smart[1]
        if vmin == 'min':
            vmin = _amp[:,:mmax+1,:].min()
        if vmax == 'max':
            vmax = _amp[:,:mmax+1,:].max()
        if vmin == 'auto':
            vmin = max(data.min(), 1e-6)
        _opt = {'vmin': vmin, 'vmax': vmax, 'interpolation': 'nearest'}
        if log:
            _opt['norm'] =  mpl.colors.LogNorm()
        _opt.update(popt)

        if fig is None and ax is None:
            fig = plt.figure(**fopt)
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        fig = plt.gcf()

        plt.imshow(data, **_opt)
        plt.ylim(None, mmax)
        if self.name == 'test_run':
            plt.axvline(101.5, c='k', ls=':', lw=1)
        plt.xlabel('Time')
        plt.ylabel('Mode')
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:.2f}$'.format(r))
        #plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:}$'.format(helpers.eformat(r, math=False)))
        if cb:
            cb = plt.colorbar()
            if cbl is None:
                cbl = r'$\left|a_m\right|^2$'
            if cbl:
                cb.set_label(cbl)
        if save or fn:
            if log:
                log = 'log'
            else:
                log = 'lin'
            if fn is None:
                fn = self.name + '_mode-time_r={0:.2e}_{1:}.'.format(r, log) + ext
            fig.savefig(fn)
            plt.close()

    def cross_corr(self, t1, t2=None, var='pseudo', dt=None, plot=False,
                   norm=True, save=False, ext='png'):
        if t2 is None:
            t2 = t1 + 1
        try:
            tmp = t2 - t1
        except:
            tmp = None
        if not hasattr(t1, 'data'):
            t1 = self.loadfile(t1)
        if not hasattr(t2, 'data'):
            t2 = self.loadfile(t2)
        if dt is None:
            try:
                dt = t2.data['t'] - t1.data['t']
            except KeyError:
                if not tmp is None:
                    dt = tau * tmp
                else:
                    dt = tau
        d1 = t1.get2d(var)
        d2 = t2.get2d(var)
        out = np.empty(d1.shape)
        for ir in xrange(d1.shape[1]):
            #out[:,ir] = helpers.fftcorrelate(d2[:,ir], d1[:,ir])
            out[:,ir] = helpers.c_correlate(d2[:,ir], d1[:,ir])
        #if norm:
            #out /= np.abs(out).max(axis=0)[np.newaxis,:]
            #out /= np.sqrt(.5 * d1**2 + .5 * d2**2).max(axis=0)[np.newaxis,:]
        if plot:
            title = 'cross_corr({0:d}, {1:d})'.format(t1.t, t2.t)
            fn = None
            if save:
                fn = self.name + '_cc_{:d}-{:d}.'.format(t1.t, t2.t) + ext
            t1.plot2d(out, title=title, vmax=1.1, phi_shift=np.pi, save=save, fn=fn)
        return out

    def _verify_fft_tar(self):
        tar = os.path.join(self.path, 'fft.tar')
        if not os.path.isfile(tar):
            return False
        #n = len(glob(os.path.join(self.path, 'disk.FT*.athdf')))
        hash = os.path.join(self.path, 'hash')
        if not os.path.isfile(hash):
            return False
        cksum = os.path.join(self.path, 'cksum')
        if not os.path.isfile(cksum):
            return False
        m = 0
        with open(hash, 'r') as a:
            with open(cksum, 'r') as b:
                while True:
                    al = a.readline()
                    bl = b.readline()
                    if al != bl:
                        return False
                    if (not al) and (not bl):
                        break
                    m +=1
        #if m != n:
        #    return False
        return True


def refreshSim(sim):
    attr = ['coarse_data', 'fft_time', 'mode_detect']
    opt = {i: getattr(sim, '_' + i, None) for i in attr}
    return BLsim(sim.path, **opt)

######################
# End of BLsim class #
######################

def parallel_compile(func, arglist, T=None):
    '''Usage : parallel_compile(func, arglist=None, T=None)
    Similar to comp_wrapper, but strings in arglist are not automatically turned
    into zeussim_extended class instances.'''
    out = parmap(func, arglist)
    out = filter(None, out)
    if T : out = zip(*out)
    return out

def comp_wrapper(func, simlist=None, include=None, tmin=200, T=False, args=None, kwargs=None):
    '''Usage : comp_wrapper(func, simlist=None, include=None, tmin=40, T=False, load_eos=False)
    Evaluate function 'fun' on each simulation in 'simlist' and return the
    compiled result, and use parallel processing to do so.

    Keyword include (None):
    Only simulations for which include(sim) are true will be
    included in the output. If include=None then include defaults to the following
    function: sim.t[-1] >= tmin

    Keyword T (False):
    Whether to transpose the output.

    Keyword load_eos (False):
    Whether to load EOS before computation.
    '''
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if not simlist:
        dirs = [i for i in _dirs[::-1] if os.path.isdir(i)]
        while True:
            tmp = glob(os.path.join(dirs[0], '*/athinput.*'))
            if tmp:
                break
            dirs.pop(0)
        simlist = sorted(set(os.path.split(i)[0] for i in tmp))

    sims = []

    #if not simlist: simlist = simlist4

    if tmin == None: tmin = 200

    if include == None:
        def include(sim):
            return sim.fft_time[-1] >= tmin
    elif include == True:
        def include(sim):
            return True

    def mapper(sim):
        # if sim is a string
        try:
            sim.rstrip()
            name = sim
            sim = BLsim(sim)
        # otherwise assume it's a simulation
        except AttributeError:
            name = sim.name
        # if we get here there's no hope
        except:
            print('Bad sim (no load): ' + name)
            print(traceback.format_exc())
            return None

        try:
            if include(sim):
                if not callable(func):
                    return sim.parse_func(func, *args, **kwargs)
                return func(sim)
        except KeyboardInterrupt:
            raise
        except:
            print('Bad sim (func err): ' + name)
            print(traceback.format_exc())
        return None

    return parallel_compile(mapper, arglist=simlist, T=T)

def mkplots(simlist=None, maps=False):
    comp_wrapper('main_plots', simlist=simlist, kwargs=dict(quiet=True, working_dir=True, maps=maps))

def _old_mkplots(sims=None, path='', ext='png'):
    if sims is None:
        sims = [i for i in glob(os.path.join(path, '*')) if os.path.isdir(i)]
    cwd = os.getcwd()
    for sim in sims:
        try:
            if not hasattr(sim, 'name'):
                sim = BLsim(sim)
            print('\nPlotting Simulation {0:}'.format(sim.name))
            if not hasattr(sim, 'path'):
                sim = BLsim(os.path.join(path, sim))
            if path:
                tmp = os.path.split(sim.path)[-1]
                if tmp:
                    if not os.path.isdir(tmp):
                        os.mkdir(tmp)
                os.chdir(tmp)
            else:
                os.chdir(sim.path)
            sdir = os.getcwd()
            opt = dict(save=True, ext=ext, sdir=sdir)
            sim.mode_plot(main_plots=True, **opt)
            rdir = os.path.join(sdir, 'modes_at_r')
            if rdir:
                if not os.path.isdir(rdir):
                    os.mkdir(rdir)
            #opt['sdir'] = rdir
            os.chdir(rdir)
            for r in [.95,1,1.1,1.5,2,3]:
                sim.mt_plot(r, log=True, **opt)
                sim.mt_plot(r, log=False, vmin=0, **opt)
            os.chdir(cwd)
            print('Finished simulation {0:}'.format(sim.name))
        except KeyboardInterrupt:
            raise
        except:
            os.chdir(cwd)
            name = getattr(sim, 'name', sim)
            print('\n!!! Error\nUnable To finish Simulation {0:}.'.format(name))

if __name__ == '__main__':
    path = os.path.expanduser('~/BLayer')
    tmp = os.path.join(path, 'Mach{0:}stampede')
    d = '[0-9]'
    sims = glob(tmp.format(d)) + glob(tmp.format(d*2))
    print(sims)
    #path = '/home/mcoleman/data/perseus_data/'
    #sims = None
    mkplots(sims, path=path)
