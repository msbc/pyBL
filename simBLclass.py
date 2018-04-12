#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
#from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
#import h5py
#from mayavi import mlab
import numpy as np
#import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import scoreatpercentile as percentile
from scipy.signal import gaussian
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

from . import athena_read as ar
from . import helpers


tau = 2 * np.pi

#mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb,amsmath}"]

_dirs = ['', '~/', '~/Dropbox/dev/pyBL', '/scratch/gpfs/sashaph/BLayer', '/perseus/scratch/gpfs/sashaph/BLayer', '~/BLayer', '~/BLayer/fft_tests']
_dirs = list(map(os.path.expanduser, _dirs))
_dirs += [os.path.join(d, 'Mach8stampede') for d in _dirs]
_data_base = '/scratch/gpfs/sashaph/BLayer'
_pre = ['BL', 'disk', 'mock']
_i = map(str, range(1,5))
_ext = ['athdf', 'npy']
#_file_fmts = ['BL.out2.%5.5d.athdf', 'disk.out1.%5.5d.athdf']
_int_fmt = '%5.5d'
_file_fmts = ['.'.join([a, 'out' + b, _int_fmt, c]) for a in _pre for b in _i for c in _ext]

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

def mod_grad(data, mod=1, axis=0):
    loc = (slice(None),) * axis
    out = np.zeros_like(data)
    tmp = (data[loc + (slice(1, None),)] - data[loc + (slice(None, -1),)]) % mod
    out[loc + (slice(1, None),)] = .5 * tmp
    out[loc + (slice(None, -1),)] += .5 * tmp
    out[loc + (0,)] *= 2
    out[loc + (-1,)] *= 2
    return out

def grad(t, data, axis=0):
    nd = len(data.shape)
    if axis == -1:
        axis += nd
    loc = (slice(None),) * axis
    l = loc + (slice(None, -2),)
    c = loc + (slice(1, -1),)
    r = loc + (slice(2, None),)
    fill = [np.newaxis] * nd
    fill[axis] = slice(None)
    fill = tuple(fill)
    Dinv =  (t[r[-1]] - t[l[-1]])[fill]**-1
    dl = (t[c[-1]] - t[l[-1]])[fill]
    dr = (t[r[-1]] - t[c[-1]])[fill]
    rat = dr / dl
    #norm = (dl * dr * D)**-2
    out = np.zeros_like(data)
    out[c] = (dl**-1 - dr**-1) * data[c] + Dinv / rat * data[r] - rat * Dinv * data[l]
    #data[c] *= norm
    out[loc + (0,)] = (data[loc + (1,)] - data[loc + (0,)]) / dl[loc + (0,)]
    out[loc + (-1,)] = (data[loc + (-2,)] - data[loc + (-1,)]) / dr[loc + (-1,)]
    return out

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

def _parse_file(fn):
    ext = fn.split('.')[-1]
    if ext == 'athdf':
        return BLfile(fn)
    if ext == 'npy':
        return np.load(fn)[()]
    raise IOError('Cannot identify file type of "{0:}"'.format(fn))

def intr(dr, data, axis=-1):
    if axis == -1:
        axis += len(data.shape)
    loc = [np.newaxis] * axis + [slice(None)]
    return (data * dr).sum(axis=axis)

class BLfile(object):
    def __init__(self, fn, sim_path=None, t=None, trim=True, data=None,
                 defvar=None, ai_data=None, sim=None):
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
        self.data = ar.athdf(self.fn)
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
        for i in [self._default_var, 'pseudo', 'dens', 'FT-mag', 'FT-Re']:
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
            out = self.data[key]
            #return super().__getitem__(key)
        except KeyError:
            out = self._parse_self(key)
        if self._Qtrim:
            return self._trim(out)
        return out

    def intr(self, data, axis=-1):
        data = self._parse_data(data)
        return intr(self.dr, data, axis=axis)

    def ddphi(self, data, axis=0):
        data = self._parse_data(data)
        return (np.roll(data, -1, axis=axis) - np.roll(data, 1, axis=axis)) / (self.phic[2] - self.phic[0])

    def fft(self, data, axis=-2, mag=False):
        try:
            data.shape
        except AttributeError:
            data = self[data]
        out = np.fft.rfft(data, axis=axis)
        if mag:
            out = np.absolute(out)
        return out

    def rloc(self, r):
        return np.abs(r - self.rc).argmin()

    def plot2d(self, data=None, fn=None, save=False, subsample=False, title=None,
               name=None, ext='pdf', popt=None, cb=True, cbl=None, zerocent=None,
               vmin=None, vmax=None, cmap=None, cbopt=None, fig=None, fopt=None,
               ax=None, log=False, aspect=1, sdir=None, smooth=None,
               phi_shift=0, r_cut=None, phi_dot=0, ret_fn=False):
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

class BLConsPrim(BL3Dfile):
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
        s = tmp.sum(axis=0)
        loc = sorted(range(tmp.shape[0]), key=lambda x: 1/s[x])
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
        nm = 5
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
                i = self._ai_data['meshblock']['nx2'] - 1
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

############################
# End of BLfile subclasses #
############################

def loadBLfile(fn, **kwargs):
    fn = _findAbsPath(fn, kwargs.get('sim_path', None))
    data = _parse_file(fn)
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
        # course unwrap
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
    def __init__(self, path, fmts=None, fft_data=None, fft_time=None,
                 athinput=None, mode_mask=None, main_modes=None,
                 phase_angle=None):
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
            self.mach = self.inputs['hydro']['iso_sound_speed']
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
            for out in outs:
                files = []
                b = self.inputs[out].get('id', 'out' + out[6:])
                searches = ['.'.join([a, b, c, ext]) for ext in _ext]
                for search in searches:
                    files += [os.path.split(i)[-1] for i in glob(os.path.join(path, search))]
                self.fileDict[out] = sorted(files)
                var = self.inputs[out].get('variable')
                if varlist.count(var) == 1:
                    self.varDict[var] = out
                self.idDict[b] = out

        #for attr in ['r', 'phi', 'rc', 'phic']:
        #    setattr(self, attr, getattr(tmp, attr))
        self._fft_data = fft_data
        self._fft_time = fft_time
        self._phase_angle = phase_angle
        self._mode_mask = mode_mask
        self._sigmas = [0,4]
        self._main_modes = main_modes

        return None
        # End init

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

    def FFT_composite(self):
        ffts = [out for out in self.fileDict.keys()
                if self.inputs.get(out, {}).get('variable') == "FT-Range"]
        return CompositeFFTSet([FFTset(self.fileDict[i], sim=self) for i in ffts])

    def _collect_fft_data(self):
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

    def _load_fft_data(self):
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
            plt.plot(self.rc, (a-b)/a, lw=1)
        plt.ylim(-5e-3, 5e-3)
        plt.legend(varlist)
        plt.title('Time/$2\pi={0:.1f}$'.format(bf.t / tau))

    def intr(self, data, axis=-1):
        return intr(self.dr, data, axis=axis)

    @property
    def fft(self):
        if not self._fft_data is None:
            return self._fft_data
        return self._collect_fft_data()[1]

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

    def __repr__(self):
        return '<BLsim "{0:}">'.format(self.name)

    def _gen_fft_time(self):
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
    def fft_time(self):
        if not self._fft_time is None:
            return self._fft_time
        return self._gen_fft_time()

    @property
    def phase_angle(self):
        if not self._phase_angle is None:
            return self._phase_angle
        if self._fft_data is None:
            return self._collect_fft_data()[2]
        self._phase_angle = self._unwrap()
        return self._phase_angle

    def rloc(self, r):
        return np.abs(r - self.rc).argmin()

    def tloc(self, t):
        return np.abs(t - self.fft_time).argmin()

    def loadfile(self, fn, index=None):
        if not index is None:
            fn = self.files(fn)[index]
        if fn in self.filenames:
            return loadBLfile(fn, sim_path=os.path.abspath(self.path), sim=self, ai_data=self.inputs)

    def mode_mask(self, data=None, info=False, amin=None):
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

    def mode_phase(self):
        DeprecationWarning('This function (mode_phase) is deprecated.')
        out = np.array([np.abs(self.fft), self.phase_angle])
        return out

    def prop_speed(self, dt=None, ir=None):
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

    def main_modes(self, nm=None, skip_zero=True):
        if self._main_modes is None:
            fft = self.fft * self.rc[np.newaxis, np.newaxis, :]
            nt = fft.shape[0]
            a = self.intr(np.abs(fft[nt//2:]).sum(axis=0))
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
        mask = self.mode_mask()[:,:,ir]
        data = np.ma.array(self.phase_angle[:,:,ir], mask=mask)
        m = self._r_phase_plotter(r, data, ret_m=ret_m)
        plt.ylabel('Phase')
        return m

    def _old_r_speed(self, r, ret_m=False, fig=True):
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

    def _old_r_amp(self, r, ret_m=False, fig=True):
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

    def _r_phase_plotter(self, r, data, modes=None, nm=5, add_modes=None,
                         ret_m=None, smooth=False, sw=20):
        ir = self.rloc(r)
        if modes is None:
            modes = self.main_modes(nm=nm)[::-1]
        if not add_modes is None:
            add_modes = list(np.atleast_1d(add_modes))
            modes += add_modes
        handles = {}
        if len(data.shape) == 3:
            rdata = data[:,:,ir]
        else:
            rdata = data
        for m in modes:
            line = rdata[:,m]
            if smooth:
                if smooth in [True, 1]:
                    smooth = 'flat'
                line = helpers.smooth(line, window=smooth, window_len=sw)
            handles[m] = plt.plot(self.fft_time / tau, line, lw=1)[0]
        opt = {'loc': 0, 'frameon': True, 'handlelength': .7, 'prop': {'size':8}, 'ncol': 3}
        modes.sort()
        handles = [handles[m] for m in modes]
        lbls = ['$%d$' % m for m in modes]
        leg = plt.legend(handles, lbls, **opt)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.xlabel(r'Time/$2\pi$')
        #plt.ylabel('Phase')
        plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:.2f}$'.format(r))

    def r_speed(self, r, fig=True, **kwarg):
        if fig is True:
            plt.figure()
        if not 'smooth' in kwarg:
            kwarg['smooth'] = 'flat'
            if not 'sw' in kwarg:
                kwarg['sw'] = 41
        ir = self.rloc(r)
        data = self.prop_speed(ir=ir)
        self._r_phase_plotter(r, data, **kwarg)
        ylim = list(plt.ylim())
        ylim[0] = 0
        ylim[1] = 1
        plt.ylim(*ylim)
        plt.ylabel('Speed')
        return None

    def r_amp(self, r, fig=True, **kwarg):
        if not 'smooth' in kwarg:
            kwarg['smooth'] = 'flat'
            if not 'sw' in kwarg:
                kwarg['sw'] = 41
        if fig is True:
            plt.figure()
        ir = self.rloc(r)
        data = np.abs(self.fft[:,:,ir])
        self._r_phase_plotter(r, data, **kwarg)
        nt = self.fft_time.size
        m0 = data[:nt//5, 1:].max()
        m1 = data[nt//5:, 1:].max()
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
        speed = self.prop_speed()
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
                            kwargs.pop['title']
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
            data = ['Rpseudo', 'vorticity', 'vortensity']
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

    def diagnostic(self, rs=[.8,1.3], save=False, fn=None, ext='pdf', figsize=None,
                   sdir=None, subsample=None, sz=4):
        self.fft #make sure data is loaded
        self.mode_mask()
        rs = np.atleast_1d(rs)
        nr = rs.size
        nx = nr + 1
        ny = 2
        if figsize is None:
            figsize = (nx * sz, ny * sz)
        fig = plt.figure(figsize=figsize)
        gs = mpl.gridspec.GridSpec(ny, nx, top=.9, bottom=.1, hspace=.3)

        for i, r in enumerate(rs):
            ax = plt.subplot(gs[0,i])
            self.r_amp(r, fig=False)

            ax = plt.subplot(gs[1,i])
            self.r_speed(r, fig=False)

        ax = plt.subplot(gs[0,nr])
        f = self.loadfile(self.files('cons')[-1])
        f.plot2d('Rpseudo', ax=ax, vmin='smart', cbl=r'$v_r\sqrt{\rho}$', subsample=subsample)
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
        ax.text(.5, 1, now, ha='center', va='top')
        info = {}
        pars = []
        def _add(key, val):
            info[key] = val
            pars.append(key)
        # populate
        _add('Name', self.name)
        _add('$N_r$', self.rc.size)
        _add(r'$N_\phi$', self.phic.size)
        _add('$r$', '[{:g}, {:g}]'.format(self.r[0], self.r[-1]))
        _add(r'$\mathcal{M}$', 1. / self.mach)
        # print the stuff in a grid
        j = 0
        ncol = 3
        for par in pars :
            try :
                txt = '{0:s}: {1:g}'.format(par, float(info[par]))
            except (TypeError, ValueError) :
                txt = '{0:s}: {1:}'.format(par, info[par])
            ax.text(.03 + .35 * (j % ncol), 1. - .5 * .12 * (j // ncol + 2), txt)
            j += 1
        if save or fn:
            if fn is None:
                fn = self.name + '_diag.' + ext
            if not sdir is None:
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

def refreshSim(sim):
    attr = ['fft_data', 'fft_time', 'main_modes', 'mode_mask', 'phase_angle']
    opt = {i: getattr(sim, '_' + i, None) for i in attr}
    return BLsim(sim.path, **opt)

######################
# End of BLsim class #
######################

def mkplots(sims=None, path='', ext='png'):
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
                if not os.path.isdir(tmp):
                    os.mkdir(tmp)
                os.chdir(tmp)
            else:
                os.chdir(sim.path)
            sdir = os.getcwd()
            opt = dict(save=True, ext=ext, sdir=sdir)
            sim.mode_plot(main_plots=True, **opt)
            rdir = os.path.join(sdir, 'modes_at_r')
            if not os.path.isdir(rdir):
                os.mkdir(rdir)
            #opt['sdir'] = rdir
            os.chdir(rdir)
            for r in [.95,1,1.1,1.5,2,3]:
                sim.mt_plot(r, log=True, **opt)
                sim.mt_plot(r, log=False, vmin=0, **opt)
            os.chdir(cwd)
            print('Finnished Simulation {0:}'.format(sim.name))
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
