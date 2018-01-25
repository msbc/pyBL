#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
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
        return ar.athdf(fn)
    if ext == 'npy':
        return np.load(fn)[()]
    raise IOError('Cannot identify file type of "{0:}"'.format(fn))

class BLfile(dict):
    def __init__(self, fn, sim_path=None, t=None, trim=True, data=None,
                 defvar=None, ai_data={}, sim=None):
        self.t = t
        self.sim = sim
        self._ai_data = ai_data
        self.fn = _findAbsPath(fn, sim_path)
        self.path = os.path.split(self.fn)[0]
        if data is None:
            self.data = self._parse_file()
        else:
            self.data = data
        if self.t is None:
            for i in ['Time', 'time', 'T', 't']:
                if i in self.data:
                    self.t = self.data[i]
                    break
        if self.t is None:
            self.t = int(os.path.split(fn)[1].split('.')[2])
        if not 'Time' in self:
            self['Time'] = self.t
        self._prefix = '.'.join(os.path.split(fn)[-1].split('.')[:-1])
        #if sim_path and not self.t is None:
        #    self.name = os.path.split(sim_path)[-1] + ' {0:05d}'.format(self.t)
        #    self._prefix = os.path.split(sim_path)[-1] + '_{0:05d}'.format(self.t)
        #else:
        self.name = os.path.split(fn)[-1]
        self.t_str = '%.04g' % self.t
        self._Qtrim = trim
        if trim:
            self.update({i: self._trim(self.data[i]) for i in self.data})
        else:
            self.update(self.data)
        self.r = self.data['x1f']
        self.phi = self.data['x2f']
        self.rc = .5 * (self.r[:-1] + self.r[1:])
        self.phic = .5 * (self.phi[:-1] + self.phi[1:])
        self._shape = self.phic.size, self.rc.size
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
                if self[i].shape == self._shape:
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

    def _parse_file(self, fn=None):
        if fn is None:
            fn = self.fn
        return _parse_file(fn)

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
                if out.shape == self._shape:
                    return out
            except (AttributeError, TypeError):
                pass
        raise KeyError('Unable to parse {0:}'.format(key))

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self._parse_self(key)

    def rloc(self, r):
        return np.abs(r - self.rc).argmin()

    def plot2d(self, data=None, fn=None, save=False, subsample=False, title=None,
               name=None, ext='pdf', popt={}, cb=True, cbl=None, zerocent=None,
               vmin=None, vmax=None, cmap=None, cbopt={}, fig=None, fopt={},
               ax=None, log=False, aspect=1, sdir=None, smooth=None, phi_shift=0, r_cut=None):
        '''Plot 2D sim data'''
        r = self.r[np.newaxis, :]
        phi = self.phi[:,np.newaxis] + phi_shift
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        _popt = {}
        if data is None:
            data = self._defvar
        if data == 'pseudo' and r_cut is None:
            r_cut = .85
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

        if name:
            if title is None:
                #title = self.name + ' ' + self.t_str + ' ' + name
                title = self.name + ' ' + name
            if save and fn is None:
                fn = self._prefix + '_' + name + '_plot.' + ext

        _popt = dict(cmap=cmap, vmin=vmin, vmax=vmax)
        _popt.update(popt)

        if fig is None and ax is None:
            fig = plt.figure(**fopt)
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        if aspect:
            ax.set_aspect(aspect)

        #start plotting
        pcm = plt.pcolormesh(x, y, data, **_popt)
        if title:
            plt.title(helpers.sanitize_lbl(title))
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

        return pcm

    def smooth(self, data, width=64):
        return smooth(self._parse_data(data), width=width)

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

    def FFT_errors(self, vmin='smart', vmax='max', mmax=None):
        pfft = self.fft('pseudo')
        afft = (self['FT-Re'] + 1j * self['FT-Im'])[:225]
        loc = slice(None)
        if not mmax is None:
            loc = slice(None, mmax + 1)
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
        dth = np.angle(pfft) - np.angle(afft)
        _opt = dict(interpolation='nearest', norm=mpl.colors.LogNorm())
        fig = plt.figure(figsize=(8,8))
        #fig, axs = plt.subplots(1, 3, figsize=(12,4))

        #plt.figure()
        #plt.sca(axs[0])
        plt.subplot(221)
        data = abs(dmag)
        opt = {}
        opt.update(_opt)
        if 'smart' in [vmin, vmax]:
            tmp = helpers.smartlim(data[loc])
        if vmin == 'smart':
            opt['vmin'] = tmp[0]
        else:
            opt['vmin'] = vmin
        if vmax == 'smart':
            opt['vmax'] = tmp[1]
        elif vmax == 'max':
            opt['vmax'] = data[loc].max()
        else:
            opt['vmax'] = vmax
        im = plt.imshow(data, **opt)
        cb = plt.colorbar(im)
        plt.xlabel('r index')
        plt.ylabel('mode')
        cb.set_label(r'$\left|\Delta|{\rm FFT}|/|{\rm FFT})|\right|$')
        ylim = plt.ylim(None, mmax)

        #plt.figure()
        #plt.sca(axs[1])
        plt.subplot(222)
        data = abs(dth)
        del(opt)
        opt1 = {}
        opt1.update(_opt)
        if 'smart' in [vmin, vmax]:
            tmp = helpers.smartlim(data[loc])
        if vmin == 'smart':
            opt1['vmin'] = tmp[0]
        else:
            opt1['vmin'] = vmin
        if vmax == 'smart':
            opt1['vmax'] = tmp[1]
        elif vmax == 'max':
            opt1['vmax'] = data[loc].max()
        else:
            opt1['vmax'] = vmax
        im1 = plt.imshow(data, **opt1)
        cb1 = plt.colorbar(im1)
        plt.xlabel('r index')
        plt.ylabel('mode')
        cb1.set_label(r'$\left|\Delta\theta\right|$')
        plt.ylim(*ylim)
        #return None

        #plt.figure()
        #plt.sca(axs[2])
        plt.subplot(223)
        data = amp
        del(cb,cb1,opt1,vmin,vmax)
        opt2 = {}
        opt2.update(_opt)
        if 0:
            if 'smart' in [vmin, vmax]:
                tmp = helpers.smartlim(data[loc])
            if vmin == 'smart':
                opt2['vmin'] = tmp[0]
            else:
                opt2['vmin'] = vmin
            if vmax == 'smart':
                opt2['vmax'] = tmp[1]
            else:
                opt2['vmax'] = vmax
        #opt2['vmin'] = helpers.smartlim(data[loc])[0]
        #opt2['vmax'] = data[loc].max()
        print(opt2)
        im2 = plt.imshow(data,interpolation='nearest', norm=mpl.colors.LogNorm(),
                         vmin=helpers.smartlim(data[loc])[0], vmax=data[loc].max())#, vmin=helpers.smartlim(data[loc])[0], **opt2)
        cb2 = plt.colorbar(im2)
        plt.xlabel('r index')
        plt.ylabel('mode')
        cb2.set_label(r'$\left|{\rm FFT}\right|$')
        plt.ylim(*ylim)

        ax = plt.subplot(224)
        self.plot2d('pseudo', ax=ax)

class BLConsPrim(BLfile):

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
            M = self.M
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

    def CLplot(self, Op, M=None):
        cl = self.CL(), self.CL20(), self.CL24(), self.CL25(Op, M), self.CS(), self.CA()
        lbls = ['BRS12 Eqn %d' % i for i in [19,20,24,25]] + ['BRS13 $C_S$','BRS13 $C_A$']
        for i in cl:
            plt.plot(self.rc, i)
        plt.axhline(0, ls=':', lw=1, c='k')
        plt.legend(lbls)
        plt.xlabel('R')
        plt.ylabel('$C_L$')

    def Ok(self):
        return self.rc**-1.5

    def Oloc(self):
        return self.rhoWeight('vel2') / self.rc

class BLcons(BLConsPrim):
    def _special_keys(self, key):
        if key[:3] == 'vel' and len(key) == 4:
            return self.vel(key[3])
        return None

    def vel(self, i):
        return self['mom{0:}'.format(i)] / self['dens']

    def pseudo(self):
        return self['mom1'] / np.sqrt(self['dens'])

class BLprim(BLConsPrim):
    def _special_keys(self, key):
        if key[:3] == 'mom' and len(key) == 4:
            return self.mom(key[3])
        return None

    def mom(self, i):
        return self['vel{0:}'.format(i)] * self['dens']

    def pseudo(self):
        return self['vel1'] * np.sqrt(self['dens'])

class BLFT(BLfile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {}
        self.data.update(self)

    def _trim(self, data):
        data = super()._trim(data)
        if len(data.shape) == 2:
            a = data.min(axis=1)
            b = data.max(axis=1)
            i = a.size - 1
            while a[i] == 0 and b[i] == 0 and i > 0:
                i -= 1
            try:
                i = max(i, self._ai_data['meshblock']['nx2'] - 1)
            except KeyError:
                pass
            data = data[slice(0, i + 1)].copy()
        return data

    def _special_keys(self, key):
        if key == 'FT':
            return self['FT-Re'] + 1j * self['FT-Im']
        if key in ['mag', 'amp']:
            return np.abs(self['FT'])
        if key == 'angle':
            return np.angle(self['FT'])
        return None

############################
# End of BLfile subclasses #
############################

def loadBLfile(fn, **kwargs):
    fn = _findAbsPath(fn, kwargs.get('sim_path', None))
    data = _parse_file(fn)
    ai_fn = kwargs.pop('athinput_fn', None)
    ai_data = kwargs.pop('athinput_data', None)
    if ai_fn is None:
        tmp = glob('athinput.*')
        if len(tmp) == 1:
            ai_fn = tmp[0]
    if ai_fn and ai_data is None:
        ai_data = ar.athinput(ai_fn)
    kind = None
    if not ai_data is None:
        kwargs['ai_data'] = ai_data
        tmp = os.path.split(fn)[-1].split('.')
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
    return BLfile(fn, data=data, **kwargs)

class _old_BLsliceFile(object):
    def __init__(self, fn):
        self.prefix, self.block, self.var, self.index, self.ext = fn.split('.')
        if self.ext == 'vtk':
            x1, x2, x3, data = ar.vtk(fn)
        elif self.ext == 'tab':
            x1, x2, x3, data = ar.tab(fn)
        else:
            raise ValueError('Cannot parse filetype ' + ext)
        self.r = x1
        self.phi = x2
        if x1.size > 1:
            self.rc = .5 * (self.r[:-1] + self.r[1:])
        else:
            self.rc = self.r
        if x2.size > 1:
            self.phic = .5 * (self.phi[:-1] + self.phi[1:])
        else:
            self.phic = self.phi
        self._x3 = x3
        self.data = data
        self.axes = [x3, x2, x1]

class _old_BLslice(object):
    def __init__(self, name, index, block_order=None, ifmt='%05d', prefix=None, ext='vtk', direction=None):
        ext = ext.lstrip('.')
        if type(index) == int:
            index = ifmt % index
        if prefix:
            prefix = np.atleast_1d(prefix)
        else:
            prefix = _pre[:]
        for p in prefix:
            qry = '.'.join([p, 'block[0-9]*', name, index, ext])
            #print qry
            files = glob(qry)
            if files:
                break
        if not files:
            raise IOError('Cannot find slice files')
        files = map(BLsliceFile, files)
        f0 = files[0]
        if direction is None:
            direction = [i.size > 1 for i in f0.axes].index(True)
        else:
            direction = 3 - direction
        files = {i.block: i for i in files}
        if block_order is None:
            block_order = sorted(files.keys(), key=lambda x:files[x].axes[direction][0])
        self.block_order = block_order
        data = {}
        self.axes = None
        for i, b in enumerate(block_order):
            f = files[b]
            if self.axes is None:
                self.axes = [i.copy() for i in f.axes]
            else:
                tmp = [self.axes[direction], f.axes[direction]]
                if tmp[0][-1] != tmp[1][0]:
                    raise ValueError('Missing gaps in block reconstruction.')
                tmp[1] = tmp[1][1:]
                self.axes[direction] = np.concatenate(tmp)
            for var in f.data:
                if var in data:
                    data[var] = np.concatenate((data[var], f.data[var]), axis=direction)
                else:
                    data[var] = f.data[var]
        self.data = data
        self.r = self.axes[2]
        self.phi = self.axes[1]
        self.direction = 3 - direction
        self._dir = direction
        self.files = files
        if direction == 1:
            self.rc = .5 * (self.r[:-1] + self.r[1:])
            self.phic = self.phi
        else:
            self.phic = .5 * (self.phi[:-1] + self.phi[1:])
            self.rc = self.r

class _old_BLmodes(object):
    def __init__(self, index, mrng=None, mfmt='%02d'):
        Re = 'FT-Re'
        Im = 'FT-Im'
        if mrng is None:
            qry = '*.block[0-9]*.mode[0-9]*.[0-9]*.*'
            mrng = np.array([int(i.split('.')[2][4:]) for i in glob(qry)])
            mrng = mrng.min(), mrng.max()
        mrng = np.atleast_1d(mrng).astype(int)
        if len(mrng) == 1:
            mrng = [0, mring[0]]
        modes = range(mrng[0], mrng[1] + 1)
        self.modes = modes
        data = None
        for m in modes:
            s = BLslice('mode' + mfmt % m, index)
            if data is None:
                tmp = list(s.data[Re].shape)
                tmp[1] = len(modes)
                self.axes = s.axes
                self.axes[1] = modes
                self.rc = s.rc
                data = np.empty(tmp, dtype=np.complex)
            data[0, m, :] = s.data[Re][0, 0, :]
            data[0, m, :].imag = s.data[Im][0, 0, :]
        self.data = data



class BLsim(object):
    def __init__(self, path, fmts=None, fft_data=None, fft_time=None, athinput=None, mode_mask=None):
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
        if os.path.isfile(athinput):
            self.inputs = ar.athinput(athinput)
        else:
            self.inputs = {}
        self.fileDict = {}
        self.varDict = {}
        axes = []
        if not self.inputs:
            searches = [fmt.split('%')[0] + '*.' + fmt.split('d.')[-1] for fmt in fmts]
            raise NotImplementedError('Currently needs athinput.')
        else:
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

        #for attr in ['r', 'phi', 'rc', 'phic']:
        #    setattr(self, attr, getattr(tmp, attr))
        self._fft_data = fft_data
        self._fft_time = fft_time
        self._mode_mask = mode_mask

    def _load_fft_data(self):
        try:
            data = []
            t = []
            for fn in self.files('FT-Range'):
                f = self.loadfile(fn)
                data.append(f['FT'])
                t.append(f.t)
            data = np.array(data)
            t = np.array(t)
        except ValueError:
            raise NotImplementedError
            data = np.array([self.loadfile(f)['FT'] for f in self.files('FT')])
        if self._fft_data is None:
            self._fft_data = data
        if self._fft_time is None:
            self._fft_time = t
        return data

    @property
    def fft(self):
        if not self._fft_data is None:
            return self._fft_data
        return self._load_fft_data()

    @property
    def filenames(self):
        return sum(self.fileDict.values(), [])

    def files(self, key=None):
        if key is None:
            return self.filenames
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
            self._load_fft_data()
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

    def rloc(self, r):
        return np.abs(r - self.rc).argmin()

    def loadfile(self, fn):
        if fn in self.filenames:
            return loadBLfile(fn, sim_path=os.path.abspath(self.path), sim=self, ai_data=self.inputs)

    def mode_mask(self, data=None, info=False, amin=None):
        if amin is None:
            amin = 1e-3 / self.rc.size / self.phic.size
        amp = np.abs(self.fft)
        if self._mode_mask is None:
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
                tmp = np.logical_and(amp > amin, m + 3. * s - amp < 0) # signal
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
        DeprecationWarning('This function is deprecated.')
        out = np.array([np.abs(self.fft), np.angle(self.fft)])
        return out

    def prop_speed(self, dt=None):
        if dt is None:
            for var in ['FT-Range', 'FT']:
                if var in self.varDict:
                    dt = self.inputs[self.varDict[var]]['dt']
                    break
        data = np.angle(self.fft)
        #dphi = np.gradient(data[1] / dt, axis=0)
        dphi = mod_grad(data, axis=0, mod=tau)
        m = np.arange(dphi.shape[1])
        return dphi / dt / (m[np.newaxis,:,np.newaxis])

    def _r_phase_plotter(self, r, data, ret_m=False, tloc=None):
        ir = self.rloc(r)
        #data = self.mode_phase()
        modes = []
        for i in xrange(data.shape[1]):
            tmp = data[:,i,ir].copy()
            tmp = tmp[tloc]
            j = 0
            # make sure data covers 5 adjacent times
            while np.isfinite(tmp).any() and j < 5:
                tmp = np.gradient(tmp)
                j += 1
            if np.isfinite(tmp).any():
                plt.plot(self.fft_time(), data[:,i,ir], lw=1)
                modes.append(i)
        plt.legend(['$m=%d$' % m for m in modes])
        plt.xlabel('Time step')
        #plt.ylabel('Phase')
        plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:.2f}$'.format(r))
        if ret_m:
            return modes

    def r_phase(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        m = self._r_phase_plotter(r, self.mode_mask(np.angle(self.fft)), ret_m=ret_m)
        plt.ylabel('Phase')
        return m

    def r_speed(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        m = self._r_phase_plotter(r, self.mode_mask(self.prop_speed()), ret_m=ret_m)
        plt.ylabel('Speed')
        return m

    def r_amp(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        info = self.mode_mask(self.mode_phase()[0], info=True)
        m = self._r_phase_plotter(r, info['data'], ret_m=ret_m)
        ir = self.rloc(r)
        ls = '-'
        for i in [0,1,3]:
            y = info['mean'][:,ir] + i * info['std'][:,ir]
            plt.plot(y, c='k', ls=ls, lw=1)
            ls = ':'
        plt.ylabel('Amplitude')
        return m

    def mt_plot(self, r, fn=None, save=False, ext='pdf', sdir=None,
                fig=None, ax=None, fopt={}, vmin='smart', vmax='max', cb=True,
                cbl=None, popt={}, log=True):
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

    def _r_phase_plotter(self, r, data, ret_m=False, tlim=None):
        ir = self.rloc(r)
        #data = self.mode_phase()
        modes = []
        for i in xrange(data.shape[1]):
            tmp = data[:,i,ir].copy()
            if tlim:
                tmp = tmp[tlim:]
            j = 0
            while np.isfinite(tmp).any() and j < 5:
                tmp = np.gradient(tmp)
                j += 1
            if np.isfinite(tmp).any():
                plt.plot(self.fft_time / tau, data[:,i,ir], lw=1)
                modes.append(i)
        opt = {'loc': 0, 'frameon': False, 'handlelength': .7, 'prop': {'size':8}, 'ncol': 3}
        plt.legend(['$%d$' % m for m in modes], **opt)
        plt.xlabel(r'Time/$2\pi$')
        #plt.ylabel('Phase')
        plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:.2f}$'.format(r))
        if ret_m:
            return modes

    def r_phase(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        m = self._r_phase_plotter(r, self.mode_mask(self.mode_phase()[1]), ret_m=ret_m)
        plt.ylabel('Phase')
        return m

    def r_speed(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        m = self._r_phase_plotter(r, self.mode_mask(self.prop_speed()), ret_m=ret_m)
        plt.ylabel('Speed')
        return m

    def r_amp(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        info = self.mode_mask(self.mode_phase()[0], info=True)
        m = self._r_phase_plotter(r, info['data'], ret_m=ret_m)
        ir = self.rloc(r)
        ls = '-'
        for i in [0,1,3]:
            y = info['mean'][:,ir] + i * info['std'][:,ir]
            plt.plot(self.fft_time / tau, y, c='k', ls=ls, lw=1)
            ls = ':'
        plt.ylabel('Amplitude')
        return m

    def diagnostic(self, r=1.3, save=False, fn=None, ext='pdf', figsize=(8,8),
                   sdir=None, subsample=None):
        self.fft #make sure data is loaded
        self.mode_mask()
        fig = plt.figure(figsize=figsize)
        gs = mpl.gridspec.GridSpec(2, 2, top=.9, bottom=.05, hspace=.2)

        ax = plt.subplot(gs[0,0])
        self.r_amp(r, fig=False)

        ax = plt.subplot(gs[0,1])
        self.r_speed(r, fig=False)

        ax = plt.subplot(gs[1,0])
        f = self.loadfile(self.files('cons')[-1])
        f.plot2d('pseudo', ax=ax, vmin='smart', cbl=r'$v_r\sqrt{\rho}$', subsample=subsample)
        fig.suptitle('Diagnostic for ' + helpers.sanitize_lbl(self.name))

        ax = plt.subplot(gs[1,1])
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
        _add(r'$\mathcal{M}$', 1. / self.inputs['hydro']['iso_sound_speed'])
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

class auxBLsim(BLsim):
    def _store_mode(self, data, save=True):
        if save is None:
            tmp = os.path.abspath(self.path).lower()
            if 'matt' in tmp or 'colema' in tmp:
                save = True
        if save:
            np.save(self._mode_fn, data, allow_pickle=True)
        self._mode_phase = data
        return None

    def rloc(self, r):
        return np.abs(r - self.rc).argmin()

    def philoc(self, phi):
        return np.abs(phi - self.phic).argmin()

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

    def Omega_seq(self, t0, t, **kwargs):
        kwargs['plot'] = True
        for i in t:
            self.prop_speed(t0, t0 + i, **kwargs)
            kwargs['fig'] = False
        yl = plt.ylim(.5,None)
        if yl[1] > 1:
            plt.ylim(.5,1)
        plt.legend(t)

    def mode_phase(self, mmax=None, main_plots=False, mpopt={}):
        if mmax is None:
            mmax = self._mmax
        if not self._mode_phase is None and not main_plots:
            return self._mode_phase * self.kern
        if main_plots:
            out = []
            _opt = {}
            _opt.update(mpopt)
            for i, fn in enumerate(self.filenames):
                helpers.update_progress(float(i) / len(self.filenames))
                bf = self.loadfile(fn)
                bf.main_plots(**_opt)
                out.append(bf.mode_phase(mmax=mmax))
            if not self._mode_phase is None:
                return self._mode_phase
            out = np.array(out)
        else:
            out = np.array([self.loadfile(i).mode_phase(mmax=mmax) for i in self.times])
        out = np.array([out[:,0], out[:,1]])
        self._store_mode(out)
        return out * self.kern

    def prop_speed(self, dt=None):
        if dt is None:
            dt = self.dt
        data = self.mode_phase()
        #dphi = np.gradient(data[1] / dt, axis=0)
        dphi = mod_grad(data[1], axis=0, mod=tau)
        m = np.arange(dphi.shape[1]) + 1.
        return dphi / dt / (m[np.newaxis,:,np.newaxis] - 1)

    def _r_phase_plotter(self, r, data, ret_m=False, tlim=None):
        if tlim is None:
            tlim = self.times.size // 10
        ir = self.rloc(r)
        #data = self.mode_phase()
        modes = []
        for i in xrange(data.shape[1]):
            tmp = data[:,i,ir].copy()
            if tlim:
                tmp = tmp[tlim:]
            j = 0
            while np.isfinite(tmp).any() and j < 5:
                tmp = np.gradient(tmp)
                j += 1
            if np.isfinite(tmp).any():
                plt.plot(self.fft_time, data[:,i,ir], lw=1)
                modes.append(i)
        plt.legend(['$m=%d$' % m for m in modes])
        if self.dt:
            plt.xlabel('Time / $%.4f$' % self.dt)
        else:
            plt.xlabel('Time step')
        #plt.ylabel('Phase')
        plt.title(helpers.sanitize_lbl(self.name) + ' $r={0:.2f}$'.format(r))
        if ret_m:
            return modes

    def r_phase(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        m = self._r_phase_plotter(r, self.mode_mask(self.mode_phase()[1]), ret_m=ret_m)
        plt.ylabel('Phase')
        return m

    def r_speed(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        m = self._r_phase_plotter(r, self.mode_mask(self.prop_speed()), ret_m=ret_m)
        plt.ylabel('Speed')
        return m

    def r_amp(self, r, ret_m=False, fig=True):
        if fig is True:
            plt.figure()
        info = self.mode_mask(self.mode_phase()[0], info=True)
        m = self._r_phase_plotter(r, info['data'], ret_m=ret_m)
        ir = self.rloc(r)
        ls = '-'
        for i in [0,1,3]:
            y = info['mean'][:,ir] + i * info['std'][:,ir]
            plt.plot(self.fft_time, y, c='k', ls=ls, lw=1)
            ls = ':'
        plt.ylabel('Amplitude')
        return m

    def plot2d(self, t, *args, **kwargs):
        f = self.loadfile(t)
        return f.plot2d(*args, **kwargs)

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
