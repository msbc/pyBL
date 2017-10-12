#! /usr/bin/env python

#import h5py
#from mayavi import mlab
import numpy as np
#import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import math
import gc
#import psutil
from athena_read import athdf as athdf
import os
from glob import glob

import helpers


#mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb,amsmath}"]

_dirs = ['', '~/', '~/Dropbox/dev/pyBL', '/scratch/gpfs/sashaph/BLayer', '/perseus/scratch/gpfs/sashaph/BLayer', ]
_dirs = map(os.path.expanduser, _dirs)
_dirs += [os.path.join(d, 'Mach8stampede') for d in _dirs]
_data_base = '/scratch/gpfs/sashaph/BLayer'
_file_fmt = 'BL.out2.%5.5d.athdf'

class BLfile(object):
    def __init__(self, fn, sim_path=None):
        self.t = None
        try:
            if fn == int(fn):
                self.t = fn
                fn = _file_fmt % fn
        except ValueError:
            pass
        if (sim_path is None) and (not os.path.isfile(fn)):
            for d in _dirs:
                tmp = os.path.join(d, fn)
                if os.path.isfile(tmp):
                    fn = tmp
                    break
        elif sim_path:
            fn = os.path.join(sim_path, fn)
        self.fn = fn
        if self.t is None:
            self.t = int(os.path.split(fn)[1].split('.')[2])
        self._prefix = '.'.join(os.path.split(fn)[-1].split('.')[:-1])
        if sim_path and not self.t is None:
            self.name = os.path.split(sim_path)[-1] + ' {0:05d}'.format(self.t)
            self._prefix = os.path.split(sim_path)[-1] + '_{0:05d}'.format(self.t)
        else:
            self.name = os.path.split(fn)[-1]
        self.athdf = athdf(fn)
        self.r = self.athdf['x1f']
        self.phi = self.athdf['x2f']

    def __getitem__(self, key):
        try:
            return self.athdf[key]
        except KeyError:
            if key[:3] == 'vel' and len(key) == 4:
                return self.vel(key[3])
            if hasattr(self, key):
                try:
                    out = getattr(self, key)()
                    if out.shape == self['dens'].shape:
                        return out
                except AttributeError, TypeError:
                    pass
        raise KeyError('Unable to parse {0:}'.format(key))

    def __repr__(self):
        path, fn = os.path.split(self.fn)
        meh, head = os.path.split(path)
        return '<BLfile {0:}>'.format(os.path.join(head, fn))

    def get2d(self, var):
        return self[var][0,:,:]

    def vel(self, i):
        return self['mom{0:}'.format(i)] / self['dens']

    def pseudo(self):
        return self['mom1'] / np.sqrt(self['dens'])

    def fft(self, data, axis=-2, mag=True):
        try:
            data.shape
        except AttributeError:
            data = self.get2d(data)
        sp = np.fft.rfft(data, axis=axis)
        nu = np.fft.rfftfreq(self.phi.size)
        if mag:
            sp = np.absolute(sp)
        return nu, sp

    def channel_map(self, var='pseudo', save=False, fn=None, mmax=30, log=True,
                    fig=None, ax=None, aspect=None, fopt={}, popt={}, cbl=None,
                    title=None, vmin=1e-1, vmax=None, cb=True, cbopt={},
                    sdir=None, ext='pdf'):
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
            title = self.name
        _popt.update(popt)

        loc = (slice(None,mmax+1), slice(None))
        ft = (self.fft(self.get2d(var))[-1]**2)[loc]
        one = np.ones((ft.shape[0] + 1, self.r.size))
        mode = (np.arange(ft.shape[0] + 1)[:, np.newaxis] - .5) * one
        r = self.r[np.newaxis, :] * one

        pcm = plt.pcolormesh(mode, r, ft, **_popt)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        if title:
            plt.title(title)
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

    def plot2d(self, data, fn=None, save=False, subsample=False, title=None,
               name=None, ext='pdf', popt={}, cb=True, cbl=None, zerocent=None,
               vmin=None, vmax=None, cmap=None, cbopt={}, fig=None, fopt={},
               ax=None, log=False, aspect=1):
        '''Plot 2D sim data'''
        r = self.r[np.newaxis, :]
        phi = self.phi[:,np.newaxis]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        try:
            data.shape
        except AttributeError:
            if name is None:
                name = data
            data = self.get2d(data)

        #parse/set options and defaults
        if subsample:
            loc = (slice(None,None,subsample),) * len(x.shape)
            x = x[loc]
            y = y[loc]
            data = data[loc]
        if log:
            _popt['norm'] = mpl.colors.LogNorm()
        if zerocent is None and not log:
            zerocent = helpers.isZeroCent(data)
        if zerocent:
            if cmap is None:
                cmap = helpers.NCcmap
        if name:
            if title is None:
                title = name
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
            plt.title(title)
        if cb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(pcm, cax=cax, **cbopt)
            cb.ax.yaxis.set_offset_position('left')
            if cbl:
                cb.set_label(cbl)

        #save fig
        if save or fn:
            if fn is None:
                fn = self._prefix + '_plot.' + ext
            plt.savefig(fn)
            plt.close()

        return pcm

class BLsim(object):
    def __init__(self, path, fmt=None):
        if fmt is None:
            fmt = _file_fmt
        self._fmt = fmt
        self.name = os.path.split(os.path.abspath(path))[-1]
        if path == self.name and not os.path.isdir(path):
            for d in _dirs:
                tmp = os.path.join(d, path)
                if os.path.isdir(tmp):
                    path = tmp
                    break
        if not os.path.isdir(path):
            raise IOError('Simulation directory "{0:}" not found.'.format(path))
        self.path = path
        tmp = fmt.split('%')[0] + '*.' + fmt.split('d.')[-1]
        self.filenames = [os.path.split(i)[-1] for i in glob(os.path.join(path, tmp))]
        self.times = np.array([int(i.split('.')[2]) for i in self.filenames])
        if not (self.times == np.arange(self.times.size, dtype=int)).all():
            Warning('Incomplete dataset.')
        tmp = self.loadfile(self.filenames[0])
        for attr in ['r', 'phi']:
            setattr(self, attr, getattr(tmp, attr))

    def __repr__(self):
        return '<BLsim "{0:}">'.format(self.name)

    def loadfile(self, fn):
        if fn in self.filenames or fn in self.times:
            return BLfile(fn, sim_path=os.path.abspath(self.path))

    def mode_data(self, plot=False, popt={}, channel_maps=False, cmopt={}):
        out = np.zeros((self.times.size, self.r.size - 1))
        for i, fn in enumerate(self.filenames):
            helpers.update_progress(float(i) / len(self.filenames))
            bf = self.loadfile(fn)
            if channel_maps:
                _opt = {'sdir': os.path.join(self.path, 'channel_maps')}
                _opt.update(cmopt)
                bf.channel_map(save=True, **_opt)
            out[bf.t] = bf.fft('pseudo')[-1].argmax(axis=0)
        helpers.update_progress(1)
        if plot:
            mode_plot(out, **popt)
        return out

    def mode_plot(self, data=None, cb=True, title=None, cbl=None, vmin=0,
                  vmax=20, channel_maps=False, cmopt={}, save=False, fn=None):
        if data is None:
            data = self.mode_data(channel_maps=channel_maps, cmopt=cmopt)
        one = np.ones((self.times.size, self.r.size))
        r = self.r[np.newaxis, :] * one
        t = np.concatenate(self.times, [self.times[-1]+1])[:,np.newaxis] * one
        t -= .5
        cmap = plt.get_cmap(lut = vmax - vmin + 1)
        pcm = plt.pcolormesh(t, r, data, vmin=vmin, vmax=vmax, cmap=cmap)
        ax = plt.gca()
        if title is None:
            title = self.name
        if title:
            plt.title(title)
        if cb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(pcm, cax=cax)
            cb.ax.yaxis.set_offset_position('left')
            if cbl:
                cb.set_label(cbl)
        if save or fn:
            if fn is None:
                fn = self.name + '_ST_mode.' + ext
            plt.savefig(fn)
            plt.close()

def mkplots(sims=None, path=''):
    if sims is None:
        sims = []
        tmp = [i for i in glob(os.path.join(path, '*')) if os.path.isdir(i)]
        for i in tmp:
            try:
                sims.append(BLsim(i))
            except:
                print '"{0:}" is not a simulation'.format(i)
    cwd = os.getcwd()
    for sim in sims:
        if not hasattr(sim, 'path'):
            sim = BLsim(os.path.join(path, sim))
        if path:
            tmp = os.path.split(sim.path)[-1]
            if not os.path.isdir(tmp):
                os.mkdir(tmp)
            os.chdir(tmp)
        else:
            os.chdir(sim.path)
        sdir = os.path.join(os.getcwd(), 'channel_maps')
        sim.mode_plot(save=True, channel_maps=True, cmopt={'sdir':sdir})
        os.chdir(cwd)

if __name__ == '__main__':
    path = '~/tigress/BLayer'
    mkplots(glob(os.path.join(path, 'Mach*stampede')), path=path)
