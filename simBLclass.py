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

from athena_read import athdf as athdf
import helpers


#mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb,amsmath}"]

_dirs = ['', '~/', '~/Dropbox/dev/pyBL', '/scratch/gpfs/sashaph/BLayer', '/perseus/scratch/gpfs/sashaph/BLayer', ]
_dirs = map(os.path.expanduser, _dirs)
_dirs += [os.path.join(d, 'Mach8stampede') for d in _dirs]
_data_base = '/scratch/gpfs/sashaph/BLayer'
_file_fmts = ['BL.out2.%5.5d.athdf', 'disk.out1.%5.5d.athdf']

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

class BLfile(object):
    def __init__(self, fn, sim_path=None):
        self.t = None
        try:
            if fn == int(fn):
                if not sim_path is None:
                    fmts = [os.path.join(sim_path, i) for i in _file_fmts]
                else:
                    fmts = _file_fmts
                self.t = fn
                try:
                    fn = [i % fn for i in fmts if os.path.isfile(i % fn)][0]
                except IndexError:
                    tmp = ' or '.join([i % fn for i in fmts])
                    raise IOError('Cannot find file(s) ' + tmp)
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
        self.rc = .5 * (self.r[:-1] + self.r[1:])
        self.phic = .5 * (self.phi[:-1] + self.phi[1:])

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

    def _parse_data(self, data):
        try:
            data.shape
        except AttributeError:
            data = self.get2d(data)
        return data

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
            data = self.get2d(data)
        if not smooth is None:
            data = self.smooth(data, smooth)
        modes = self.fft(data)[1].argmax(axis=0)
        sqr = data**2
        loc = sqr.argmax(axis=0)
        phase = self.phic[loc]
        phase -= .5 * (1 - np.sign(data[loc,np.arange(data.shape[1])])) * np.pi
        phase %= 2 * np.pi
        if mod:
            phase %= 2. * np.pi / modes
        return phase

    def smooth(self, data, width=64):
        return smooth(self._parse_data(data), width=width)

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


    def plot2d(self, data, fn=None, save=False, subsample=False, title=None,
               name=None, ext='pdf', popt={}, cb=True, cbl=None, zerocent=None,
               vmin=None, vmax=None, cmap=None, cbopt={}, fig=None, fopt={},
               ax=None, log=False, aspect=1, sdir=None, smooth=None):
        '''Plot 2D sim data'''
        r = self.r[np.newaxis, :]
        phi = self.phi[:,np.newaxis]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        _popt = {}
        try:
            data.shape
        except AttributeError:
            if name is None:
                name = data
            data = self.get2d(data)
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

class BLsim(object):
    def __init__(self, path, fmts=None, mmax=100, mode_data=None):
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
        tmp = [fmt.split('%')[0] + '*.' + fmt.split('d.')[-1] for fmt in fmts]
        self.filenames = []
        for search in tmp:
            self.filenames += [os.path.split(i)[-1] for i in glob(os.path.join(path, search))]
        self.times = np.array([int(i.split('.')[2]) for i in self.filenames])
        if not (self.times == np.arange(self.times.size, dtype=int)).all():
            Warning('Incomplete dataset.')
        tmp = self.loadfile(self.filenames[0])
        for attr in ['r', 'phi', 'rc', 'phic']:
            setattr(self, attr, getattr(tmp, attr))
        self._mode_data = mode_data
        self._mmax = mmax
        self._mode_fn = os.path.join(path, 'mode_data.npy')
        if self._mode_data is None:
            if os.path.isfile(self._mode_fn):
                print('Loading file "{0:}"'.format(self._mode_fn))
                self._mode_data = np.load(self._mode_fn)
                self._mmax = self._mode_data.shape[1] - 1
        else:
            self._mmax = mode_data.shape[1] - 1
        if self._mmax is None:
            self._mloc = (None, None)
        else:
            self._mloc = (slice(None, self._mmax + 1))

    def __repr__(self):
        return '<BLsim "{0:}">'.format(self.name)

    def loadfile(self, fn):
        if fn in self.filenames or fn in self.times:
            return BLfile(fn, sim_path=os.path.abspath(self.path))

    def _store_mode(self, data, save=True):
        if save is None:
            tmp = os.path.abspath(self.path).lower()
            if 'matt' in tmp or 'colema' in tmp:
                save = True
        if save:
            np.save(self._mode_fn, data, allow_pickle=True)
        self._mode_data = data
        return None

    def rloc(self, r):
        return np.abs(r - self.rc).argmin()

    def philoc(self, phi):
        return np.abs(phi - self.phic).argmin()

    def all_mode_data(self):
        if not self._mode_data is None:
            return self._mode_data
        self._store_mode(np.array([self.loadfile(i).fft('pseudo')[-1][self._mloc] for i in self.times]))
        return self._mode_data

    def mode_data(self, plot=False, popt={}, main_plots=False, mpopt={}):
        out = np.zeros((self.times.size, self.r.size - 1))
        full = []
        for i, fn in enumerate(self.filenames):
            helpers.update_progress(float(i) / len(self.filenames))
            bf = self.loadfile(fn)
            if main_plots:
                _opt = {}#'sdir': os.path.join(self.path, 'channel_maps')}
                _opt.update(mpopt)
                bf.main_plots(**_opt)
            tmp = bf.fft('pseudo')[-1]
            full.append(tmp[self._mloc])
            out[bf.t] = tmp.argmax(axis=0)
        helpers.update_progress(1)
        self._store_mode(np.array(full))
        if plot:
            mode_plot(out, **popt)
        return out

    def mode_plot(self, data=None, cb=True, title=None, cbl=None, vmin=0,
                  vmax=20, main_plots=False, mpopt={}, save=False, fn=None,
                  ext='pdf', sdir=None, fig=None, fopt={}, ax=None):
        _mpopt = {'ext':ext, 'sdir':sdir}
        _mpopt.update(mpopt)
        if data is None:
            if self._mode_data is None or main_plots:
                data = self.mode_data(main_plots=main_plots, mpopt=_mpopt)
            else:
                data = self._mode_data.argmax(axis=1)
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
            title = self.name
        if title:
            plt.title(helpers.sanitize_lbl(title))
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
                fig=None, ax=None, fopt={}, vmin=1e-5, vmax=10, cb=True,
                cbl=None, popt={}, log=True):
        ir = np.abs(self.rc - r).argmin()
        r = self.rc[ir]
        data = self.all_mode_data()[:,:,ir].T

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

    def cross_corr(self, t1, t2, var='pseudo', dt=2 * np.pi, plot=False, norm=True):
        t1 = self.loadfile(t1)
        t2 = self.loadfile(t2)
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
            t1.plot2d(out, title='{0:d}, {1:d}'.format(t1.t, t2.t), vmax=1.1)
        return out

    def rect_cc_plot(self, t1, t2, var='pseudo', fn=None, save=False, ext='pdf',
                     sdir=None, fig=None, ax=None, fopt={}, vmin=None,
                     vmax=1.1, cb=True, cbl=None, popt={}, log=True, cmap=None):
        data = self.cross_corr(t1, t2, var=var).T

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
        _popt = dict(cmap=cmap, vmin=vmin, vmax=vmax)
        if log:
            _popt['norm'] = mpl.colors.LogNorm()
        _popt.update(popt)

        one = np.ones(data.shape)
        r = self.r[:, np.newaxis] * one
        phi = self.phi[np.newaxis, :] * one
        pcm = plt.pcolormesh(r, phi, data, **_popt)
        plt.colorbar()

    def prop_speed(self, t1, t2=None, dt=2 * np.pi, plot=False, fig=True, smooth=False, f=.9):
        if t2 is None:
            t2 = t1 + 1
        cc = self.cross_corr(t1, t2)
        if not smooth is False:
            if smooth is True or smooth is None:
                smooth = 64
            cc = _smooth(cc, smooth)
        out = np.empty(cc.shape[-1])
        for ir in xrange(out.size):
            tmp = cc[:,ir]
            loc = np.where(tmp < f * tmp.max())
            tmp[loc] = 0
            out[ir] = self.phic[argrelextrema(tmp, np.greater, mode='wrap')[0].min()]
        out /= (t2 - t1) * dt

        if plot:
            if fig is True:
                fig = plt.figure()
            plt.plot(self.rc, out, lw=1)
            plt.xlabel('Radius')
            plt.ylabel(r'$\Omega_p$')

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
