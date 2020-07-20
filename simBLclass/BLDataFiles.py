import h5py
# from mayavi import mlab
import argparse
import numpy as np
import sympy as sp
import matplotlib as mpl
if __name__ == "__main__":
    mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import scoreatpercentile as percentile
from scipy.stats import linregress
import scipy.signal
import gc
import os
from glob import glob
import sys
import traceback
from ._local_helpers import *

try:
    from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel, Box1DKernel
except ImportError:
    pass
from scipy.ndimage.filters import convolve1d
import time
import tarfile
import subprocess

from .. import athena_read as ar
from ..delayed_read import athdf
from .. import helpers
from ..helpers import rolling_weighted_triangle_conv as running_mean
from ..helpers import grad, mod_grad
from ..parmap import parmap
from .defaults import rc
from ._local_helpers import *

class BLfileBase(dict):
    def __init__(self, fn, sim_path=None, t=None, trim=True, data=None, num_ghost=0,
                 defvar=None, ai_data=None, sim=None, file_handle=None, x2_face=None):
        self.t = t
        self.sim = sim
        self._Qtrim = trim
        # if ai_data is None:
        #    ai_data = {}
        self._ai_data = ai_data
        if sim is not None:
            self.mach = sim.mach
        elif ai_data is not None:
            try:
                self.mach = 1. / self.inputs['hydro']['iso_sound_speed']
            except KeyError:
                self.mach = 1. / self.inputs['hydro']['invMach']
        self.fn = findAbsPath(fn, sim_path)
        if file_handle is None:
            file_handle = fn
        self._file_handle = file_handle
        if data is None:
            # self.data = ar.athdf(file_handle, face_func_2=x2_face, return_levels=True, num_ghost=num_ghost)
            self.data = athdf(file_handle, face_func_2=x2_face, return_levels=True)
        else:
            self.data = data
        self.x2_face = x2_face
        for i in self.data.keys():
            if i not in self:
                self[i] = None
            else:
                print('Warning: key {0:} already exists.'.format(i))
        # super(BLfile, self).__init__(self.fn)
        self.path = os.path.split(self.fn)[0]
        # print(data)
        # if data is not None:
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
        self.orbit = self.t / tau
        self._prefix = '.'.join(os.path.split(fn)[-1].split('.')[:-1])
        # if sim_path and not self.t is None:
        #    self.name = os.path.split(sim_path)[-1] + ' {0:05d}'.format(self.t)
        #    self._prefix = os.path.split(sim_path)[-1] + '_{0:05d}'.format(self.t)
        # else:
        self.name = os.path.split(fn)[-1]
        self.t_str = '%.04g' % self.t
        self.r = self['x1f']
        self.dr = self.r[1:] - self.r[:-1]
        self.phi = self['x2f']
        self.rc = .5 * (self.r[:-1] + self.r[1:])
        self.phic = .5 * (self.phi[:-1] + self.phi[1:])
        self._grid_shape = self.phic.size, self.rc.size
        self._default_var = defvar

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        del self.data
        return False

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
        for i in filter(None, [self._default_var, 'Rpseudo', 'pseudo', 'dens', 'FT-mag',
                               'FT-Re']):
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
        return (np.roll(data, -1, axis=axis) - np.roll(data, 1, axis=axis)) / (
                self.phic[2] - self.phic[0])

    def rloc(self, r, subsample=None):
        rc = self.rc[::subsample]
        return np.abs(r - rc).argmin()


class BLfile(BLfileBase):
    def __getitem__(self, item):
        assert(item != "FT-FT-test-Re-Re")
        try:
            return super(BLfile, self).__getitem__(item)
        except KeyError:
            return self.expr_eval(item)

    def expr_eval(self, expr):
        if expr in self:
            return self[expr]
        expr = sp.sympify(expr)
        sym = [i for i in expr.atoms() if isinstance(i, sp.symbol.Symbol)]
        data = [super(BLfile, self).__getitem__(str(i)) for i in sym]
        #subs = {i: self[str(i)] for i in expr.atoms() if type(i) == sp.symbol.Symbol}
        return sp.lambdify(sym, expr, "numpy")(*data)

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
               ax=None, log=False, aspect=1, sdir=None, smooth=None, cax=None,
               phi_shift=0, r_cut=None, phi_dot=0, ret_fn=False, rplot=1, lnorm=None,
               overwrite=True, display=False, minmax=True, txt_opt=None, printvmax=False,
               figsize=None, dpi=300):
        """Plot 2D sim data"""
        _fopt = dict(figsize=figsize, dpi=dpi)
        if fopt:
            _fopt.update(fopt)
        fopt = _fopt
        if popt is None:
            popt = {}
        if cbopt is None:
            cbopt = {}
        r = self.r[np.newaxis, :]
        phi = self.phi[:, np.newaxis] + phi_shift
        if self.t and phi_dot:
            phi -= phi_dot * self.t
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        _popt = {}
        if data is None:
            data = self._defvar
        if hasattr(data, 'lower'):
            if data in ['pseudo', 'Rpseudo'] and r_cut is None:
                r_cut = 1.03
                if vmax is None and vmin is None:
                    vmin = '99.5%'
            if data in ['vorticity', 'vortensity']:
                if r_cut is None:
                    r_cut = 1.03
                if zerocent is None:
                    zerocent = True
                if vmax is None and vmin is None:
                    vmax = 'smart'
            if data in ['vi', 've']:
                vmin = '99%'
            if cbl is None:
                cbl = helpers.labeler(data)
                name = data

        if name:
            if name in [True, 1]:
                name = ''
            if title is None:
                # title = self.name + ' ' + self.t_str + ' ' + name
                title = self.name + ' ' + name
                if self.sim:
                    title = self.sim.name + r' $t/2\pi = {0:g}$'.format(self.t / tau)
            if save and fn is None:
                fn = self._prefix + '_' + name + '_plot.' + ext

        if save or fn:
            save = True
            if fn is None:
                fn = self._prefix + '_plot.' + ext
            if not sdir is None:
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                fn = os.path.join(sdir, fn)
        if parse_not_overwrite(overwrite, fn):
            if ret_fn:
                return fn
            return None
        if lnorm:
            if lnorm is True:
                if name in ['Rpseudo', 'pseudo']:
                    lnorm = -3 #int(np.round(-2 * np.log10(self.sim.mach) - 3))
                else:
                    lnorm = 0
            if lnorm:
                lntxt = '/10^{' + str(lnorm) + '}'
        else:
            lnorm = 0
        if display:
            print('    Map of t/orb={:d}'.format(int(self.orbit + .5)))

        data = self._parse_data(data)
        if type(data) != np.ndarray:
            raise TypeError('Data has type "{:}", not ndarray.'.format(type(data)))
        data = data * 10**-lnorm
        if not smooth is None:
            data = self.smooth(data, smooth)

        # parse/set options and defaults
        if subsample:
            loc = (slice(None, None, subsample),) * len(x.shape)
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
        except (TypeError, IndexError):
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

        # start plotting
        if printvmax:
            if 'vmax' in _popt:
                print(_popt['vmax'])
        pcm = plt.pcolormesh(x, y, data, **_popt)
        if minmax:
            if r_cut is None:
                r_cut = 1.03
            dmin = data[:, self.rloc(r_cut):-5].min()
            dmax = data[:, self.rloc(r_cut):-5].max()
            txt = 'max: {0:.4g}\nmin: {1:.4g}'.format(dmax, dmin)
            x = .99 * self.r[-1]
            _txt_opt = dict(x=-x, y=x, ha='left', va='top')
            try:
                _txt_opt.update(txt_opt)
            except TypeError:
                pass
            plt.text(_txt_opt.pop('x'), _txt_opt.pop('y'), txt, **_txt_opt)
        if rplot:
            rplot = np.atleast_1d(rplot)
            for r in rplot:
                plt.plot(r * np.cos(self.phic), r * np.sin(self.phic), lw=1, c='1',
                         ls=':')
        if title:
            plt.title(helpers.sanitize_lbl(title.format(**self.__dict__)))
        if cb:
            divider = make_axes_locatable(ax)
            if cax is None:
                cax = divider.append_axes("right", size="5%", pad=0.05)
            pos = cbopt.pop('pos', 'left')
            cb = plt.colorbar(pcm, cax=cax, **cbopt)
            if pos in ['left', 'right']:
                cb.ax.yaxis.set_offset_position(pos)
            elif pos == 'top':
                cb.ax.xaxis.set_label_position('top')
                cb.ax.xaxis.set_ticks_position('top')
            if cbl:
                if lnorm:
                    cbl = cbl.rstrip('$') + lntxt
                    if '$' in cbl:
                        cbl = cbl + '$'
                cb.set_label(cbl)

        plt.sca(ax)
        if printvmax:
            print(plt.clim())
        # save fig
        if save:
            plt.savefig(fn)
            plt.close()
        if ret_fn:
            return fn

        return pcm

    def stripe(self, data=None, fn=None, save=False, subsample=False, title=None,
               name=None, ext='png', popt=None, cb=True, cbl=None, zerocent=None,
               vmin=None, vmax=None, cmap=None, cbopt=None, fig=None, fopt=None, cax=None,
               ax=None, log=False, aspect=None, sdir=None, smooth=None, rmin=None, rmax=None, lbls=True,
               phi_shift=0, r_cut=None, phi_dot=0, ret_fn=False, rplot=1, dpi=300,
               figsize=None, overwrite=True, display=False, minmax=False, txt_opt=None,
               ps=None, mode=1, phi_norm=True, rm_last=False, printvmax=False, lnorm=False,
               dv=None):
        """Plot 2D sim data"""
        _fopt = dict(dpi=dpi, figsize=figsize)
        if fopt is None:
            fopt = {}
        _fopt.update(fopt)
        if popt is None:
            popt = {}
        if cbopt is None:
            cbopt = {}
        if rmin is None:
            rmin = self.r[0]
        if rmax is None:
            rmax = r_main(self.mach)
        rmax = min(rmax, self.r[-1])
        r = self.r[np.newaxis, :]
        phi = self.phi[:, np.newaxis] + phi_shift
        if self.t and phi_dot:
            phi -= phi_dot * self.t
        _popt = {}
        if data is None:
            data = self._defvar
        if hasattr(data, 'lower'):
            if data in ['pseudo', 'Rpseudo', 'mom1'] and r_cut is None:
                r_cut = 1.03
                if vmax is None and vmin is None:
                    vmin = '99.5%'
            if data in ['vorticity', 'vortensity']:
                if r_cut is None:
                    r_cut = 1.03
                if zerocent is None:
                    zerocent = True
                if vmax is None and vmin is None:
                    vmax = 'smart'
            if data in ['vi', 've']:
                vmin = '99%'
            if cbl is None:
                cbl = helpers.labeler(data)
                name = data

        if name:
            if name in [True, 1]:
                name = ''
            if title is None:
                # title = self.name + ' ' + self.t_str + ' ' + name
                title = self.name + ' ' + name
                if self.sim:
                    title = self.sim.name + r' $t/2\pi = {0:g}$'.format(self.t / tau)
            if save and fn is None:
                fn = self._prefix + '_' + name + '_stripe.' + ext

        if save or fn:
            save = True
            if fn is None:
                fn = self._prefix + '_stripe.' + ext
            if not sdir is None:
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                fn = os.path.join(sdir, fn)
        if parse_not_overwrite(overwrite, fn):
            if ret_fn:
                return fn
            return None
        if lnorm:
            if lnorm is True:
                if name in ['Rpseudo', 'pseudo']:
                    lnorm = -3 #int(np.round(-2 * np.log10(self.sim.mach) - 3))
                else:
                    lnorm = 0
        else:
            lnorm = 0
        if display:
            print('    Map of t/orb={:d}'.format(int(self.orbit + .5)))

        data = self._parse_data(data)
        if dv is not None:
            data -= dv
        if type(data) != np.ndarray:
            raise TypeError('Data has type "{:}", not ndarray.'.format(type(data)))
        data = data * 10**-lnorm
        if not smooth is None:
            data = self.smooth(data, smooth)

        # parse/set options and defaults
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
        except (TypeError, IndexError):
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
            fig = plt.figure(**_fopt)
            if name is None:
                name = True
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        if aspect:
            ax.set_aspect(aspect)

        # start plotting
        if phi_norm is True:
            phi_norm = np.pi
        if not phi_norm:
            phi_norm = 1
        if printvmax:
            if 'vmax' in _popt:
                print(_popt['vmax'])
        pcm = plt.pcolormesh(self.r, self.phi / phi_norm, data, **_popt)
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
        plt.xlim(rmin, rmax)
        plt.ylim(0, self.phi[-1] / phi_norm)
        if lbls:
            plt.xlabel('$r$')
            if phi_norm == np.pi:
                plt.ylabel(r'$\phi/\pi$')
            elif phi_norm == 1:
                plt.ylabel(r'$\phi$')
            elif phi_norm == 2 * np.pi:
                plt.ylabel(r'$\phi/2\pi$')
            else:
                plt.ylabel(r'$\phi/{:.3g}$'.format(phi_norm))
        if minmax:
            if r_cut is None:
                r_cut = 1.03
            dmin = data[:, self.rloc(r_cut):-5].min()
            dmax = data[:, self.rloc(r_cut):-5].max()
            txt = 'max: {0:.4g}\nmin: {1:.4g}'.format(dmax, dmin)
            x = .99 * self.r[-1]
            _txt_opt = dict(x=-x, y=x, ha='left', va='top')
            try:
                _txt_opt.update(txt_opt)
            except TypeError:
                pass
            plt.text(_txt_opt.pop('x'), _txt_opt.pop('y'), txt, **_txt_opt)
        if rplot:
            rplot = np.atleast_1d(rplot)
            for r in rplot:
                plt.axvline(r, lw=1, color='1', ls=':')
        if ps is not None:
            if ps is True:
                raise NotImplementedError
            tmp = lindblad_loc(ps, mode)
            plt.axvline(tmp[0], lw=1, color='1', ls='--')
            plt.axvline(tmp[1], lw=1, color='1', ls='-')
            plt.axvline(tmp[2], lw=1, color='1', ls='--')
        if title:
            plt.title(helpers.sanitize_lbl(title.format(**self.__dict__)))
        if cb:
            divider = make_axes_locatable(ax)
            if cax is None:
                cax = divider.append_axes("right", size="5%", pad=0.05)
            pos = cbopt.pop('pos', 'left')
            cb = plt.colorbar(pcm, cax=cax, **cbopt)
            if pos in ['left', 'right']:
                cb.ax.yaxis.set_offset_position(pos)
            elif pos == 'top':
                cb.ax.xaxis.set_label_position('top')
                cb.ax.xaxis.set_ticks_position('top')
            if cbl:
                if lnorm:
                    lntxt = '/10^{' + str(lnorm) + '}'
                    cbl = cbl.rstrip('$') + lntxt
                    if '$' in cbl:
                        cbl = cbl + '$'
                cb.set_label(cbl)
        if rm_last:
            xticks = ax.xaxis.get_major_ticks()
            xticks[-1].label1.set_visible(False)

        plt.sca(ax)
        if printvmax:
            print(plt.clim())
        # save fig
        if save:
            plt.savefig(fn)
            plt.close()
        if ret_fn:
            return fn

        return pcm

    def stripe_and_data(self, var=None, fn=None, save=False, sdir=None, overwrite=True,
                        dpi=300, figsize=None, fopt=None, ext='png', ret_fn=False,
                        rho0=None, omega0=True, **kwargs):
        if figsize is None:
            figsize = (5, 6)
        _fopt = dict(dpi=dpi, figsize=figsize)
        if fopt is None:
            fopt = {}
        _fopt.update(fopt)
        if save or fn:
            save = True
            if fn is None:
                fn = self._prefix + '_stripe_data.' + ext
            if not sdir is None:
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                fn = os.path.join(sdir, fn)
        if parse_not_overwrite(overwrite, fn):
            if ret_fn:
                return fn
            return None

        if np.all(omega0 == 'k'):
            omega0 = self.rc ** -1.5
        if omega0 is True or rho0 is True:
            df = self.sim.loadfile('cons', 0)
            if rho0 is True:
                rho0 = df['dens'].mean(axis=0)
            if omega0 is True:
                omega0 = df['vel2'].mean(axis=0) / self.rc

        fig = plt.figure(**_fopt)
        gs = mpl.gridspec.GridSpec(3, 2, height_ratios=[1, .2, .2], width_ratios=[1, .05],
                                   top=.95, bottom=.09, left=.13, right=.85, wspace=.01,
                                   hspace=.1)
        ax0 = plt.subplot(gs[0, 0])
        cax = plt.subplot(gs[0, 1])
        pcm = self.stripe(var, ax=ax0, cax=cax, lbls=False, **kwargs)
        xlim = ax0.get_xlim()
        #print(xlim)
        plt.ylabel(r'$\phi/\pi$')
        ax0.set_xticklabels([])

        ax = plt.subplot(gs[1, 0])
        omega = self.vel(2).mean(axis=0) / self.rc
        plt.plot(self.rc, omega, 'k')
        #ylim = plt.ylim()
        if omega0 is not None:
            plt.plot(self.rc, omega0, lw=1, c='.5', ls=':')
            #plt.legend([r'$\Omega$', r'$\Omega_{\rm K}$'])
        plt.ylabel(r'$\Omega$')
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
        ax.set_xticklabels([])
        plt.xlim(*xlim)
        plt.ylim(None, 1)

        ax = plt.subplot(gs[2, 0])
        dens = self['dens'].mean(axis=0)
        plt.plot(self.rc, dens, 'k')
        if rho0 is not None:
            plt.plot(self.rc, rho0, lw=1, c='.5', ls=':')
            #plt.legend([r'$\rho$', r'$\rho_0$'])
        plt.ylabel(r'$\rho$')
        plt.xlabel(r'$r$')
        plt.ylim(0, 3)
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.5))
        plt.xlim(*xlim)

        # save fig
        if save:
            plt.savefig(fn)
            plt.close()
        if ret_fn:
            return fn

        return fig

    def smooth(self, data, width=64):
        return smooth(self._parse_data(data), width=width)

    def ddr(self, data):
        nm1 = len(data.shape) - 1
        loc = [slice(None)] * nm1
        left = loc + [slice(0, -1)]
        right = loc + [slice(1, None)]
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
        plt.ylabel(r'$\left<\rho({:.1f}\times 2\pi)\right>-\left<\rho(0)\right>$'.format(
            self.t / tau))


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

        loc = (slice(None, mmax + 1), slice(None))
        ft = (np.abs(self.fft(self._parse_data(var))) ** 2)[loc]
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

        # save fig
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
        sqr = data ** 2
        loc = sqr.argmax(axis=0)
        phase = self.phic[loc]
        phase -= .5 * (1 - np.sign(data[loc, np.arange(data.shape[1])])) * np.pi
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

        # save fig
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
        ft = self.fft('pseudo')[:mmax + 1, :]
        phi = -np.angle(ft)
        a = np.abs(ft)
        # tmp = np.ma.array(a, mask=a - .5 * a.max(axis=0)[np.newaxis, :] > 0)
        # m = tmp.mean(axis=0)
        # s = tmp.std(axis=0)
        # loc = np.where(a - (m + 3 * s)[np.newaxis, :] > 0)
        # modes = np.array(sorted(set(loc[0])))
        # md = {modes[i]: i for i in range(len(modes))}
        # amp = np.ones(a.shape) * np.nan
        # ang = amp.copy()
        # amp[loc] = a[loc]
        # ang[loc] = phi[loc]
        # return amp, ang
        return a, phi

    def _main_plots(self, save=True, ext='png', sdir=None):
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
            data = self[data + '1'], self[data + '2']
        x, y = data
        y *= self.rc[np.newaxis, :]
        return (grad(self.rc, y, axis=1) - self.ddphi(x)) / self.rc[np.newaxis, :]

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
        phi = spiral(r, rp, 1. / self.mach) + phi0
        print(phi)
        plt.plot(r * np.cos(phi), r * np.sin(phi), **opt)


class BLConsPrim(BL3Dfile):
    def rhobar(self):
        return self['dens'].mean(axis=0)

    def Mdot(self):
        return self['dens'] * self['vel1']

    def v1v2(self):
        return self['vel1'] * self['vel2']

    def vorticity(self, dvphi=False):
        if dvphi:
            out = self['vel2']
            out -= self.rhoWeight(out)[np.newaxis, :]
            out = self.curl((self['vel1'], out))
        else:
            out = self.curl('vel')
        # out[:,0] = out[:, 1]
        # out[:,-1] = out[:,-2]
        return out

    def vortensity(self, dvphi=False):
        return self.vorticity(dvphi=dvphi) / self['dens']

    def d_vortensity(self):
        ve = self.vortensity()
        return ve - ve.mean(axis=0)

    def plt_vortensity(self, init=None, fopt=None, vmax=None, fig=None, sdir=None,
                       fn=None, save=False, overwrite=True, ext='png'):
        if save or fn:
            save = True
            if fn is None:
                fn = self._prefix + '_delta_vortensity.' + ext
            if not sdir is None:
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                fn = os.path.join(sdir, fn)
        if parse_not_overwrite(overwrite, fn):
            return init

        if init is None:
            if self.t == 0:
                init = self.vortensity().mean(axis=0)
            else:
                init = self.sim.loadfile('cons', 0).vortensity().mean(axis=0)
        dv = (self.vortensity() - init[np.newaxis, :]) * self.rc[np.newaxis, :] ** 2
        if vmax is True:
            vmax = dv.mean(axis=0)[self.rloc(1):self.rloc(3)].max()

        if fopt is None:
            fopt = dict()
        if fig is None:
            _fopt = dict(figsize=(5, 5), dpi=300)
            _fopt.update(fopt)
            fig = plt.figure(**_fopt)

        gs = mpl.gridspec.GridSpec(2, 2, height_ratios=[1, .22], width_ratios=[1, .05],
                                   top=.95, bottom=.09, left=.13, right=.85, wspace=.01,
                                   hspace=.15)
        ax0 = plt.subplot(gs[0, 0])
        cax = plt.subplot(gs[0, 1])
        pcm = self.plot2d(dv, fig=fig, ax=ax0, vmax=vmax, cb=False, name=True,
                          zerocent=True)
        cb = plt.colorbar(pcm, ax=ax0, cax=cax)
        lbl = r'$R^2\left(\omega/\rho-\left.\left<\omega/\rho\right>_\phi\right|_0\right)$'
        cb.set_label(lbl)
        pos0 = np.array(ax0.get_position())
        posc = np.array(cax.get_position())
        cax.set_position(
            [posc[0, 0], pos0[0, 1], posc[1, 0] - posc[0, 0], pos0[1, 1] - pos0[0, 1]])
        ax = plt.subplot(gs[1, 0])
        plt.plot(self.rc, dv.mean(axis=0))
        ylim = plt.ylim()
        plt.ylim(max(-1, ylim[0]), min(dv.mean(axis=0)[5:-5].max() * 1.1, ylim[1]))
        plt.axhline(0, c='k', lw=1, ls=':')
        plt.xlabel(r'$R$')
        plt.ylabel(lbl)

        if save:
            plt.savefig(fn)
            plt.close()

        return init

    def omega(self):
        return self['vel2'] / self.rc[np.newaxis, :]

    def vi(self):
        return self.rc[np.newaxis, :] ** 2 * self.vorticity(True)

    def ve(self):
        return self.rc[np.newaxis, :] ** 2 * self.vortensity(True)

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
        return self.rc ** 2 * (self['mom1'] * self['vel2']).mean(axis=0) * tau

    # BRS12 ApJ 760:22
    def CL20(self):
        return tau * self.rc ** 2 * self['dens'].mean(axis=0) * self.rhoWeight(
            self.deltaVal('vel1') * self.deltaVal('vel2'))

    def CL24(self):
        return tau * self.rc ** 2 * self['dens'].mean(axis=0) * self.rhoWeight(
            np.abs(self.deltaVal('vel1') * self.deltaVal('vel2')))

    def CL25(self, Op, M=None):
        if M is None:
            M = self.mach
        s = 1. / M
        dens = self['dens']
        Mdens = dens.mean(axis=0)
        return self.rc * s ** 3 * tau * np.mean((dens - Mdens[np.newaxis, :]) ** 2,
                                                axis=0) / (
                       dens.mean(axis=0) * (self.Ok() - Op))

    def CS(self):
        dens = self['dens']
        Mdens = dens.mean(axis=0)
        dv2 = self['vel2'] - self.Oloc() * self.rc
        return tau * self.rc ** 2 * Mdens * self.rhoWeight(dv2 * self['vel1'])

    def CA(self):
        dens = self['dens']
        Mdens = dens.mean(axis=0)
        return tau * self.rc ** 3 * Mdens * self.Oloc() * self.rhoWeight('vel1')

    def CLplot(self, Op=None, M=None, nm=6):
        cl = [self.CL(), self.CL24()]
        lbls = ['BRS12 Eqn %d' % i for i in [19, 24]]
        if not Op is None:
            cl.append(self.CL25(Op, M))
        cl += [self.CS(), self.CA()]
        lbls += ['BRS13 $C_S$', 'BRS13 $C_A$']
        tmp = self.CSm()
        # s = tmp.sum(axis=0)
        # loc = sorted(range(tmp.shape[0] // 2), key=lambda x: 1/s[x])
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
        return self.rc ** -1.5

    def Oloc(self):
        return self.rhoWeight('vel2') / self.rc

    def pseudo(self):
        err = 'Must be implemented in subclase of {0:}.'.format(self.__class__)
        raise NotImplementedError(err)

    def Rpseudo(self):
        return self.rc[np.newaxis, :] * self.pseudo()

    def CSm(self, delta=False):  # dvm=None, dum=None, dSm=None, o=None):
        rho = self['dens']
        S = rho.mean(axis=0)[np.newaxis, :]
        if delta:
            norm = 1. / (self.phic.size - 1)
            dvm = self.fft(self.deltaVal('vel1')) * norm
            dum = self.fft(self.deltaVal('vel2')) * norm
        else:
            dvm = self.fft('vel1') / self.phic.size
            dum = self.fft('vel2') / self.phic.size
        r = self.rc[np.newaxis, :]
        return .5 * np.pi * r ** 2 * S * (np.conj(dvm) * dum + dvm * np.conj(dum))

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
        print("norm:", norm, "post-norm:", np.median(np.abs(pfft) / np.abs(afft)))
        dmag = (np.abs(pfft) - amp) / np.abs(pfft)
        dth = (np.angle(pfft) + np.angle(afft))  # % tau

        fig = plt.figure(figsize=(8, 8))
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

        im0 = plt.imshow(data, vmin=avmin, vmax=avmax, interpolation='nearest',
                         norm=mpl.colors.LogNorm())
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

        im1 = plt.imshow(data1, vmin=bvmin, vmax=bvmax, interpolation='nearest',
                         norm=mpl.colors.LogNorm())
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

    def wave_power(self):
        data = self['vel1'].mean(axis=0)
        data[self.rc <= 1.5] = 0
        data[self.rc > 3.9] = 0
        return self.intr(data ** 2)

    def wave_power_2(self):
        data = self['pseudo']
        mean = data.mean(axis=0)
        delta = np.mean((data - mean[np.newaxis, :]) ** 2, axis=0)
        mean[self.rc <= 1.5] = 0
        mean[self.rc > 3.9] = 0
        delta[self.rc <= 1.5] = 0
        delta[self.rc > 3.9] = 0
        return self.intr(mean ** 2) - self.intr(delta)

    def vortensity_prof(self):
        return self.vortensity().mean(axis=0)

    def d_vortensity_prof(self, init=None):
        if init is None:
            if self.t == 0:
                init = self.vortensity().mean(axis=0)
            else:
                init = self.sim.loadfile('cons', 0).vortensity().mean(axis=0)
        dv = (self.vortensity() - init[np.newaxis, :]) * self.rc[np.newaxis, :] ** 2
        return dv.mean(axis=0)

    def flux_est(self, i=0, p=1, phase=0, rpow=-3, total=True):
        flux = self['dens']**p
        flux *= np.diff(self.phi)[:, None] * self.rc[None, :] * np.diff(self.r)[None, :]
        x = np.cos(self.phic - phase)[:, None] * self.rc[None, :]
        y = np.sin(self.phic - phase)[:, None] * self.rc[None, :]
        rloc = np.where(self.rc < 1)[0].max()
        flux[:, :rloc+1] = 0
        rho = np.sqrt(1 - np.minimum(y**2, 1)) / np.cos(i)
        flux[np.logical_and(x > 0, x < rho)] = 0
        flux *= self.rc[None, :] ** rpow
        if total:
            return flux.sum()
        return flux

    def integrate_spiral(self, omega_p, phi0=0, r0=1, nan_out=False):
        omega = self['mom2'].mean(axis=0) / self['dens'].mean(axis=0) / self.rc
        integrand = (omega - omega_p) * self.mach * np.diff(self.r)
        phi = integrand.cumsum()
        ri = self.rloc(r0)
        phi += phi0 - phi[ri]
        if nan_out:
            phi[:ri] = np.nan
        return phi

    def draw_arm(self, omega_p, phi0=0, r0=1, nan_out=True, popt=None, cart=True,
                 phi_norm=1):
        if popt is None:
            popt = dict()
        _popt = dict(c='w', lw=1, ls=':')
        _popt.update(popt)
        phi = self.integrate_spiral(omega_p, phi0=phi0, r0=r0, nan_out=nan_out)
        r = self.rc
        if cart:
            plt.plot(r * np.cos(phi), r * np.sin(phi), **_popt)
        else:
            plt.plot(r, (phi % tau) / phi_norm, **_popt)

    def bl_width(self, rl=.25, ru=.75):
        omega = self['mom2'].mean(axis=0) / (self['dens'].mean(axis=0) * self.rc)
        ri = omega.argmax()
        out = np.empty((2))
        omax = omega.max()
        tmp = omega.copy()
        tmp[ri:] = 0
        out[0] = self.sim.rc[np.abs(tmp - rl * omax).argmin()]
        out[1] = self.sim.rc[np.abs(tmp - ru * omax).argmin()]
        return out

    def plateau(self, frac=.9):
        omega = self['mom2'].mean(axis=0) / (self['dens'].mean(axis=0) * self.rc)
        ri = omega.argmax()
        out = np.empty((2))
        omax = omega.max()
        inner = omega.copy()
        outer = omega.copy()
        inner[ri:] = 0
        outer[:ri] = 0
        out[0] = self.sim.rc[np.abs(inner - frac * omax).argmin()]
        out[1] = self.sim.rc[np.abs(outer - frac * omax).argmin()]
        return out

    def bl_stats(self, fn=None):
        omega = self['mom2'].mean(axis=0) / (self['dens'].mean(axis=0) * self.rc)
        out = dict()
        out['bl'] = self.bl_width()
        out['plateau'] = self.plateau()
        ri = omega.argmax()
        out['peak'] = self.sim.rc[ri]
        out['omax'] = omega.max()
        out['d1omega'] = 1 - out['omax']
        out['dkomega'] = out['peak']**-1.5 - out['omax']
        return out


class BLcons(BLConsPrim):
    def _special_keys(self, key):
        if key[:3] == 'vel' and len(key) == 4:
            return self.vel(key[3])
        if key == 'dens**2':
            return self['dens'] ** 2
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
            return self['dens'] ** 2
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
        # super().__init__(*args, **kwargs)
        # self.data = {}
        # self.data.update(self)

    def _trim(self, data):
        data = super(BLFT, self)._trim(data)
        # data = super()._trim(data)
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
                    i = self.sim.inputs['meshblock']['nx2'] - 1
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
        if key == 'CS':
            return self.CS()
        if key == 'CS_RRR':
            return self.CS_RRR()
        return None

    def CS(self):
        r2 = tau * self.rc[np.newaxis, :] ** 2
        u = self['FT-vel2']
        return r2 * (self['FT-CL'] - u[0][np.newaxis, :] * self['FT-Mdot'])

    def CS_RRR(self):
        r2 = (tau * self.rc[np.newaxis, :] ** 2).astype(np.float32)
        u = self['FT-vel2']
        v = self['FT-vel1']
        return r2 * np.real(self['FT-dens'][0])[np.newaxis, :] * (
                np.conj(v) * u + np.conj(u) * v)

    def fluxes(self):
        u = self['FT-vel2']
        r2 = tau * self.rc[np.newaxis, :] ** 2
        out = {'CL': r2 * self['FT-CL'],
               'CA': r2 * u[0][np.newaxis, :] * self['FT-Mdot']}
        for i in ['Mdot', 'dens']:
            out[i] = np.real(self['FT-' + i][0])
        out['CS'] = out['CL'] - out['CA']
        v = self['FT-vel1']
        out['CSm'] = r2 * np.real(self['FT-dens'][0])[np.newaxis, :] * (
                np.conj(v) * u + np.conj(u) * v)
        out['drho'] = np.real(self['FT-dens'][0]) - self.sim.rho_ref
        out['vphi'] = np.real(u[0])
        out['vr'] = np.real(v[0])
        return out

    def flux_data(self):
        retrieved = []
        def _get(var):
            try:
                out = np.real(self['FT-' + var + '-Re'][0])
                retrieved.append(var)
                return out
            except KeyError:
                return None
        r2 = (tau * self.rc ** 2).astype(np.float32)
        cl = r2 * _get('CL')
        u = _get('vel2')
        md = _get('Mdot')
        v = _get('vel1')
        ca = r2 * u * md
        cs = cl - ca
        d = _get('dens')
        d2 = _get('dens**2')
        dd = d2 / d ** 2 - 1.0
        rest = [_get(i) for i in ['pseudo', 'v1v2', 'vortensity', 'rhov2']
                if 'FT-' + i + '-Re' in self]
        remaining = [i for i in self if 'FT-' in i and '-Re' in i]
        remaining = [i[3:-3] for i in remaining if i[3:-3] not in retrieved]
        rest += [_get(i) for i in remaining]
        return np.array([cs, ca, cl, md, dd, d, v, u] + rest)

    def flux_variables(self):
        used = ['CL', 'vel2', 'Mdot', 'vel1', 'dens', 'dens**2']
        out = ['CS', 'CA', 'CL', 'Mdot', 'dd', 'dens', 'vr', 'vphi']
        out += [i for i in ['pseudo', 'v1v2', 'vortensity', 'rhov2']
                if 'FT-' + i + '-Re' in self]
        used = ['CL', 'vel2', 'Mdot', 'vel1', 'dens', 'dens**2'] + out
        ft_keys = [i for i in self if 'FT-' in i and '-Re' in i]
        return out + [i[3:-3] for i in ft_keys if i[3:-3] not in used]

    def wave_power(self):
        data = np.real(self['FT-vel1'][0])
        data[self.rc <= 1] = 0
        return self.intr(data ** 2)

    def vortensity_prof(self):
        return np.real(self['FT-vortensity-Re'][0])

def _parse_file(fn, file_handle=None):
    ext = fn.split('.')[-1]
    if ext == 'athdf':
        return BLfile(fn, file_handle=file_handle)
    if ext == 'npy':
        return np.load(fn)[()]
    raise IOError('Cannot identify file type of "{0:}"'.format(fn))

def loadBLfile(fn, **kwargs):
    fn = findAbsPath(fn, kwargs.get('sim_path', None))
    #data = _parse_file(fn, file_handle=kwargs.get('file_handle', None))
    _fn = fn
    if 'file_handle' in kwargs:
        if kwargs['file_handle']:
            _fn = kwargs['file_handle']
    data = athdf(_fn, face_func_2=kwargs.get('x2_face'), return_levels=True)
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
    # raise RuntimeError
    return BLfile(fn, data=data, **kwargs)
