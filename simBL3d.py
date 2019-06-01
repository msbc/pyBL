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
import subprocess

from .parmap import parmap
from . import athena_read as ar
from . import helpers
from . import simBLclass as blc

tau = 2 * np.pi
#hpi = .5 * np.pi
hpi = 1.5708
zero = 0.

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
    if 0:
        ax.plot(ax1_r * np.sin(ax1_th), ax1_r * np.cos(ax1_th), 'k-', lw=1)
    else:
        for i in range(ax1_r.size - 1):
            line_plt((ax1_r[i], ax1_th[i]), (ax1_r[i+1], ax1_th[i+1]), c='k', lw=1)
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

class BLfile(blc.BLfileBase):
    def __init__(self, fn, sim_path=None, t=None, data=None, defvar=None, ai_data=None, sim=None,
                 file_handle=None, x2_face=None, num_ghost=0):
        if x2_face is None and sim is not None:
            x2_face = sim.x2_face
        super(BLfile, self).__init__(fn, sim_path=sim_path, t=t, data=data, defvar=defvar, ai_data=ai_data, sim=sim,
                                     file_handle=file_handle, x2_face=x2_face, trim=False, num_ghost=num_ghost)
        self.r = self['x1f']
        self.dr = self.r[1:] - self.r[:-1]
        self.rc = .5 * self.r[1:] + .5 * self.r[:-1]
        self.theta = self['x2f']
        self.thetac = .5 * self.theta[1:] + .5 * self.theta[:-1]
        self.dtheta = .5 * self.theta[1:] + .5 * self.theta[:-1]
        self.phi = self['x3f']
        self.phic = .5 * self.phi[1:] + .5 * self.phi[:-1]
        self._grid_shape = self.phic.size, self.thetac.size, self.rc.size
        na = np.newaxis
        self.coord = self.phic[:, na, na], self.thetac[na, :, na], self.rc[na, na, :]
        #self.grid = helpers.ndmesh(self.phi, self.theta, self.r)
        #p, t, r = self.grid
        #self.x = r * np.sin(t) * np.cos(p)
        #self.y = r * np.sin(t) * np.sin(p)
        #self.z = r * np.cos(t)

    def _special_keys(self, key):
        if key[:3] == 'vel' and len(key) == 4:
            return self.vel(key[3])
        if key == 'dens**2':
            return self['dens']**2
        if key[0:4] == 'fft-':
            try:
                return self.fft(self[key[4:]])
            except KeyError:
                pass
        if key[0:2] == 'd-':
            try:
                out = self[key[2:]]
                return out - self.phi_mean(out)
            except KeyError:
                pass
        return None

    def ddr(self, data):
        data = self._parse_data(data)
        return helpers.grad(self.rc, data, 2)

    def ddtheta(self, data):
        data = self._parse_data(data)
        return helpers.grad(self.thetac, data, 1)

    def ddphi(self, data, axis=0):
        data = self._parse_data(data)
        return (np.roll(data, -1, axis=axis) - np.roll(data, 1, axis=axis)) / (self.phic[2] - self.phic[0])

    def v_del_vr(self):
        v1 = self['vel1']
        v2 = self['vel2']
        v3 = self['vel3']
        ph, th, r = self.coord
        return v1 * self.ddr(v1) + (v2 * self.ddtheta(v1) + v3 * self.ddphi(v1) / np.sin(th) - v2**2 - v3**2) / r

    def grad(self, data):
        data = self._parse_data(data)
        ph, th, r = self.coord
        return (self.ddphi(data) / (r * np.sin(th)), self.ddtheta(data) / r, self.ddr(data))

    def div(self, data):
        data = self._parse_data(data)
        ph, th, r = self.coord
        st = np.sin(th)
        return (self.ddr(r**2 * data) / r + (self.ddtheta(data * st) + self.ddphi(data)) / st) / r

    def vel(self, i):
        return self['mom{0:}'.format(i)] / self['dens']

    def pseudo(self):
        return self['mom1'] / np.sqrt(self['dens'])

    def Rpseudo(self):
        return self.rc[np.newaxis, :] * self.pseudo()

    def axis_loc(self, axis, value):
        return np.argmin(np.abs([self.phic, self.thetac, self.rc][axis] - value))

    def theta_loc(self, theta):
        return self.axis_loc(1, theta)

    def phi_loc(self, phi):
        return self.axis_loc(1, phi)

    def fft(self, data, axis=0, mag=False):
        try:
            data.shape
        except AttributeError:
            data = self[data]
        out = np.fft.rfft(data, axis=axis)
        if mag:
            out = np.absolute(out)
        return out

    def theta_mean(self, data, weight=1, shape=False):
        data = self._parse_data(data)
        if hasattr(weight, 'lower'):
            weight = self[weight]
        dtheta = np.diff(self.theta)[np.newaxis, :, np.newaxis]
        out = (data * weight * dtheta).mean(axis=1) / (weight * dtheta).mean(axis=1)
        if shape:
            out = out[:,np.newaxis,:]
        return out

    def r_mean(self, data, weight=1, shape=False):
        data = self._parse_data(data)
        if hasattr(weight, 'lower'):
            weight = self[weight]
        dr = np.diff(self.r)[np.newaxis, np.newaxis, :]
        out = (data * weight * dr).mean(axis=2) / (weight * dr).mean(axis=2)
        if shape:
            out = out[:,:,np.newaxis]
        return out

    def phi_mean(self, data, weight=1, shape=True):
        data = self._parse_data(data)
        if hasattr(weight, 'lower'):
            weight = self[weight]
        dphi = np.diff(self.phi)[:, np.newaxis, np.newaxis]
        out = (data * weight * dphi).mean(axis=0) / (weight * dphi).mean(axis=0)
        if shape:
            out = out[np.newaxis,:,:]
        return out

    def midplane(self, data):
        data = self._parse_data(data)
        i = np.argmax(self.thetac[self.thetac < .5 * np.pi])
        return data[:, i:i + 2, :].mean(axis=1)

    def pres(self):
        return self['dens'] * self.sim.mach**-2

    def mom_r(self, pre=False, post=False, dvdt=None):
        v1 = self['vel1']
        if dvdt is None:
            if pre is not False:
                dl = self.t - pre.t
                if post is not False:
                    Dinv = (post.t - pre.t)**-1
                    dr = post.t - self.t
                    rat = dr / dl
                    dvdt = (dl**-1 - dr**-1) * v1 + Dinv / rat * post['vel1'] - rat * Dinv * pre['vel1']
                else:
                    dvdt = (v1 - pre['vel1']) / dl
            elif post is not False:
                dvdt = (post['vel1'] - v1) / (post.t - self.t)
        d = self['dens']
        return [dvdt, self.v_del_vr(), self.ddr(d) / (d * self.sim.mach**2), 1. / self.coord[2]**2]

    def wave_power(self):
        data = self.midplane('vel1').mean(axis=0)
        data[self.rc <= 1.5] = 0
        data[self.rc > 3.9] = 0
        return self.intr(data**2)

    def wave_power_2(self):
        loc1 = slice(np.argmin(np.abs(self.rc - 1.2)), np.argmin(np.abs(self.rc - 3.8)) + 1)
        dx1 = np.diff(self.r)
        pi7 = np.pi / 7.
        deg90 = np.pi * .5
        loc2 = slice(np.argmin(np.abs(self.thetac - deg90 + pi7)), np.argmin(np.abs(self.thetac - deg90 - pi7)) + 1)
        dx2 = np.diff(self.theta)
        p = tau * self['pseudo'].mean(axis=0)
        na = np.newaxis
        da = dx1[na, loc1] * dx2[loc2, na] * self.rc[na, loc1]**2 * np.sin(self.thetac[loc2, na])
        return np.sum(p[loc2, loc1]**2 * da) / np.sum(da)

    def swp(self):
        out = self['pseudo']
        #out[self['dens'] == self['dens'].min()] = 0
        out = out.mean(axis=(0,1))**2
        out[self.rc > 1.1] = 0
        return self.intr(out)

    def rprof(self, data):
        return self._parse_data(data).mean(axis=(0,1))

    def _plot_slice(self, sim_slice, data=None, fn=None, save=False, subsample=False, title=None,
                   name=None, ext='png', popt=None, cb=True, cbl=None, zerocent=None,
                   vmin=None, vmax=None, cmap=None, cbopt=None, fig=None, fopt=None,
                   ax=None, log=False, aspect=1, sdir=None, smooth=None,
                   phi_shift=0, r_cut=None, phi_dot=0, ret_fn=False, rplot=1):
        if fopt is None:
            fopt = {}
        if popt is None:
            popt = {}
        if cbopt is None:
            cbopt = {}
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

        tmp = list(range(3))
        for i, s in enumerate(sim_slice):
            if (not s) and (s != 0):
                s = None
            if type(s) == float:
                s = self.axis_loc(i, s)
            elif s is None:
                s = slice(s)
            if s is not None:
                tmp.remove(i)
            sim_slice[i] = s

        if len(data.shape) == 3:
            data_slice = data[sim_slice]
        elif len(data.shape) == 2:
            data_slice = data

        axes = []
        for i, axis in enumerate([self.phic, self.thetac, self.rc]):
            a = axis[sim_slice[i]]
            if a.shape == axis.shape:
                axes.append([self.phi, self.theta, self.r][i])
            else:
                axes.append(a)

        if axes[0].shape == self.phi.shape:
            axes[0] = axes[0][:, np.newaxis]

        print(sim_slice)
        return data_slice.shape

    def _data_opt_parser(self, data=None, vmin=None, vmax=None, r_cut=None, cbl=None, zerocent=None, name=None):
        if data is None:
            data = self._defvar
        if hasattr(data, 'lower'):
            if data in ['pseudo', 'Rpseudo'] and r_cut is None:
                r_cut = .85
                if vmax is None and vmin is None:
                    vmin = 'smart'
                if zerocent is None:
                    zerocent = True
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
        return data, dict(vmax=vmax, vmin=vmin, zerocent=zerocent, cbl=cbl, name=name, r_cut=r_cut)

    def _opt_parser(self, data=None, vmin=None, vmax=None, r_cut=None, cbl=None, zerocent=None, name=None,
                         fopt=None, popt=None, cbopt=None, log=None, cmap=None, title=None):
        if fopt is None:
            fopt = {}
        if popt is None:
            popt = {}
        if cbopt is None:
            cbopt = {}
        _popt = {}

        if name:
            if name in [True, 1]:
                name = ''
            if title is None:
                title = self.name + ' ' + name
                if self.sim:
                    title = self.sim.name + r' $t/2\pi = {0:g}$'.format(self.t / tau)

        if log:
            _popt['norm'] = mpl.colors.LogNorm()

        # parse smart lim options
        tmp = {}
        try:
            if '%' == vmin[-1]:
                tmp['low'] = float(vmin[:-1])
                vmin = 'smart'
        except (TypeError, IndexError):
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
                rloc = slice(self.rloc(r_cut), None)
            try:
                tmp = helpers.smartlim(data[:, rloc], **tmp)
            except (TypeError, IndexError):
                tmp = helpers.smartlim(data, **tmp)
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

        return dict(vmax=vmax, vmin=vmin, zerocent=zerocent, cbl=cbl, name=name, log=log, fopt=fopt, popt=_popt,
                    cbopt=cbopt, r_cut=r_cut, title=title)

    def _labler(self, ax, artist, title=None, cb=None, cbl=None, cbopt=None):
        if cbopt is None:
            cbopt = {}
        if title:
            plt.title(helpers.sanitize_lbl(title.format(**self.__dict__)))
        if cb:
            if 'cax' not in cbopt:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = plt.colorbar(artist, cax=cax, **cbopt)
            else:
                cb = plt.colorbar(artist, **cbopt)
            cb.ax.yaxis.set_offset_position('left')
            if cbl:
                cb.set_label(cbl)
        return

    def _save_fig(self, fn, def_fn, sdir=None):
        if fn is None:
            fn = def_fn
        if not sdir is None:
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
        plt.savefig(fn)
        plt.close()
        return fn

    def _fig_ax(self, fig=None, ax=None, aspect=None, fopt=None):
        if fopt is None:
            fopt = {}
        if fig is None and ax is None:
            fig = plt.figure(**fopt)
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        if aspect:
            ax.set_aspect(aspect)
        return fig, ax

    def fft_plot(self, data, weight=None, mmax=31, fn=None, save=False, title=None,
                   name=None, ext='png', popt=None, cb=True, cbl=None, zerocent=None,
                   vmin=None, vmax=None, cmap=None, cbopt=None, fig=None, fopt=None,
                   ax=None, log=False, aspect=1, sdir=None, run_fft=None,
                   r_cut=None, ret_fn=False, rplot=1):
        data, opt = self._data_opt_parser(data=data, vmax=vmax, vmin=vmin, zerocent=zerocent, cbl=cbl, name=name,
                                          r_cut=r_cut)
        data_slice = data
        if data.ndim == 3:
            if weight is None:
                weight = 'dens'
            if run_fft is None:
                run_fft = True
        try:
            if weight:
                data_slice = self.theta_mean(data, weight=weight)
        except ValueError:
            data_slice = self.theta_mean(data, weight=weight)
        if run_fft:
            data_slice = self.fft(data, mag=True)
        opt = self._opt_parser(data=data_slice, log=log, fopt=fopt, popt=popt, cbopt=cbopt, cmap=cmap, title=title,
                               **opt)

        x = self.r[np.newaxis, :]
        y = np.arange(mmax + 1)[:, np.newaxis]
        fig, ax = self._fig_ax(fig, ax, aspect=aspect, fopt=opt['fopt'])
        pcm = ax.pcolormesh(x, y, data_slice, **opt['popt'])
        if rplot:
            rplot = np.atleast_1d(rplot)
            for r in rplot:
                plt.plot(r * np.cos(self.phic), r * np.sin(self.phic), lw=1, c='1', ls=':')
        self._labler(ax, pcm, cb=cb, **{k: opt.get(k) for k in ['title', 'cbl', 'cbopt']})
        plt.sca(ax)
        plt.xlabel('$R$')
        plt.ylabel('$m$')
        if save or fn:
            fn = self._save_fig(fn, self._prefix + '_r-theta_plot.' + ext, sdir=sdir)
        if ret_fn:
            return fn

        return pcm

    def plot_rprof(self, data=None, theta=None, fn=None, save=False, title=None, zerocent=False,
                   name=None, ext='pdf', popt=None, fig=None, fopt=None, ret_fn=False,
                   ax=None, log=False, aspect=None, sdir=None, xlim=None, ylim=None):
        data, opt = self._data_opt_parser(data=data, name=name)
        data = self.rprof(data)
        opt = self._opt_parser(data=data, log=log, fopt=fopt, popt=popt, title=title,**opt)
        fig, ax = self._fig_ax(fig, ax, aspect=aspect, fopt=opt['fopt'])
        for i in ['vmin', 'vmax', 'cmap', 'norm']:
            try:
                opt['popt'].pop(i)
            except KeyError:
                pass
        if opt.get('log'):
            line = plt.semilogy(self.rc, data, **opt['popt'])
        else:
            line = plt.plot(self.rc, data, **opt['popt'])
        plt.xlim(self.r[0], self.r[-1])
        if xlim:
            plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)
        if zerocent:
            ylim = np.abs(plt.ylim()).max()
            plt.ylim(-ylim, ylim)
        self._labler(ax, line, cb=False, **{k: opt.get(k) for k in ['title', 'cbl', 'cbopt']})

        plt.sca(ax)
        #save fig
        if save or fn:
            fn = self._save_fig(fn, self._prefix + '_rprof_plot.' + ext, sdir=sdir)
        if ret_fn:
            return fn
        return

    def r_phi_plot(self, data=None, theta=None, fn=None, save=False, title=None,
                   name=None, ext='png', popt=None, cb=True, cbl=None, zerocent=None,
                   vmin=None, vmax=None, cmap=None, cbopt=None, fig=None, fopt=None,
                   ax=None, log=False, aspect=1, sdir=None,
                   phi_shift=0, r_cut=None, phi_dot=0, ret_fn=False, rplot=1):
        data, opt = self._data_opt_parser(data=data, vmax=vmax, vmin=vmin, zerocent=zerocent, cbl=cbl, name=name,
                                          r_cut=r_cut)
        if theta is None:
            i = np.argmax(self.thetac[self. thetac <.5 * np.pi])
            data_slice = data[:, i:i+2, :].mean(axis=1)
            theta = .5 * np.pi
        else:
            if type(theta) == float:
                theta = self.theta_loc(theta)
            data_slice = data[:, theta, :]
            theta = self.thetac[theta]
        opt = self._opt_parser(data=data_slice, log=log, fopt=fopt, popt=popt, cbopt=cbopt, cmap=cmap, title=title,
                               **opt)

        r = self.r[np.newaxis, :]
        phi = self.phi[:, np.newaxis] + phi_shift + self.t * phi_dot
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)

        fig, ax = self._fig_ax(fig, ax, aspect=aspect, fopt=opt['fopt'])

        #start plotting
        pcm = plt.pcolormesh(x, y, data_slice, **opt['popt'])
        if rplot:
            rplot = np.atleast_1d(rplot)
            for r in rplot:
                plt.plot(r * np.cos(self.phic), r * np.sin(self.phic), lw=1, c='1', ls=':')
        self._labler(ax, pcm, cb=cb, **{k: opt.get(k) for k in ['title', 'cbl', 'cbopt']})

        plt.sca(ax)
        #save fig
        if save or fn:
            fn = self._save_fig(fn, self._prefix + '_r-phi_plot.' + ext, sdir=sdir)
        if ret_fn:
            return fn

        return pcm

    def r_theta_plot(self, data=None, phi=None, fn=None, save=False, title=None, name=None,
                     ext='png', popt=None, cb=True, cbl=None, zerocent=None, vmin=None, vmax=None,
                     cmap=None, cbopt=None, fig=None, fopt=None, ax=None, log=False, aspect=1,
                     sdir=None, r_cut=None, ret_fn=False, rplot=1, both=False, xs=1, lim=None):
        if xs is None:
            xs = 1
        if abs(xs) != 1:
            raise ValueError('|xs| must be 1.')
        data, opt = self._data_opt_parser(data=data, vmax=vmax, vmin=vmin, zerocent=zerocent, cbl=cbl, name=name,
                                          r_cut=r_cut)
        if data.ndim == 3:
            if phi is None or phi == 'mean':
                data_slice = data.mean(axis=0)
                phi = 0.
                if both:
                    raise ValueError('Cannot use mean and both together.')
            else:
                if type(phi) == float:
                    phi = self.phi_loc(phi)
                data_slice = data[phi, :, :]
                phi = self.phic[phi]
        else:
            data_slice = data.copy()
        opt = self._opt_parser(data=data_slice, log=log, fopt=fopt, popt=popt, cbopt=cbopt, cmap=cmap, title=title,
                               **opt)

        r = self.r[np.newaxis, :]
        theta = self.theta[:, np.newaxis]
        x = xs * r * np.sin(theta)
        y = r * np.cos(theta)

        fig, ax = self._fig_ax(fig, ax, aspect=aspect, fopt=opt['fopt'])

        #start plotting
        pcm = plt.pcolormesh(x, y, data_slice, **opt['popt'])
        if lim is not None:
            plt.xlim(0, lim)
            plt.ylim(-lim, lim)
        if both:
            phi = int((phi + self.phic.size // 2) % self.phic.size)
            data_slice = data[phi, :, :]
            plt.pcolormesh(-x, y, data_slice, **opt['popt'])
            if lim is not None:
                plt.xlim(-lim, lim)
                plt.ylim(-lim, lim)
        if rplot:
            rplot = np.atleast_1d(rplot)
            for r in rplot:
                plt.plot(r * np.sin(self.thetac), r * np.cos(self.thetac), lw=1, c='1', ls=':')
        self._labler(ax, pcm, cb=cb, **{k: opt.get(k) for k in ['title', 'cbl', 'cbopt']})

        plt.sca(ax)
        #save fig
        if save or fn:
            tmp = []
            try:
                tmp.append(self.sim.name)
            except AttributeError:
                pass
            tmp += [self._prefix, opt['name'], 'r-theta_plot']
            fn = self._save_fig(fn, '_'.join([i for i in tmp if i]) + '.' + ext, sdir=sdir)
        if ret_fn:
            return fn

        return pcm

    def shell_plot(self, data=None, r=None, fn=None, save=False, title=None, name=None,
                   ext='png', popt=None, cb=True, cbl=None, zerocent=None, vmin=None, vmax=None,
                   cmap=None, cbopt=None, fig=None, fopt=None, ax=None, log=False, aspect=1,
                   sdir=None, ret_fn=False):
        data, opt = self._data_opt_parser(data=data, vmax=vmax, vmin=vmin, zerocent=zerocent, cbl=cbl, name=name)
        if r is None:
            r=1.
        if type(r) == float:
            r = self.rloc(r)
        data_slice = data[:, :, r]
        r = self.rc[r]
        if name is None:
            name = '$R={0:.2f}$'.format(r)
        opt = self._opt_parser(data=data_slice, log=log, fopt=fopt, popt=popt, cbopt=cbopt, cmap=cmap, title=title,
                               **opt)
        if name:
            opt['title'] += ' ' + name
        one = np.ones((self.phi.size, self.theta.size))
        x = self.phi[:, np.newaxis] * one
        y = self.theta[np.newaxis, :] * one

        fig, ax = self._fig_ax(fig, ax, aspect=aspect, fopt=opt['fopt'])

        #start plotting
        pcm = plt.pcolormesh(x, -y, data_slice, **opt['popt'])
        self._labler(ax, pcm, cb=cb, **{k: opt.get(k) for k in ['title', 'cbl', 'cbopt']})

        plt.sca(ax)
        #save fig
        if save or fn:
            fn = self._save_fig(fn, self._prefix + '_r-theta_plot.' + ext, sdir=sdir)
        if ret_fn:
            return fn

        return pcm

    def plot2(self, data=None, phi=None, fn=None, save=False, title=None, name=None, ext='png', popt=None,
              cb=True, cbl=None, zerocent=None, vmin=None, vmax=None, cmap=None, cbopt=None,
              fig=None, fopt=None, log=False, aspect=1, sdir=None, r_cut=None, ret_fn=False):
        data, opt = self._data_opt_parser(data=data, vmax=vmax, vmin=vmin, zerocent=zerocent, cbl=cbl, name=name,
                                          r_cut=r_cut)
        i = np.argmax(self.thetac[self.thetac < .5 * np.pi])
        data_slice = np.vstack([data.mean(axis=0), data[:,i:i+2,:].mean(axis=1)])
        opt = self._opt_parser(data=data_slice, log=log, popt=popt, cbopt=cbopt, cmap=cmap, title=title,
                               **opt)
        if fopt is None:
            fopt = {'figsize': (8,6), 'dpi': 300}
        if fig is None:
            fig = plt.figure(**fopt)
        _gsopt = dict(right=.9, width_ratios=[1,.5, .05], top=.95, left=.05, bottom=.08, wspace=.15, hspace=.25)
        gs = mpl.gridspec.GridSpec(1, 3, **_gsopt)
        ax = plt.subplot(gs[0])
        _opt = {}
        _opt.update(opt)
        _opt['cb'] = False
        a0 = self.r_phi_plot(data, ax=ax, **_opt)


        ax = plt.subplot(gs[1])
        cax = plt.subplot(gs[2])
        _opt = {}
        _opt.update(opt)
        _opt['cbopt']['cax'] = cax
        _opt['name'] = None
        _opt['title'] = False
        a1 = self.r_theta_plot(data, ax=ax, phi=phi, **_opt)
        ax.set_xlim(0, None)

        vmin, vmax = a0.get_clim()
        tmp = a1.get_clim()
        vmin = min(vmin, tmp[0])
        vmax = max(vmax, tmp[1])

        a0.set_clim(vmin, vmax)
        a1.set_clim(vmin, vmax)

        if save or fn:
            fn = self._save_fig(fn, self._prefix + '_map_plot.' + ext, sdir=sdir)

    def zoom_plot(self, data=None, fn=None, title=None, name=True, ext='png', popt=None, save=False, phi_dot=0,
                  cb=True, cbl=None, zerocent=None, vmin=None, vmax='98%', cmap=None, cbopt=None, phi_shift=0,
                  fig=None, fopt=None, log=False, aspect=1, sdir=None, r_cut=None, ret_fn=False, rmax=1.6):
        data, opt = self._data_opt_parser(data=data, vmax=vmax, vmin=vmin, zerocent=zerocent, cbl=cbl, name=name,
                                          r_cut=r_cut)
        i = np.argmax(self.thetac[self.thetac < .5 * np.pi])
        data_slice = data[:,i:i+2,:].mean(axis=1)
        opt = self._opt_parser(data=data_slice, log=log, popt=popt, cbopt=cbopt, cmap=cmap, title=title,
                               **opt)
        if fopt is None:
            fopt = {'figsize': (8,6), 'dpi': 300}
        if fig is None:
            fig = plt.figure(**fopt)
        _gsopt = dict(right=.9, width_ratios=[1,.15, .6, .05], top=.95, left=.1, bottom=.08, wspace=.05, hspace=.25)
        gs = mpl.gridspec.GridSpec(1, 4, **_gsopt)
        ax = plt.subplot(gs[0])
        _opt = {}
        _opt.update(opt)
        _opt['cb'] = False
        a0 = self.r_phi_plot(data, ax=ax, **_opt)
        plt.xlim(-rmax, rmax)
        plt.ylim(-rmax, rmax)
        plt.xlabel('$x$')
        plt.ylabel('$y$')

        ax = plt.subplot(gs[2])
        cax = plt.subplot(gs[3])
        r = self.r[np.newaxis, :]
        phi = self.phi[:, np.newaxis] + phi_shift + self.t * phi_dot
        one = np.ones_like(r * phi)
        x = r * one
        y = phi / np.pi * one
        a1 = ax.pcolormesh(x, y, data_slice, **opt['popt'])
        plt.sca(ax)
        plt.xlabel('$R$')
        plt.ylabel(r'$\phi/\pi$')
        plt.axvline(1., lw=1, ls=':', c='1')
        ax.set_ylim(0, 2)
        ax.set_xlim(self.r[0], rmax)

        _opt = {}
        _opt.update(opt)
        _opt['cbopt']['cax'] = cax
        _opt['name'] = None
        _opt['title'] = False
        self._labler(ax, a1, cb=cb, **{k: _opt.get(k) for k in ['title', 'cbl', 'cbopt']})

        vmin, vmax = a0.get_clim()
        tmp = a1.get_clim()
        vmin = min(vmin, tmp[0])
        vmax = max(vmax, tmp[1])

        a0.set_clim(vmin, vmax)
        a1.set_clim(vmin, vmax)

        if save or fn:
            fn = self._save_fig(fn, self._prefix + '_zoom_plot.' + ext, sdir=sdir)

    def mom1_plot(self, pre, post, data=None):
        if data is None:
            data = self.mom_r(pre, post)
        _data = [i.copy() for i in data]
        lbl = [r'$\partial_t v_r$', r'$(v\cdot\nabla v)_r$', r'$c_s^2\partial_r \rho/\rho$', r'$g_r$']
        if data[0].ndim == 3:
            for i in range(3):
                _data[i] = self.midplane(_data[i]).mean(axis=0)
            _data[3] = _data[3][0,0,:]
        _data[3] = - self.rc**-2
        plt.figure()
        _gsopt = dict(right=.98, height_ratios=[1, .3], top=.95, left=.15, bottom=.08, wspace=.15, hspace=.15)
        gs = mpl.gridspec.GridSpec(2, 1, **_gsopt)
        ax0 = plt.subplot(gs[0])
        for i in range(3):
            plt.plot(self.rc, _data[i], label=lbl[i], zorder=4 - i)
        i = 3
        plt.plot(self.rc, _data[i], label=lbl[i], zorder=5, ls=':')
        plt.legend()
        plt.subplot(gs[1], sharex=ax0)
        plt.plot(self.rc, _data[0] + _data[1] + _data[2] - _data[3], label='Residule')
        return data

    def _mom1_plots(self, pre, post, pseudo=True):
        data = self.mom_r(pre, post)
        data[3] *= np.ones_like(data[0])
        tot = data[1] + data[2] + data[3]
        pwd = os.getcwd()
        if pseudo:
            self.zoom_plot(phi=0, save=1)
            self.r_theta_plot(phi=0, lim=1.2, save=1)
        try:
            path = 'midplane'
            if not os.path.isdir(path):
                os.mkdir(path)
            os.chdir(path)
            self.zoom_plot(data[0], cbl=r'$\partial_t v_r$', fn='figure_1.png')
            self.zoom_plot(data[1], cbl=r'$(v\cdot\nabla v)_r$', fn='figure_2.png')
            self.zoom_plot(data[2], cbl=r'$c_s^2\partial_r \rho/\rho$', fn='figure_3.png')
            self.zoom_plot(data[3], cbl=r'$|g_r|$', fn='figure_4.png')
            self.zoom_plot(tot    , cbl=r'$(v\cdot\nabla v)_r+c_s^2\partial_r \rho/\rho-g_r$', fn='figure_5.png')
            os.chdir(pwd)
            path = 'meridional'
            if not os.path.isdir(path):
                os.mkdir(path)
            os.chdir(path)
            opt = dict(vmax=1, phi=0, lim=1.2,)
            self.r_theta_plot(data[0], cbl=r'$\partial_t v_r$', fn='figure_1.png', **opt)
            self.r_theta_plot(data[1], cbl=r'$(v\cdot\nabla v)_r$', fn='figure_2.png', **opt)
            self.r_theta_plot(data[2], cbl=r'$c_s^2\partial_r \rho/\rho$', fn='figure_3.png', **opt)
            self.r_theta_plot(data[3], cbl=r'$|g_r|$', fn='figure_4.png', **opt)
            self.r_theta_plot(tot, cbl=r'$(v\cdot\nabla v)_r+c_s^2\partial_r \rho/\rho-g_r$', fn='figure_5.png', **opt)
        finally:
            os.chdir(pwd)

    def damp_vel(self, s=5.):
        na = np.newaxis
        r = self.rc[na, :]
        th = self.thetac[:, na]
        y = r * np.cos(th)
        x = r * np.sin(th)
        v1 = self['vel1'].mean(axis=0) * _window(y, (.6 * x)**1.7, s) * (np.tanh(4 * s * (r - .95)) + 1) * .5
        v2 = self['vel2'].mean(axis=0) * _window(y, (.6 * x)**1.9, s) * (np.tanh(4 * s * (r - .95)) + 1) * .5
        v3 = self['vel3'].mean(axis=0) * (1. - _window(y, (.75 * x)**1.5, s))
        return v3, v2, v1

    def write_IC(self, fn="blic.data", damp=True, sym=True):
        out = [self['dens'].mean(axis=0)]
        if damp:
            out += list(self.damp_vel())[::-1]
        else:
            out += [self['vel' + str(i + 1)].mean(axis=0) for i in range(3)]
        if sym:
            for i in range(len(out)):
                if i == 2:
                    out[i] = .5 * out[i] - .5 * out[i][::-1]
                else:
                    out[i] = .5 * out[i] + .5 * out[i][::-1]
        one = np.ones_like(out[0])
        out.append(self.rc[np.newaxis, :] * one)
        out.append(self.thetac[:, np.newaxis] * one)
        out = np.array(out)
        dR = np.diff(self.r)
        dR0 = dR[0]
        Rrat = np.mean(dR[1:] / dR[:-1])
        print(out.shape)
        with open(fn, 'wb') as f:
            np.array(out.shape, 'int32').tofile(f)
            np.array([Rrat, dR0], 'float64').tofile(f)
            out.tofile(f)
        return out

def _window(x, x0, s=5.):
    return (np.tanh(s * (x - x0)) * np.tanh(s * (x + x0)) + 1.) * .5

class BL3dSim(object):
    def __init__(self, path, athinput=None, x2_face=None, fmts=None, defvar='Rpseudo'):
        if fmts is None:
            fmts = blc._file_fmts
        self.name = os.path.split(os.path.abspath(path))[-1]
        path = os.path.expanduser(path)
        if path == self.name and not os.path.isdir(path):
            dirs = blc._dirs[:]
            dirs += [os.path.join(d, '3d') for d in dirs]
            for d in dirs:
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
            self.athinput = athinput
        elif os.path.isfile(os.path.join(self.path, athinput)):
            athinput = os.path.join(self.path, athinput)
            self.inputs = ar.athinput(athinput)
            self.athinput = athinput
        else:
            self.inputs = {}
        self.fileDict = {}
        self.varDict = {}
        self.idDict = {}
        self.defvar = defvar
        axes = []
        if not self.inputs:
            searches = [fmt.split('%')[0] + '*.' + fmt.split('d.')[-1] for fmt in fmts]
            raise NotImplementedError('Currently needs athinput.')
        else:
            try:
                self.mach = 1. / self.inputs['hydro']['iso_sound_speed']
            except KeyError:
                self.mach = 1. / self.inputs['hydro']['invMach']
            mesh = self.inputs['mesh']
            for i in [1, 2, 3]:
                x = 'x' + str(i)
                nx = mesh['n' + x]
                Dx = mesh[x + 'max'] - mesh[x + 'min']
                if x + 'rat' in mesh:
                    rat = mesh[x + 'rat']
                    if x == 'x2' and rat == -1:
                        if x2_face is None:
                            x2_face = helpers.x2_face
                        xf = x2_face(mesh[x + 'min'], mesh[x + 'max'], -1, nx + 1)
                    elif rat != 1:
                        dx0 = (rat - 1.) / (rat**nx - 1.) * Dx
                        xf = np.ones(nx + 1) * mesh[x + 'min']
                        xf[1:] += (rat**np.arange(nx) * dx0).cumsum()
                    else:
                       xf = (np.arange(nx + 1) * Dx / nx) + mesh[x + 'min']
                else:
                    xf = (np.arange(nx + 1) * Dx / nx) + mesh[x + 'min']
                axes.append(xf)
            self.x2_face = x2_face
            self.r = axes[0].copy()
            self.dr = self.r[1:] - self.r[:-1]
            self.rc = .5 * self.r[1:] + .5 * self.r[:-1]
            try:
                self.theta = axes[1].copy()
            except AttributeError:
                self.theta
            self.thetac = .5 * self.theta[1:] + .5 * self.theta[:-1]
            self.dtheta = .5 * self.theta[1:] + .5 * self.theta[:-1]
            self.phi = axes[2].copy()
            self.phic = .5 * self.phi[1:] + .5 * self.phi[:-1]
            outs = [i for i in self.inputs.keys() if i[:6] == 'output']
            a = self.inputs['job']['problem_id']
            c = '[0-9]*'
            varlist = list(filter(None, [self.inputs[i].get('variable') for i in outs]))
            self._coarse_data = None
            self._fine_data = None
            for out in outs:
                files = []
                b = self.inputs[out].get('id', 'out' + out[6:])
                searches = ['.'.join([a, b, c, ext]) for ext in blc._ext]
                for search in searches:
                    files += [os.path.split(i)[-1] for i in glob(os.path.join(path, search))]
                self.fileDict[out] = sorted(files)
                var = self.inputs[out].get('variable')
                if varlist.count(var) == 1:
                    self.varDict[var] = out
                self.idDict[b] = out

    def __repr__(self):
        return '<BL3dSim "{0:}">'.format(self.name)

    def run(self, athinput=None, args=None, rundir=None):
        if athinput is None:
            athinput = self.athinput
        if not os.path.exists(athinput):
            athinput = os.path.split(athinput)[1]
        print(athinput)
        if args is None:
            args = []
        if rundir is None:
            rundir = self.path
        pwd = os.getcwd()
        try:
            os.chdir(rundir)
            if not os.path.exists(athinput):
                athinput = os.path.split(athinput)[1]
            run_command = ['./athena', '-i', athinput]
            try:
                subprocess.check_call(run_command + args)
            except subprocess.CalledProcessError as err:
                raise RuntimeError('Return code {0} from command \'{1}\''
                                  .format(err.returncode, ' '.join(err.cmd)))
        finally:
            os.chdir(pwd)

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

    def loadfile(self, fn, index=None, data=None, num_ghost=0):
        if not index is None:
            files = self.files(fn)
            if index < 0 or files[-1].split('.')[2] == len(files):
                fn = files[index]
            else:
                fn = list(files[0].split('.'))
                fn[2] = '%5.5d' % index
                fn = '.'.join(fn)
        if fn in self.filenames:
            return BLfile(os.path.join(self.path, fn), sim_path=os.path.abspath(self.path), sim=self,
                          ai_data=self.inputs, data=data, num_ghost=num_ghost)
        raise ValueError('Unknown file.')

    def mk_maps(self, files=None, key='out1', data=None, skip_existing=True, popt=None):
        if popt is None:
            popt = {}
        if files is None:
            files = self.files(key)
        for f in files:
            print(f)
            mk_plt = True
            try:
                f.fn
            except AttributeError:
                f = self.loadfile(f)
            try:
                if skip_existing:
                    tmp = '.'.join(os.path.split(f.fn)[1].split('.')[:-1]) + '_map_plot.png'
                    if os.path.exists(tmp):
                        mk_plt = False
                if mk_plt:
                    f.plot2(data=data, save=True, **popt)
                else:
                    print(tmp + ' Exists. Skipping ' + f.fn)
            except KeyboardInterrupt:
                raise
            except:
                print(f, "Failed")

    def mk_zooms(self, files=None, key='out1', data=None, skip_existing=True, popt=None):
        if popt is None:
            popt = {}
        if files is None:
            files = self.files(key)
        for f in files:
            print(f)
            mk_plt = True
            try:
                f.fn
            except AttributeError:
                f = self.loadfile(f)
            try:
                if skip_existing:
                    tmp = '.'.join(os.path.split(f.fn)[1].split('.')[:-1]) + '_zoom_plot.png'
                    if os.path.exists(tmp):
                        mk_plt = False
                if mk_plt:
                    f.zoom_plot(data=data, save=True, **popt)
                else:
                    print(tmp + ' Exists. Skipping ' + f.fn)
            except KeyboardInterrupt:
                raise
            except:
                print(f, "Failed")

    def wp(self, key='out1', quiet=False, save=True, overwrite=False):
        fn = os.path.join(self.path, "wave_power.csv")
        if not overwrite:
            if os.path.isfile(fn):
                try:
                    return np.loadtxt(fn, delimiter=",").T
                except KeyboardInterrupt:
                    raise
                except:
                    pass
        out = []
        for i in self.files(key):
            bf = self.loadfile(i)
            out.append((bf.t / tau, bf.wave_power_2()))
            if not quiet: print(out[-1])
        out = np.array(out).T
        if save:
            np.savetxt(fn, out.T, delimiter=",")
        return out

    def swp(self, key='out1', save=True, overwrite=False):
        fn = os.path.join(self.path, "swp.csv")
        if not overwrite:
            if os.path.isfile(fn):
                try:
                    return np.loadtxt(fn, delimiter=",").T
                except KeyboardInterrupt:
                    raise
                except:
                    pass
        out = []
        for i in self.files(key):
            bf = self.loadfile(i)
            out.append((bf.t / tau, bf.swp()))
        out = np.array(out).T
        if save:
            np.savetxt(fn, out.T, delimiter=",")
        return out

    def test_plots(self, overwrite=False, movie=False, ll=False):
        name = helpers.sanitize_lbl(os.path.abspath(self.path).split('1d_tests/')[-1])

        if movie:
            mdir = os.path.join(os.path.abspath(self.path), 'movie')
            def f(fn):
                bf = self.loadfile(fn)
                bf.plot_rprof(ext='png', save=True, sdir=mdir, zerocent=True,
                              xlim=[None, 1.2], name=name + r'$t/2\pi = {orbit:g}$')
            if ll:
                parmap(f, self.files('out1'))
            else:
                [f(i) for i in self.files('out1')]


        out = self.swp(overwrite=overwrite)
        plt.semilogy(*out)
        plt.xlabel('$r$')
        plt.ylabel(r'$\int v_r^2\rho dr$')
        plt.title(name)
        plt.savefig(os.path.join(self.path, 'power_time.pdf'))
        plt.close()

        bf = self.loadfile('out1', -1)
        plt.plot(bf.rc, bf['pseudo'].mean(axis=(0,1)))
        plt.xlabel('$r$')
        plt.ylabel(r'$v_r\sqrt{\rho}$')
        plt.title(name)
        plt.savefig(os.path.join(self.path, 'pseudo.pdf'))
        plt.close()

        return out

    def parse_func(self, func, *args, **kwargs):
        return getattr(self, func)(*args, **kwargs)


def comp_wrapper(func, simlist=None, include=True, tmin=30, T=False, args=None, kwargs=None, sim_class=BL3dSim):
    return blc.comp_wrapper(func, simlist=simlist, include=include, tmin=tmin, T=T, args=args, kwargs=kwargs, sim_class=sim_class)
