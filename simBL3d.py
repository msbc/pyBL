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
                 file_handle=None, x2_face=None):
        if x2_face is None and sim is not None:
            x2_face = sim.x2_face
        super(BLfile, self).__init__(fn, sim_path=sim_path, t=t, data=data, defvar=defvar, ai_data=ai_data, sim=sim,
                                     file_handle=file_handle, x2_face=x2_face, trim=False)
        self.r = self['x1f']
        self.dr = self.r[1:] - self.r[:-1]
        self.rc = .5 * self.r[1:] + .5 * self.r[:-1]
        self.theta = self['x2f']
        self.thetac = .5 * self.theta[1:] + .5 * self.theta[:-1]
        self.dtheta = .5 * self.theta[1:] + .5 * self.theta[:-1]
        self.phi = self['x3f']
        self.phic = .5 * self.phi[1:] + .5 * self.phi[:-1]
        self._grid_shape = self.phic.size, self.thetac.size, self.rc.size
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
        return None

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

    def plot_slice(self, sim_slice, data=None, fn=None, save=False, subsample=False, title=None,
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
                     sdir=None, r_cut=None, ret_fn=False, rplot=1, both=False, xs=1):
        if xs is None:
            xs = 1
        if abs(xs) != 1:
            raise ValueError('|xs| must be 1.')
        data, opt = self._data_opt_parser(data=data, vmax=vmax, vmin=vmin, zerocent=zerocent, cbl=cbl, name=name,
                                          r_cut=r_cut)
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
        opt = self._opt_parser(data=data_slice, log=log, fopt=fopt, popt=popt, cbopt=cbopt, cmap=cmap, title=title,
                               **opt)

        r = self.r[np.newaxis, :]
        theta = self.theta[:, np.newaxis]
        x = xs * r * np.sin(theta)
        y = r * np.cos(theta)

        fig, ax = self._fig_ax(fig, ax, aspect=aspect, fopt=opt['fopt'])

        #start plotting
        pcm = plt.pcolormesh(x, y, data_slice, **opt['popt'])
        if both:
            phi = int((phi + self.phic.size // 2) % self.phic.size)
            data_slice = data[phi, :, :]
            plt.pcolormesh(-x, y, data_slice, **opt['popt'])
        if rplot:
            rplot = np.atleast_1d(rplot)
            for r in rplot:
                plt.plot(r * np.cos(self.phic), r * np.sin(self.phic), lw=1, c='1', ls=':')
        self._labler(ax, pcm, cb=cb, **{k: opt.get(k) for k in ['title', 'cbl', 'cbopt']})

        plt.sca(ax)
        #save fig
        if save or fn:
            fn = self._save_fig(fn, self._prefix + '_r-theta_plot.' + ext, sdir=sdir)
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
            r = self.theta_loc(r)
        data_slice = data[:, r, :]
        r = self.rc[r]
        opt = self._opt_parser(data=data_slice, log=log, fopt=fopt, popt=popt, cbopt=cbopt, cmap=cmap, title=title,
                               **opt)

        one = np.ones((self.phi.size, self.theta.size))
        x = self.phi[:, np.newaxis] * one
        y = self.theta[np.newaxis, :] * one

        fig, ax = self._fig_ax(fig, ax, aspect=aspect, fopt=opt['fopt'])

        #start plotting
        pcm = plt.pcolormesh(x, y, data_slice, **opt['popt'])
        self._labler(ax, pcm, cb=cb, **{k: opt.get(k) for k in ['title', 'cbl', 'cbopt']})

        plt.sca(ax)
        #save fig
        if save or fn:
            fn = self._save_fig(fn, self._prefix + '_r-theta_plot.' + ext, sdir=sdir)
        if ret_fn:
            return fn

        return pcm

    def plot2(self, data=None, fn=None, save=False, title=None, name=None, ext='png', popt=None,
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
        a1 = self.r_theta_plot(data, ax=ax, **_opt)
        ax.set_xlim(0, None)

        vmin, vmax = a0.get_clim()
        tmp = a1.get_clim()
        vmin = min(vmin, tmp[0])
        vmax = max(vmax, tmp[1])

        a0.set_clim(vmin, vmax)
        a1.set_clim(vmin, vmax)

        if save or fn:
            fn = self._save_fig(fn, self._prefix + '_map_plot.' + ext, sdir=sdir)


class BL3dSim(object):
    def __init__(self, path, athinput=None, x2_face=None, fmts=None):
        if fmts is None:
            fmts = blc._file_fmts
        self.name = os.path.split(os.path.abspath(path))[-1]
        path = os.path.expanduser(path)
        if path == self.name and not os.path.isdir(path):
            for d in blc._dirs:
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
                    else:
                        dx0 = (rat - 1.) / (rat**nx - 1.) * Dx
                        xf = np.ones(nx + 1) * mesh[x + 'min']
                        xf[1:] += (rat**np.arange(nx) * dx0).cumsum()
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

    def loadfile(self, fn, index=None):
        if not index is None:
            fn = self.files(fn)[index]
        if fn in self.filenames:
            return BLfile(os.path.join(self.path, fn), sim_path=os.path.abspath(self.path), sim=self, ai_data=self.inputs)
