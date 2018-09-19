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
hpi = .5 * np.pi
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

    def axis_loc(self, axis, value):
        return np.argmin(np.abs([self.phic, self.thetac, self.rc][axis] - value))

    def theta_loc(self, theta):
        return self.axis_loc(1, theta)

    def phi_loc(self, phi):
        return self.axis_loc(1, phi)

    def plot_slice(self, slice, data=None, fn=None, save=False, subsample=False, title=None,
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
        cart_grid = self.z, self.y, self.x
        sim_grid = self.phic, self.thetac, self.rc
        for i, s in enumerate(slice):
            if type(s) == float:
                s = self.axis_loc(i, s)

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
