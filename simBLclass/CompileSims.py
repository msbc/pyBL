#! /usr/bin/env python

# from __future__ import absolute_import, division, print_function
# from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
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
from ._local_helpers import *
from .defaults import rc
from .simBLclass import BLsim

def parallel_compile(func, arglist, T=None):
    """Usage : parallel_compile(func, arglist=None, T=None)
    Similar to comp_wrapper, but strings in arglist are not automatically turned
    into BLsim class instances."""
    out = [i for i in parmap(func, arglist) if i is not None]
    if T: out = zip(*out)
    return out


def comp_wrapper(func, simlist=None, include=None, tmin=200, T=False, args=None,
                 kwargs=None, sim_class=BLsim, ll=True):
    """Usage : comp_wrapper(func, simlist=None, include=None, tmin=40, T=False, load_eos=False)
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
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if not simlist:
        dirs = [i for i in rc('dirs')[::-1] if os.path.isdir(i)]
        while True:
            tmp = glob(os.path.join(dirs[0], '*/athinput.*'))
            if tmp:
                break
            dirs.pop(0)
        simlist = sorted(set(os.path.split(i)[0] for i in tmp))

    sims = []

    # if not simlist: simlist = simlist4

    if tmin == None: tmin = 200

    if include == None:
        def include(sim):
            return sim.fft_time[-1] >= tmin
    elif include == True:
        def include(sim):
            return True

    def mapper(sim):
        output = None
        # if sim is a string
        try:
            sim.rstrip()
            name = sim
            sim = sim_class(sim)
        # otherwise assume it's a simulation
        except AttributeError:
            name = sim.name
        # if we get here there's no hope
        except KeyboardInterrupt:
            raise
        except:
            print('Bad sim (no load): ' + name)
            print(traceback.format_exc())
            return None

        try:
            if include(sim):
                if not callable(func):
                    output = sim.parse_func(func, *args, **kwargs)
                if output is None:
                    return func(sim)
        except KeyboardInterrupt:
            raise
        except:
            print('Bad sim (func err): ' + name)
            print(traceback.format_exc())
        gc.collect()
        return output

    if ll:
        out = parallel_compile(mapper, arglist=simlist, T=T)
    else:
        out = list(map(mapper, simlist))
        if T:
            out = zip(*out)
    return out


def mkplots(simlist=None, maps=False, overwrite=True, fluxes=True):
    opt = dict(quiet=True, working_dir=True, maps=maps, overwrite=overwrite,
               fluxes=fluxes)
    comp_wrapper('main_plots', simlist=simlist, kwargs=opt)


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
            # opt['sdir'] = rdir
            os.chdir(rdir)
            for r in [.95, 1, 1.1, 1.5, 2, 3]:
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


def mk_sim_tbl(sep=' & ', end_line=r'\\'):
    for mach in range(5, 16):
        for res in ['LR', 'FR', 'HR']:
            qry = 'M{0:02d}.{1:}.*'.format(mach, res)
            sims = sorted(glob(qry))
            for sim in sims:
                print(sep.join(BLsim(sim).info_row()) + end_line)
        print(r'\hline')
    return None


def grab_bl(sims=None):
    return np.array(comp_wrapper('mach_and_bl', sims))


def bl_size_plot(sims=None, data=None, cmap=None):
    if data is None:
        data = grab_bl(sims)
    if cmap is None:
        cmap = plt.get_cmap()
    machs = (data[:, 0] + .1).astype(int)
    colors = machs - machs.min()
    colors = cmap(colors / colors.max())
    # shapes = ['o', 'v', '+', '^', 's']
    shapes = ['.']
    shapes = [shapes[i % len(shapes)] for i in machs]
    handles = dict()
    for i, d in enumerate(data):
        handles[machs[i]] = plt.scatter(d[0], d[-1], color=colors[i], marker=shapes[i])
    lbls = [r'$\mathcal{M}=' + str(i) + '$' for i in handles]
    hands = [handles[i] for i in handles]
    plt.legend(hands, lbls)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.01))
    plt.ylabel('Boundary Layer Size')
    plt.xlabel('Mode Number')


def sims_within(path=None, nmin=50, nmax=20000):
    if path is None:
        path = '~/data/pleiades_data/bl'
    path = os.path.expanduser(path)
    out = []
    for x in os.walk(path):
        if glob(os.path.join(x[0], 'athinput.*')):
            if nmin <= len(glob(os.path.join(x[0], '*.athdf'))) <= nmax:
                out.append(x[0])
    return out


def _mk_cs_eff_data(sims=None, sims_path=None, **kwargs):
    if sims is None:
        sims = sims_within(sims_path)
    elif os.path.isfile(sims):
        with open(sims) as f:
            sims = [l.split('#')[0].strip() for l in f.readlines()]
        sims = [s for s in sims if s]
    data = dict()
    tmp = comp_wrapper('ratio_Mdot', simlist=sims, **kwargs)
    for i in tmp:
        try:
            data[i[0]] = np.hstack([data[i[0]], i[1]])
        except KeyError:
            data[i[0]] = i[1]
    return data


def mk_cs_datafile(fn='cs_eff.npz', **kwargs):
    data = _mk_cs_eff_data(**kwargs)
    np.savez(fn, **{str(int(i + .5)): data[i] for i in data})
    return data


def CS_eff_plot(sims=None, sims_path=None, data=None, dpi=300, figsize=None, **kwargs):
    if data is None:
        data = _mk_cs_eff_data(sims=sims, sims_path=sims_path, **kwargs)
    elif hasattr(data, 'lower'):
        fn = os.path.expanduser(data)
        if os.path.isfile(fn):
            data = np.load(fn)
    keys = [float(i) for i in data]
    vmin, vmax = int(np.min(keys) + .5), int(np.max(keys) + .5)
    norm = mpl.colors.Normalize(vmin=vmin - .5, vmax=vmax + .5)
    cmap = plt.get_cmap(None, int(vmax - vmin + 1.5))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    for m in data:
        c = cmap(norm(float(m)))
        plt.loglog(*data[m], marker='+', linewidth=0, mfc=c, mec=c)
    plt.xlabel(r'$|\dot{M}|$')
    plt.ylabel(r'$C_{\rm S}/\dot{M}$')

    # colorbar
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, orientation='vertical', norm=norm,
                                   ticks=np.arange(vmin, vmax + 1))
    cb.set_label(r'$\mathcal{M}$')

def CS_eff_bin_plot(sims=None, sims_path=None, data=None, dpi=300, figsize=None, fig=True,
                    md_min=1e-4, log_width=1./3., xp=.5, use_mean=True, cb=True, cax=True,
                    ylbl=True, ylim=None, **kwargs):
    if data is None:
        data = _mk_cs_eff_data(sims=sims, sims_path=sims_path, **kwargs)
    elif hasattr(data, 'lower'):
        fn = os.path.expanduser(data)
        if os.path.isfile(fn):
            data = np.load(fn)
        else:
            data = mk_cs_datafile(fn)
    keys = [float(i) for i in data]
    vmin, vmax = int(np.min(keys) + .5), int(np.max(keys) + .5)
    norm = mpl.colors.Normalize(vmin=vmin - .5, vmax=vmax + .5)
    cmap = plt.get_cmap(None, int(vmax - vmin + 1.5))

    xmax = max([i[0].max() for i in data.values()])
    width = 10.0**log_width
    edges = 10**np.arange(np.log10(md_min), np.log10(xmax) + log_width, log_width)
    mids = np.sqrt(edges[:-1] * edges[1:])
    nbins = edges.size - 1

    def stats(xy):
        x, y = xy
        count = np.zeros(nbins)
        yval = np.zeros(nbins)
        if use_mean:
            err = np.zeros(nbins)
        else:
            err = np.zeros((2, nbins))
        for i in range(nbins):
            tmp = y[np.logical_and(edges[i] <= x, x < edges[i + 1])]
            count[i] = tmp.size
            if use_mean:
                yval[i] = tmp.mean()
                err[i] = tmp.std()
            else:
                yval[i] = np.median(tmp)
                err[0, i] = helpers.percentile(tmp, 25)
                err[1, i] = helpers.percentile(tmp, 75)
                err[:, i] = np.abs(err[:, i] - yval[i])
        return count, yval, err


    ylist = []
    if fig is True:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    for m in data:
        c = cmap(norm(float(m)))
        ylist.extend(data[m][1][data[m][0] >= md_min])
        n, y, std = stats(data[m])
        x = mids * (float(m) / 9)**xp
        plt.errorbar(x, y, yerr=std, ecolor=c, mfc=c, mec=c, fmt='.', elinewidth=1,
                     capsize=1)
    plt.xscale('log')
    plt.yscale('log')
    xl = plt.xlim(edges[0], edges[-1])
    for e in edges[1:-1]:
        plt.axvline(e, c='.8', lw=1, ls=':', zorder=-2)
    if use_mean:
        yval = np.mean(ylist)
        err = np.std(ylist)
        a, b = yval - err, yval + err
    else:
        yval = np.median(ylist)
        a, b = [helpers.percentile(ylist, i) for i in [25, 75]]
    print(a - yval, yval, b - yval)
    plt.axhline(yval, c='k', lw=1, ls=':', zorder=-1)
    plt.fill_between(xl, a, b, zorder=-3, facecolor='.9')
    plt.xlim(*xl)
    plt.xlabel(r'$|\dot{M}|$')
    if ylbl:
        plt.ylabel(r'$C_{\rm S}/\dot{M}$')
    yl = list(plt.ylim())
    yl[0] = min(yl[0], 1e-1)
    plt.ylim(*yl)
    if np.any(ylim):
        yl = plt.ylim(*ylim)

    # colorbar
    if cb:
        if cax is True:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, orientation='vertical', norm=norm,
                                       ticks=np.arange(vmin, vmax + 1))
        cb.set_label(r'$\mathcal{M}$')

    return data, yl


def CS_both(dpi=300, figsize=None, save=False, dropbox=False, **kwargs):
    kwargs['fig'] = False
    if 'data' not in kwargs:
        kwargs['data'] = '~/data/pleiades_data/bl/cs_eff.npz'
    if 'ylim' not in kwargs:
        kwargs['ylim'] = [1e-1, None]
    if figsize is None:
        figsize = (5, 3)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = mpl.gridspec.GridSpec(1, 3, width_ratios=[1, 1, .05], top=.99, bottom=.14,
                               left=.13, right=.92, wspace=0)
    #axs = [plt.subplot(gs[i]) for i in [0, 2, 3]]
    axs = [plt.subplot(i) for i in gs]
    plt.sca(axs[0])
    kwargs['data'], kwargs['ylim'] = CS_eff_bin_plot(use_mean=True, cb=False, **kwargs)
    plt.sca(axs[1])
    CS_eff_bin_plot(use_mean=False, cax=axs[2], ylbl=False, **kwargs)
    axs[0].yaxis.set_ticks_position('both')
    axs[0].tick_params(axis='both', which='both', direction='in')
    #axs[1].yaxis.set_ticks_position('both')
    axs[1].tick_params(axis='both', which='both', direction='in')
    axs[1].set_yticklabels([])
    x, y = 2e-4, 3
    opt = dict(bbox=dict(edgecolor='none', facecolor='white', alpha=.7), fontsize=6)
    axs[0].text(x, y, "a) mean $\pm$ standard-deviation", **opt)
    axs[1].text(x, y, "b) median $\pm$ quartile", **opt)

    if save:
        fn = 'CS_Mdot_plot.pdf'
        if dropbox:
            fn = 'Dropbox/' + fn
        fig.savefig(fn)
        plt.close()

def gen_dispersion_data(sims=None, fn=None, overwrite=False, skip_gen=False):
    if fn is None:
        fn = 'dispersion_data.hdf5'
    if skip_gen and os.path.isfile(fn):
        return fn
    if sims is None:
        sims = ['M05.FR.r.a', 'M06.HR.r.lc.a', 'M07.FR.r.a', 'M08.FR.r.a',
                'M09.FR.r.lc.a',
                'M09.HR.r.a', 'M10.FR.r.a', 'M11.FR.r.a', 'M12.FR.r.lc.a', 'M13.FR.r.a',
                'M14.FR.r.a', 'M15.FR.r.a']
    with h5py.File(fn, 'a') as f:
        for s in sims:
            sim = BLsim(s)
            if not overwrite:
                if sim.name in f and sim.name + '/mach' in f:
                    print('~~~ Skipping {:}, data already exists.'.format(s))
                    continue
            print('==> Generating dispersion data for ' + s)
            data = sim.mode_detect().plot_data()
            if not sim.name in f:
                grp = f.create_group(sim.name)
            else:
                grp = f[sim.name]
            for i in data:
                if i not in grp:
                    grp.create_dataset(i, data=data[i])
                else:
                    grp[i][:] = data[i]
            del(data, sim)
            gc.collect()
    print('Finished data generation')
    return fn

def plot_dispersion_data(data=None, save=False, figsize=None, dpi=300, skip_gen=False,
                         fopt=None, lopt=None, fn=None, sdir=None, overwrite=True):
    if save or fn:
        save = True
        if not fn:
            fn = 'multi_dispersion.pdf'
    if parse_not_overwrite(overwrite, fn):
        return None
    if data is None:
        data = gen_dispersion_data(skip_gen=skip_gen)
    file_used = False
    if hasattr(data, "lower"):
        data = h5py.File(data, 'r')
        file_used = True
    bbox = dict(edgecolor='none', facecolor='white', pad=0.3)

    def upper_mode(param, r=None, omega=None, use_kep=False, calc_kappa=False):
        if r is None:
            ir = param['omega'][()].argmax()
            r = param['rc'][ir]
        else:
            ir = np.abs(r - param['rc'][()]).argmin()
        # kappa = np.sqrt(2 * omega * (grad(self.rc, self.rc * vphi)))
        if use_kep:
            omega = r ** -1.5
        if omega is None:
            omega = param['omega'][ir]
        op = np.linspace(0, 0.95 * omega, 400)
        kappa = 2 * omega
        if calc_kappa:
            _r = param['rc'][()]
            _o = param['omega']
            kappa = np.sqrt(2 * _o[ir] * grad(_r, _r**2 * _o)[ir])
        if use_kep:
            kappa = omega
        s = 1. / param['mach'][()]
        do = omega - op
        m2 = (do * kappa ** 2 * r ** 2 - 2 * s ** 2 * omega) / (
                do ** 3 * r ** 2 - s ** 2 * do)
        return np.sqrt(m2), op

    def lower_mode(param):
        mach = int(np.round(param['mach']))
        x = np.linspace(0, 32, 321)
        y = np.sqrt(mach ** -2 + (mach / (2 * rc('rl')[mach] * x)) ** 2) / rc('rl')[mach]
        return x, y

    try:
        if lopt is None:
            lopt = dict(loc='upper left')
        golden = (1 + 5 ** 0.5) / 2
        sims = sorted(list(data.keys()))
        nsim = len(sims)
        nc = int(np.round(np.sqrt(nsim / golden)))
        nr = int(np.round(np.sqrt(nsim * golden)))
        if figsize is None:
            figsize = (7.5, 9.5)
        _fopt = dict(dpi=dpi, figsize=figsize)
        if fopt is None:
            fopt = {}
        _fopt.update(fopt)
        fig = plt.figure(**_fopt)
        gs = mpl.gridspec.GridSpec(nr, nc, top=.99, bottom=.04, left=.07, right=.99,
                                   wspace=0, hspace=0)
        axs = [[None] * nc] * nr
        cap = 1
        xmin = np.ones(nc) * 32
        xmax = np.zeros(nc)
        for r in range(nr):
            ymax = -1
            ymin = 1
            for c in range(nc):
                i = c + r * nc
                if i < nsim:
                    opt = dict()
                    if c > 0:
                        opt['sharey'] = axs[r][0]
                    if r > 0:
                        opt['sharex'] = axs[0][c]
                    ax = plt.subplot(gs[r, c], **opt)
                    axs[r][c] = ax
                    ### Plot modes
                    markers = 'o', '+', 'x', '.'
                    info = data[sims[i]]
                    ax.scatter(*info['r0_modes'], marker=markers[0])
                    ax.scatter(*info['r1_modes'], marker=markers[1])
                    ax.scatter(*info['global'], marker='*')
                    if i == 1:
                        plt.legend(['Star', 'Disk', 'Global'], **lopt)
                    xlist = []
                    ylist = []
                    for key in ['r0_modes', 'r1_modes', 'global']:
                        xlist.extend(list(info[key][0]))
                        ylist.extend(list(info[key][1]))
                    xmax[c] = max(xmax[c], np.max(xlist) + 1.0)
                    xmin[c] = min(xmin[c], np.min(xlist) - 1.0)
                    xmax[c] = min(32, xmax[c])
                    xmin[c] = max(0, xmin[c])
                    ymax = max(ymax, np.max(ylist) + .05)
                    ymin = min(ymin, np.min(ylist) - .05)
                    ymax = min(.99, ymax)
                    ymin = max(0, ymin)
                    # upper mode
                    _x, _y = upper_mode(info, calc_kappa=True)#info['mach'][()] > 13.5)
                    plt.plot(_x, _y, c='xkcd:crimson', ls=':', lw=1, zorder=-1)
                    plt.plot(2 * _x, _y, c='.6', ls=':', lw=1, zorder=-1)
                    plt.plot(3 * _x, _y, c='.7', ls=':', lw=1, zorder=-1)
                    # if info['mach'][()] > 0:#13.5:
                    #     _x, _y = upper_mode(info, use_kep=True)
                    #     plt.plot(_x, _y, c='xkcd:crimson', ls='-', lw=1, zorder=-1)
                    # if int(np.round(info['mach'])) == 150:
                    #     _x, _y = upper_mode(info, omega=0.8891435977119754, r=1.0500903710175424)
                    #     plt.plot(_x, _y, c='xkcd:crimson', ls='-', lw=1, zorder=-1)
                    # elif int(np.round(info['mach'])) == 140:
                    #     _x, _y = upper_mode(info, omega=0.8854813712411813, r=1.0456745274364727)
                    #     plt.plot(_x, _y, c='xkcd:crimson', ls='-', lw=1, zorder=-1)
                    # lower mode
                    _x, _y = lower_mode(info)
                    plt.plot(_x, _y, c='xkcd:crimson', ls='-.', lw=1, zorder=-1)
                    plt.plot(2 * _x, _y, c='.6', ls='-.', lw=1, zorder=-1)
                    plt.plot(3 * _x, _y, c='.7', ls='-.', lw=1, zorder=-1)
                    plt.ylim(ymin, ymax)
                    plt.xlim(xmin[c], xmax[c])
                    # omega_max
                    plt.axhline(np.max(info['omega']), c='.5', lw=1, ls='--', zorder=-2)
                    ### End of mode plotting
                    ax.xaxis.set_ticks_position('both')
                    ax.yaxis.set_ticks_position('both')
                    ax.tick_params(axis='both', which='both', direction='in', zorder=10)
                    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.1))
                    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
                    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
                    lbl = chr(ord('a') + i) + ') ' + sims[i]
                    #if sim.name == 'M09.HR.r.a':
                    #    plt.ylim(.1, None)
                    ax.text(.96, .94, lbl, c='k', transform=ax.transAxes, ha='right',
                            fontsize=6, bbox=bbox)
                    if c:
                        plt.setp(ax.get_yticklabels(), visible=False)
                    else:
                        plt.ylabel(r'$\Omega_p$')
                    if r == nr -1:
                        plt.xlabel(r'$m$')
                    else:
                        plt.setp(ax.get_xticklabels(), visible=False)
                    gc.collect()
            cap = .95
        if save:
            print('Saving Figure')
            plt.savefig(fn)
            plt.close()
    finally:
        if file_used:
            data.close()
    return

def multi_dispersion(sims=None, data_dir=None, save=False, figsize=None, dpi=300,
                     fopt=None, lopt=None, fn=None, sdir=None, overwrite=True):
    if save or fn:
        save = True
        if not fn:
            fn = 'multi_dispersion.pdf'
        if sdir is True:
            sdir = data_dir if data_dir else ''
            sdir = os.path.join(os.path.split(sdir, 'figs'))
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if parse_not_overwrite(overwrite, fn):
        return None
    if sims is None:
        if data_dir is None:
            data_dir = rc('dirs')[-1]
        short = 'M{:02d}.{:}R.r.a'
        full = os.path.join(data_dir, short)
        sims = [short.format(m, 'H') if os.path.isdir(full.format(m, 'H'))
                else short.format(m, 'F') for m in range(5, 16)]
        sims.insert(sims.index('M09.HR.r.a'), 'M09.FR.r.a')
        sims = ['M05.FR.r.a', 'M06.HR.r.lc.a', 'M07.FR.r.a', 'M08.FR.r.a', 'M09.FR.r.lc.a',
                'M09.HR.r.a', 'M10.FR.r.a', 'M11.FR.r.a', 'M12.FR.r.lc.a', 'M13.FR.r.a',
                'M14.FR.r.a', 'M15.FR.r.a']
        if lopt is None:
            lopt = dict(loc='upper left')
    if lopt is None:
        lopt = dict()
    golden = (1 + 5 ** 0.5) / 2
    nsim = len(sims)
    nc = int(np.round(np.sqrt(nsim / golden)))
    nr = int(np.round(np.sqrt(nsim * golden)))
    if figsize is None:
        figsize = (7.5, 9.5)
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)
    fig = plt.figure(**_fopt)
    gs = mpl.gridspec.GridSpec(nr, nc, top=.99, bottom=.04, left=.07, right=.99,
                               wspace=0, hspace=0)
    axs = [[None] * nc] * nr
    cap = 1
    for r in range(nr):
        ymax = -1
        for c in range(nc):
            i = c + r * nc
            if i < nsim:
                opt = dict()
                if c > 0:
                    opt['sharey'] = axs[r][0]
                if r > 0:
                    opt['sharex'] = axs[0][c]
                ax = plt.subplot(gs[r, c], **opt)
                axs[r][c] = ax
                sim = BLsim(sims[i])
                md = sim.mode_detect()
                ymax = md.plot(mklbls=False, legend=(i == 1), ymax=ymax, use_ymax=True,
                               cap=cap, lopt=lopt)
                ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.1))
                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
                lbl = chr(ord('a') + i) + ') ' + sim.name
                if sim.name == 'M09.HR.r.a':
                    plt.ylim(.1, None)
                ax.text(.96, .94, lbl, c='k', transform=ax.transAxes, ha='right',
                        fontsize=6)
                if c:
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    plt.ylabel(r'$\Omega_p$')
                if r == nr -1:
                    plt.xlabel(r'$m$')
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)
                del(md, sim)
                gc.collect()
        cap = .95
    if save:
        plt.savefig(fn)
        plt.close()
    return

def multi_map(map_dict=None, var=None, save=False, figsize=None, dpi=300, fopt=None,
              fn=None, sdir=None, nc=None, nr=None, kind='cons', txt=True, lbl=True,
              skeys=None):
    if map_dict is None:
        map_dict = {'M05.FR.r.a': [200],
                    'M06.FR.r.a': [275],
                    'M09.FR.r.a': [100, 300, 575],
                    'M06.FR.mix.a': [87]
                    }
    if skeys is None:
        skeys = sorted(list(map_dict.keys()))
    golden = (1 + 5 ** 0.5) / 2
    nmap = sum([len(i) for i in map_dict.values()])
    if nr is None:
        nr = int(np.round(np.sqrt(nmap / golden)))
    if nc is None:
        nc = int(np.round(np.sqrt(nmap * golden)))
    if figsize is None:
        margin = .25
        s0 = 2
        figsize = (nc * s0 + margin, nr * s0 + margin)
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)

    def add_plbl(lbl, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.text(.01, .99, '(' + lbl + ')', c='k', transform=ax.transAxes, ha='left',
                va='top', fontsize=8)

    fig = plt.figure(**_fopt)
    gs = mpl.gridspec.GridSpec(nr, nc, top=.99, bottom=.04, left=.07, right=.99,
                               wspace=0, hspace=0)
    axs = [[None] * nc] * nr
    sim = None
    for r in range(nr):
        for c in range(nc):
            i = c + r * nc
            if i < nmap:
                opt = dict()
                if c > 0:
                    opt['sharey'] = axs[r][0]
                if r > 0:
                    opt['sharex'] = axs[0][c]
                ax = plt.subplot(gs[r, c], **opt)
                while not map_dict[skeys[0]]:
                    skeys.pop(0)
                    sim = None
                if sim is None:
                    sim = BLsim(skeys[0])
                with sim.loadfile(kind, map_dict[skeys[0]].pop(0)) as df:
                    df.plot2d(var, ax=ax, cb=False, title=False, minmax=False)
                    ax.yaxis.set_ticks_position('both')
                    ax.xaxis.set_ticks_position('both')
                    ax.tick_params(axis='both', which='both', direction='in')
                    if txt:
                        t = int(df.t / tau + .5)
                        txt = sim.name + "\n" + r"$t/2\pi={:d}$".format(t)
                        ax.text(.96, .99, txt, c='k', transform=ax.transAxes, ha='right',
                                va='top', fontsize=6)
                if lbl:
                    add_plbl(chr(ord('a') + i))
                if c:
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    plt.ylabel(r'$y$')
                if r == nr -1:
                    plt.xlabel(r'$x$')
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)
                gc.collect()
    if save or fn:
        save = True
        if not fn:
            fn = 'multi_maps.pdf'
        if sdir is True:
            sdir = os.path.split(sim.path)[0]
            sdir = os.path.join(os.path.split(sdir, 'figs'))
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if save:
        plt.savefig(fn)
        plt.close()
    return

def multi_stripe(plots=None, var=None, save=False, figsize=None, dpi=300, fopt=None,
              fn=None, sdir=None, nc=None, nr=None, kind='cons', txt=True, lbl=True,
              lnorm=-2, overwrite=True, cb_side=None, tsz=10):
    if plots is None:
        plots = [dict(sim='M07.FR.r.a', t=450),
                 dict(sim='M09.FR.r.lc.a', t=175, ps=.316, mode=19, rm_last=True),
                 dict(sim='M13.FR.r.a', t=375),
                 ]
        plots = [dict(sim='M06.HR.r.lc.a', t=25, op=0.75755, mode=7, phi0=1.2 * np.pi),
                 dict(sim='M09.FR.r.a', t=175, ps=.315, mode=19, op=.315, ru=1.5,
                      phi0=1.63 * np.pi),
                 #dict(sim='M13.FR.r.a', t=375),
                 ]
        if nr is None and nc is None:
            nr = 2
            nc = 1
            cb_side = 'right'
    golden = (1 + 5 ** 0.5) / 2
    nplots = len(plots)
    if cb_side is None:
        cb_side = 'top'
    cb_side = cb_side.lower()
    assert cb_side in ['top', 'right']
    if nr is None:
        nr = int(np.round(np.sqrt(nplots / golden)))
    if nc is None:
        if nr is False or nr == 1:
            nc = nplots
            nr = 1
        else:
            nc = int(np.round(np.sqrt(nplots * golden)))
    if figsize is None:
        margin = .25
        s0 = 2.5
        width, height = nc * s0 + margin, nr * s0 + margin
        if cb_side in ['right']:
            width += margin * 3
        figsize = width, height
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)

    if save and not fn:
        fn = 'multi_stripes.png'
    if parse_not_overwrite(overwrite, fn):
        return None

    def add_plbl(lbl, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.text(.01, .97, '(' + lbl + ')', c='w', transform=ax.transAxes, ha='left',
                va='top', fontsize=tsz)

    fig = plt.figure(**_fopt)
    hr = [.1, 1] * nr
    gs_opt = dict(top=.9, bottom=.15, left=.07, right=.93, wspace=0, hspace=0)
    if cb_side == 'top':
        gs_opt['height_ratios'] = [.1, 1] * nr
        gs = mpl.gridspec.GridSpec(nr * 2, nc, **gs_opt)
        orientation = 'horizontal'
    elif cb_side == 'right':
        gs_opt.update(dict(top=.93, bottom=.08, left=.15, right=.87))
        gs_opt['width_ratios'] = [1, .05] * nc
        gs = mpl.gridspec.GridSpec(nr, nc * 2, **gs_opt)
        orientation = 'vertical'
    axs = [[None] * nc] * nr
    sim = None

    fmt = oomfmt(lnorm)
    for r in range(nr):
        for c in range(nc):
            i = c + r * nc
            if i < nplots:
                opt = dict()
                if c > 0:
                    opt['sharey'] = axs[r][0]
                if r > 0:
                    opt['sharex'] = axs[0][c]
                if cb_side == 'top':
                    ax = plt.subplot(gs[2 * r + 1, c], **opt)
                    cax = plt.subplot(gs[2 * r, c])
                elif cb_side == 'right':
                    ax = plt.subplot(gs[r, 2 * c], **opt)
                    axs[r][c] = ax
                    cax = plt.subplot(gs[r, 2 * c + 1])
                sname = plots[i]['sim']
                if getattr(sim, 'name', None) != sname:
                    sim = BLsim(sname)
                with sim.loadfile(kind, plots[i]['t']) as df:
                    cbopt = dict(orientation=orientation, format=fmt)
                    sopt = dict(ax=ax, cax=cax, title=False, lbls=False, cbopt=cbopt,
                                cbl=False)
                    for key in plots[i].keys():
                        if key not in ['sim', 't']:
                            sopt[key] = plots[i][key]
                    if 'opt' in plots[i]:
                        sopt.update(plots[i]['opt'])
                    #print(sname, sopt)
                    df.stripe(var, **sopt)
                    ax.yaxis.set_ticks_position('both')
                    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
                    ax.xaxis.set_ticks_position('both')
                    ax.tick_params(axis='both', which='both', direction='in')

                    cax.xaxis.set_ticks_position('top')
                    cax.xaxis.set_label_position('top')
                    cax.tick_params(axis='both', which='both', direction='in')
                    cax.xaxis.get_offset_text().set_visible(False)
                    cax.yaxis.get_offset_text().set_visible(False)
                    if txt:
                        t = int(df.t / tau + .5)
                        txt = sim.name + "\n" + r"$t/2\pi={:d}$".format(t)
                        ax.text(.96, .96, txt, c='k', transform=ax.transAxes, ha='right',
                                va='top', fontsize=tsz)
                if lbl:
                    add_plbl(chr(ord('a') + i))
                if c:
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    plt.ylabel(r'$\phi/\pi$')
                if r == nr -1:
                    plt.xlabel(r'$r$')
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if cb_side == 'top':
                    if r == 0 and c == nc - 1:
                        cax.text(1.01, 1.13, r'$\times 10^{{{}}}$'.format(lnorm),
                                 transform=ax.transAxes, ha='left', va='bottom')#, fontsize=6)
                elif r == 0:
                    plt.title(r'$r v_r \sqrt{{\Sigma}}/10^{{{}}}$'.format(lnorm), fontsize=tsz)
                else:
                    yticks = ax.yaxis.get_major_ticks()
                    yticks[-1].label1.set_visible(False)
                    yticks[-2].label1.set_visible(False)
                gc.collect()
    plt.draw()
    if save or fn:
        save = True
        if sdir is True:
            sdir = os.path.split(sim.path)[0]
            sdir = os.path.join(os.path.split(sdir, 'figs'))
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if save:
        plt.savefig(fn)
        plt.close()
    return

def vortex_types(plots=None, lvar=None, rvar=None, save=False, figsize=None, dpi=300,
                 fopt=None, fn=None, sdir=None, kind='cons', txt=True, lbl=True,
                 overwrite=True, cb_side=None, tsz=10, hspace=.01):
    if plots is None:
        rmin, rmax = .98, 2.0
        _ropt = dict(cbl=None, xminor=False, lnorm=-2)
        ropt = dict(rmax=1*rmax)
        ropt.update(_ropt)
        _lopt = dict(rmin=rmin, vmax='98%', cbl=r'$\omega/\Sigma-\left.\left<\omega/\Sigma\right>\right|_{t=0}$', xminor=False)
        rmax = rmin + (rmax - rmin) / 3
        lopt = dict(rmax=1*rmax)
        lopt.update(_lopt)
        plots = [dict(sim='M11.FR.r.a', t=173, lopt=lopt, ropt=ropt, rp=1.1, phi0=.68*np.pi)]

        rmin, rmax = .98, 3.5
        ropt = dict(rmax=1*rmax)
        rmax = rmin + (rmax - rmin) / 3
        lopt = dict(rmax=1*rmax)
        lopt.update(_lopt)
        lopt['cbl'] = False
        ropt['cbl'] = False
        plots.append(dict(sim='M07.FR.r.a', t=450, lopt=lopt, ropt=ropt, op=.538, mode=4, phi0=1.45*np.pi, vmax=1.1e-2))
    if lvar is None:
        lvar = 'd_vortensity_0'
    if rvar is None:
        rvar = 'Rpseudo'
    nr = len(plots)
    nc = 2
    if cb_side is None:
        cb_side = 'top'
    cb_side = cb_side.lower()
    assert cb_side in ['top', 'side']
    if figsize is None:
        figsize = 3, 5
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)

    if save and not fn:
        fn = 'vortex_types.png'
    if parse_not_overwrite(overwrite, fn):
        return None

    def add_plbl(lbl, ax=None, side=0):
        if ax is None:
            ax = plt.gca()
        x = .02 if side else .06
        bbox = dict(facecolor='w', alpha=0.5, edgecolor='none', clip_on=True)
        ax.text(.01, .97, '(' + lbl + ')', c='k', transform=ax.transAxes, ha='left',
                    va='top', fontsize=tsz, bbox=bbox, clip_on=True)

    fig = plt.figure(**_fopt)
    hr = [.1, 1] * nr
    gs_opt = dict(top=.9, bottom=.0, left=.15, right=.93, wspace=0, hspace=hspace)
    if cb_side == 'top':
        gs_opt['height_ratios'] = [.05, 1, .3] * nr
        gs_opt['width_ratios'] = [1/3.0, 1]
        gs = mpl.gridspec.GridSpec(nr * 3, nc, **gs_opt)
        orientation = 'horizontal'
        _nx, _ny = 2, 3
        _dx, _dy = 0, 1
    elif cb_side == 'side':
        gs_opt.update(dict(top=.93, bottom=.08, left=.15, right=.87))
        gs_opt['width_ratios'] = [.05, 1/3.0, 1, .05]
        gs = mpl.gridspec.GridSpec(nr, nc * 2, **gs_opt)
        orientation = 'vertical'
        _nx, _ny = 4, 1
        _dx, _dy = 1, 0
    sim = None
    axs = [None, None]

    for r in range(nr):
        sname = plots[r]['sim']
        if getattr(sim, 'name', None) != sname:
            sim = BLsim(sname)
        with sim.loadfile(kind, plots[r]['t']) as df:
            for c in range(nc):
                i = c + r * nc
                opt = dict()
                if c > 0:
                    opt['sharey'] = axs[0]
                if r > 0:
                    opt['sharex'] = None  # axs[c]
                ax = plt.subplot(gs[r * _ny + _dy, c + _dx], **opt)
                if cb_side == 'top':
                    cax = plt.subplot(gs[r * _ny, c])
                elif cb_side == 'side':
                    cax = plt.subplot(gs[r * _ny, -1 if c else 0])
                axs[c] = ax
                cbopt = dict(orientation=orientation)
                sopt = dict(ax=ax, cax=cax, title=False, lbls=False, cbopt=cbopt,
                            cbl=False)
                for key in plots[r].keys():
                    if key not in ['sim', 't', 'lopt', 'ropt']:
                        sopt[key] = plots[r][key]
                if ['lopt', 'ropt'][c] in plots[r]:
                    sopt.update(plots[r][['lopt', 'ropt'][c]])
                #print(sname, sopt)
                df.stripe([lvar, rvar][c], **sopt)
                ax.yaxis.set_ticks_position('both')
                # ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
                ax.xaxis.set_ticks_position('both')
                ax.tick_params(axis='both', which='both', direction='in')

                cax.xaxis.set_ticks_position('top')
                cax.xaxis.set_label_position('top')
                cax.tick_params(axis='both', which='both', direction='in')
                cax.xaxis.get_offset_text().set_visible(False)
                cax.yaxis.get_offset_text().set_visible(False)
                if txt and c == 1:
                    t = int(df.t / tau + .5)
                    txt = sim.name + "\n" + r"$t/2\pi={:d}$".format(t)
                    ax.text(.9, .96, txt, c='k', transform=ax.transAxes, ha='right',
                            va='top', fontsize=tsz)
                if lbl:
                    add_plbl(chr(ord('a') + i), side=c)
                if c:
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    plt.ylabel(r'$\phi/\pi$')
                if r == nr -1:
                    plt.xlabel(r'$r$')
                else:
                    # plt.setp(ax.get_xticklabels(), visible=False)
                    pass
                #if cb_side == 'top':
                #    if r == 0 and c == nc - 1:
                #        cax.text(1.01, 1.13, r'$\times 10^{{{}}}$'.format(lnorm),
                #                 transform=ax.transAxes, ha='left', va='bottom')#, fontsize=6)
                #elif r == 0:
                #    plt.title(r'$r v_r \sqrt{{\Sigma}}/10^{{{}}}$'.format(lnorm), fontsize=tsz)
                #else:
                #    yticks = ax.yaxis.get_major_ticks()
                #    yticks[-1].label1.set_visible(False)
                #    yticks[-2].label1.set_visible(False)
        gc.collect()
    plt.draw()
    if save or fn:
        save = True
        if sdir is True:
            sdir = os.path.split(sim.path)[0]
            sdir = os.path.join(os.path.split(sdir, 'figs'))
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if save:
        plt.savefig(fn)
        plt.close()
    return

def multi_st(sims=None, opts=None, save=False, figsize=None, dpi=300, fopt=None,
             fn=None, sdir=None, rmin=1.0, rmax=2.0, vmax1=None, vmax2=None, overwrite=True):
    if sims is None:
        sims = ['M06R.H.r.lc.a', 'M09.FR.r.lc.a', 'M11.FR.r.a', 'M12.FR.r.lc.a',
                'M15.FR.r.a']
        sims = ['M07.FR.r.a', 'M09.FR.r.lc.a', 'M12.FR.r.lc.a', 'M15.FR.r.a']
    if save or fn:
        save = True
        if not fn:
            fn = 'multi_st.png'
        if sdir is True:
            sdir = os.path.join(rc('dirs')[-1], 'figs')
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if parse_not_overwrite(overwrite, fn):
        return None
    nvar = 3
    nsim = len(sims)
    if figsize is None:
        margin = .25
        s0 = 1.8
        figsize = (1.5 * nsim * s0 + margin, nvar * s0 + margin)
        #figsize = (7,3.5)
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)
    fig = plt.figure(**_fopt)
    gs = mpl.gridspec.GridSpec(nvar, nsim + 1, top=.985, bottom=.07, left=.035, right=.95,
                               wspace=0, hspace=0, width_ratios=[1] * nsim + [.04])
    col1 = [None] * nvar
    r1lim = -np.inf
    r2lim = -np.inf
    axs = [[None] * nvar] * nsim
    lbl = 'a'
    for j, s in enumerate(sims):
        sim = BLsim(s)
        sharex = None
        for i in range(nvar):
            sharey = col1[i]
            ax = plt.subplot(gs[i, j], sharex=sharex, sharey=sharey)
            sharex = ax
            axs[j][i] = ax
            if j == 0:
                col1[i] = ax
            opt = dict(ax=ax, cb=False, rplot=(rmin < 1), ylbl=j==0, xlbl=i==nvar-1)
            if j == nsim - 1:
                cax = plt.subplot(gs[i, j + 1])
                opt['cb'] = True
                opt['cax'] = cax
            if opts:
                opt.update(opts[i])
            if i == 0:
                sim.rho_st(delta=True, norm=MidpointNormalize(-1, .1, 0),
                            cmap=helpers.NCcmap, vmin=-1, vmax=.1, zerocent=False, **opt)
                plt.text(.5, .9, sim.name, c='k', ha='center',
                         transform=ax.transAxes)
            if i == 1:
                sim.stress_st(vmax=vmax1, **opt)
                r1lim = max(r1lim, max(plt.gci().get_clim()))
                #print(j, 'r1lim', r1lim)
            if i == 2:
                sim.acc_st(vmax=vmax2, **opt)
                r2lim = max(r2lim, max(plt.gci().get_clim()))
                #print(j, 'r2lim', r2lim)
            if i != nvar - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.text(.98, .97, lbl + ')', c='w', transform=ax.transAxes, ha='right',
                va='top', fontsize=8)
            lbl = chr(ord(lbl) + 1)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(25))
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(axis='both', which='both', direction='in')
            plt.ylim(rmin, rmax)
            if i != 0:
                yticks = ax.yaxis.get_major_ticks()
                yticks[-1].label1.set_visible(False)
            if j == nsim - 1:
                yticks = cax.yaxis.get_major_ticks()
                yticks[0].label2.set_visible(False)
            gc.collect()
        del (sim)
        gc.collect()

    if vmax1 is None:
        #print('r1lim', r1lim)
        for j in range(nsim):
            for im in axs[j][1].get_images():
                im.set_clim(-r1lim, r1lim)
    if vmax2 is None:
        #print('r2lim', r2lim)
        for j in range(nsim):
            for im in axs[j][1].get_images():
                im.set_clim(-r2lim, r2lim)

    if save:
        plt.savefig(fn)
        plt.close()
    return

def multi_omega(sims=None, var=None, save=False, figsize=None, dpi=300, fopt=None,
                fn=None, sdir=None, nc=None, nr=None, file='cons', txt=True, lbl=True,
                lnorm=-2, cmap=None, tmin=0, tmax=600, popt=None, rmin=.9, rmax=1.35,
                overwrite=True, use_maps=False):
    if sims is None:
        sims = ['M06.HR.r.lc.a', 'M07.FR.r.a', 'M08.FR.r.a', 'M09.FR.r.lc.a',
                'M10.FR.r.a', 'M11.FR.r.a', 'M12.FR.r.lc.a', 'M13.FR.r.a', 'M14.FR.r.a',
                'M15.FR.r.a']
    nsim = len(sims)
    if save or fn:
        save = True
        if not fn:
            fn = 'multi_omega.pdf'
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if parse_not_overwrite(overwrite, fn):
        return None
    if nr is None:
        nr = 2
    if nc is None:
        nc = int(np.ceil(nsim / nr))
    if figsize is None:
        margin = .25
        s0 = 2
        figsize = (nc * s0 + margin, nr * s0 + margin)
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)
    if cmap is None:
        # sample the colormaps that you want to use. Use 128 from each so we get 256
        # colors in total
        n0 = int(np.ceil(51. / 600 * 256))
        colors1 = plt.cm.cividis(np.linspace(0., 1, n0))
        colors2 = plt.cm.viridis_r(np.linspace(0, .7, 256 - n0))

        # combine them and build a new colormap
        colors = np.vstack((colors1, colors2))
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
    elif hasattr(cmap, 'lower'):
        cmap = plt.get_cmap(cmap)
    if popt is None:
        popt = dict(lw=1, ls='-')
    norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    times = np.array([0, 20, 25, 30, 40, 50] +
                     list(range(100, int(tmax) + 2, 100)))
    colors = cmap(norm(times))
    fig = plt.figure(**_fopt)
    gs = mpl.gridspec.GridSpec(nr, nc + 1, top=.98, bottom=.09, left=.05, right=.94,
                               wspace=0, hspace=0, width_ratios=[1] * nc + [.06])
    col1 = [None] * nr
    for r in range(nr):
        for c in range(nc):
            i = c + r * nc
            sharex = None
            sharey = col1[r]
            if i < nsim:
                sim = BLsim(sims[i])
                omega = []
                fd = None
                if use_maps:
                    print(sim)
                    for t in times:
                        print('t', t)
                        with sim.loadfile('cons', int(t)) as df:
                            omega.append(df['mom2'].mean(axis=0) /
                                         (sim.rc * df['dens'].mean(axis=0)))
                        gc.collect()
                else:
                    fd = sim.load_flux_data()
                    omega = fd['vphi'] / (sim.rc)
                ax = plt.subplot(gs[r, c], sharex=sharex, sharey=sharey)
                sharex = ax
                if c == 0:
                    col1[r] = ax
                for t in range(len(times)):
                    loc = t if use_maps else np.abs(fd['t'] - times[t] * tau).argmin()
                    plt.plot(sim.rc, omega[loc], c=colors[t], **popt)
                if r == nr - 1:
                    plt.xlabel(r'$r$')
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if c == 0:
                    plt.ylabel(r'$\Omega$')
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(25))
                ax.xaxis.set_ticks_position('both')
                ax.tick_params(axis='both', which='both', direction='in')
                ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.05))
                plt.xlim(rmin, rmax)
                plt.ylim(0, 1)
                if txt:
                    lbl = chr(ord('a') + i) + ') ' + sim.name
                    ax.text(.98, .94, lbl, c='k', transform=ax.transAxes, ha='right',
                            fontsize=6)
                if c != nc -1:
                    xticks = ax.xaxis.get_major_ticks()
                    xticks[-1].label1.set_visible(False)
                if r != 0:
                    yticks = ax.yaxis.get_major_ticks()
                    yticks[-1].label1.set_visible(False)
                del(sim, fd, omega)
                gc.collect()
    cax = plt.subplot(gs[:, -1])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    cb.set_label(r'$t/2\pi$')

    if save:
        plt.savefig(fn)
        plt.close()
    return

def multi_flux(sims=None, save=False, figsize=None, dpi=300, fopt=None, fn=None,
               sdir=None, txt=True, lbl=True, spacer=True, rmin=None, rmax=None, nm=5,
               lopt=None, lnorm=-3, popt=None, new=True):
    if sims is None:
        sims = [dict(name='M06.HR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                dict(name='M09.FR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                dict(name='M11.FR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                dict(name='M12.FR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                dict(name='M15.FR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                ]
        sims = [dict(name='M06.HR.r.lc.a', ts=[[50, 150], [250, 350], [500, 600]]),
                dict(name='M06.HR.r.lc.a', ts=[[50, 150], [250, 350], [500, 600]]),
                dict(name='M06.HR.r.lc.a', ts=[[50, 150], [250, 350], [500, 600]]),
                ]
    nsim = len(sims)
    if not lnorm:
        lnorm = 0
    if save or fn:
        save = True
        if not fn:
            fn = 'multi_flux.pdf'
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    nc = nsim
    nvar = 3
    nr = max([len(i['ts']) for i in sims])
    _popt = dict(lw=1)
    if popt is not None:
        _popt.update(popt)
    if spacer:
        nr = (nr + 1) * nvar - 1
    else:
        nr *= nvar
    if lopt is None:
        lopt = dict(handlelength=1, fontsize=8, handletextpad=.4, columnspacing=.7)
    if figsize is None:
        figsize = np.array([8.5, 11]) * 2
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)
    fig = plt.figure(**_fopt)

    hr = [1, .6, 1]
    if spacer:
        hr.append(.15)
    hr *= nvar
    if spacer:
        hr.pop(-1)
    gs = mpl.gridspec.GridSpec(nr, nc, top=.98, bottom=.09, left=.05, right=.94,
                               hspace=0, wspace=0.1, height_ratios=hr)
    pre = r'$\mathcal{M}^2'
    suf = '/10^{' + str(lnorm) + '}$' if lnorm else '$'
    for ns, s in enumerate(sims):
        with BLsim(s['name']) as sim:
            _nm = s.get('nm', nm)
            rlim = np.empty(2)
            rlim[0] = sim.r[0] if rmin is None else rmin
            rlim[1] = sim.r[-1] if rmax is None else rmax
            rin = sim.rloc(max(1, rlim[0]))
            rout = sim.rloc(min(3.9, rlim[1]))
            row = 0
            ax = None
            msqr = sim.mach**2 * 10**-lnorm
            for nt, tlim in enumerate(s['ts']):
                try:
                    sims[ns]['data'][nt]
                except KeyError:
                    sims[ns]['data'] = [sim.new_fluxes(*tlim, csm=True)]
                except IndexError:
                    sims[ns]['data'].append(sim.new_fluxes(*tlim, csm=True))
                finally:
                    data = sims[ns]['data'][nt]
                csm = np.real(data['CSm'])
                csm[0] = 0
                cs = data['CS']
                norm = sim.intr(np.abs(csm))
                modes = sorted(range(norm.shape[0]), key=lambda x: -norm[x])

                # C_S, C_S,m
                ax = plt.subplot(gs[row, ns], sharex=ax)
                plt.plot(sim.rc, msqr * cs, 'k-', label='$C_S$')
                lines = [msqr * cs]
                for m in modes[:_nm]:
                    lines.append(msqr * csm[m])
                    plt.plot(sim.rc, msqr * csm[m], label=str(m), **_popt)
                plt.plot(sim.rc, msqr * csm[1:].sum(axis=0), c='.5', ls=':', label='sum',
                         **_popt)
                lines.append(msqr * csm[1:].sum(axis=0))
                plt.xlim(*rlim)
                # ylim = plt.ylim()
                lines = np.array(lines)
                yu = lines[:, rin:rout + 1].max()
                yl = lines[:, rin:rout + 1].min()
                d = (yu - yl) * .01
                yu += d
                yl -= d
                plt.ylim(yl, yu)
                plt.legend(loc='lower center', ncol=_nm + 2, **lopt)
                plt.axhline(0, c='.5', ls=':', lw=1)
                plt.axvline(1, c='.5', ls=':', lw=1)
                # plt.ylim(*ylim)
                # plt.xlabel('$R$')
                if ns == 0:
                    plt.ylabel(pre +'C_S' + suf)
                # plt.setp(ax0.get_xticklabels(), fontsize=6)
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                ax.tick_params(axis='both', which='both', direction='in')
                if txt:
                    time = r"$t/2\pi={:d}-{:d}$".format(*tlim)
                    if row == 0:
                        plt.title(sim.name + ' ' + time)
                    else:
                        plt.title(time)
                def mklbl(lbl):
                    if lbl:
                        tmp = row - nt if spacer else row
                        lbl = chr(ord('A') + ns) + chr(ord('a') + tmp) + ')'
                        bbox = dict(facecolor='w', alpha=0.5, edgecolor='none')
                        ax.text(.03, .95, lbl, c='k', transform=ax.transAxes, ha='left',
                                va='top', fontsize=6, bbox=bbox)
                mklbl(lbl)
                plt.setp(ax.get_xticklabels(), visible=False)
                row += 1

                # C_L, C_A, C_S
                ax = plt.subplot(gs[row, ns], sharex=ax)
                keys = [i for i in data.keys() if i[0] == 'C' and len(i) == 2]
                yu = []
                yl = []
                for k in keys:
                    opt = {'label': '${0:}_{1:}$'.format(*k)}
                    opt.update(_popt)
                    if k == 'CS':
                        opt['c'] = 'k'
                    plt.plot(sim.rc, msqr * data[k], **opt)
                    r0 = max(sim.rloc(rlim[0]), 20)
                    yu.append(np.real(msqr * data[k][r0:rout + 1]).max())
                    yl.append(np.real(msqr * data[k][r0:rout + 1]).min())
                plt.legend(loc='lower center', ncol=3, **lopt)
                plt.axhline(0, c='.5', ls=':', lw=1)
                plt.axvline(1, c='.5', ls=':', lw=1)
                plt.xlim(sim.r[0], sim.r[-1])
                yl, yu = np.real(yl).min(), np.real(yu).max()
                dy = (yu - yl) * .05
                plt.ylim(yl - dy, yu + dy)
                # plt.xlabel('R')
                plt.setp(ax.get_xticklabels(), visible=False)
                if ns ==0:
                    plt.ylabel(pre + 'C_i' + suf)
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                ax.tick_params(axis='both', which='both', direction='in')
                mklbl(lbl)
                row += 1

                # M-dot panel
                # see AR18 Eqn. 4, BRS13a Eqn. 64
                ax = plt.subplot(gs[row, ns], sharex=ax)
                r1 = sim.rloc(1.2)
                ri2 = sim.rloc(2)
                lines = []
                # C_S, blue curve
                norm = 1 / grad(sim.rc, data['vphi'] * sim.rc)
                ycs = norm * grad(sim.rc, data['CS'])
                lines.append(msqr * ycs)
                plt.plot(sim.rc, lines[-1], label=r'$C_S$', **_popt)
                # d_t Omega, orange curve
                ydw = norm * sim.rc ** 3 * data['dens'] * data['dwdt'] * tau
                lines.append(msqr * ydw)
                plt.plot(sim.rc, lines[-1], label=r'$\partial_t \Omega$', **_popt)
                # d_t d_r P, green curve
                ydp = np.pi * sim.rc ** 3.5 * data['dtdr_rho'] * sim.mach ** -2
                ydp /= grad(sim.rc, data['vphi'] * sim.rc)
                lines.append(msqr * ydp)
                plt.plot(sim.rc, lines[-1], label=r'$\partial_t\partial_rP$', ls='--',
                         **_popt)
                # Mdot, black curve
                lines.append(msqr * -data['Mdot'])
                plt.plot(sim.rc, lines[-1], label=r'$\dot{M}$', c='k', **_popt)
                # C_S + d_t Omega, gray curve
                lines.append(msqr * (ycs + ydw))
                plt.plot(sim.rc, lines[-1], label=r'$C_S\! +\! \partial_t \Omega$',
                         c='.5', ls='-', **_popt)
                # axes settings
                plt.axhline(0, c='.5', ls=':', lw=1)
                plt.axvline(1, c='.5', ls=':', lw=1)
                plt.legend(loc='upper center', ncol=5, **lopt)
                lines = np.array(lines)
                yu = lines[:, rin:rout+1].max()
                yl = lines[:, rin:rout+1].min()
                d = (yu - yl) * .03
                yu += d
                yl -= d
                ylim = plt.ylim()
                ylim = plt.ylim(max(ylim[0], yl), min(ylim[1], yu))
                plt.xlim(rmin, rmax)
                if ns ==0:
                    plt.ylabel(pre + '\dot{M}' + suf)
                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                ax.tick_params(axis='both', which='both', direction='in')
                if nt < len(s['ts']) - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    plt.xlabel('$R$')
                mklbl(lbl)
                plt.xlim(*rlim)
                ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
                row += 1
                if spacer:
                    row += 1

    return sims


def AM_plot(sims=None, save=False, figsize=None, dpi=300, fopt=None, fn=None, sdir=None,
            txt=True, lbl=True, spacer=True, rmin=None, rmax=None, nm=5, lopt=None,
            lnorm=-3, popt=None, new=True, o=2, s=1, plt_ydp=False, overwrite=True):
    if sims is None:
        sims = [dict(name='M06.HR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                dict(name='M09.FR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                dict(name='M11.FR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                dict(name='M12.FR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                dict(name='M15.FR.r.a', ts=[[100, 200], [300, 400], [500, 600]]),
                ]
        sims = [dict(name='M06.HR.r.lc.a', ts=[[50, 150], [250, 350], [500, 600]]),
                dict(name='M09.FR.r.lc.a', ts=[[100, 200], [350, 450], [550, 600]]),
                dict(name='M12.FR.r.lc.a', ts=[[50, 150], [250, 350], [500, 600]]),
                ]
    nsim = len(sims)
    if not lnorm:
        lnorm = 0
    if save or fn:
        save = True
        if not fn:
            fn = 'AM_curves.pdf'
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if parse_not_overwrite(overwrite, fn):
        return None
    nc = nsim
    nvar = 3
    nr = max([len(i['ts']) for i in sims])
    _popt = dict(lw=1)
    if popt is not None:
        _popt.update(popt)
    if spacer:
        nr = (nr + 1) * nvar - 1
    else:
        nr *= nvar
    if lopt is None:
        lopt = dict(handlelength=1, fontsize=8, handletextpad=.4, columnspacing=.7)
    if figsize is None:
        figsize = np.array([8.5, 11]) * 2
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)
    fig = plt.figure(**_fopt)

    hr = [1, .6, 1]
    if spacer:
        hr.append(.17)
    hr *= nvar
    if spacer:
        hr.pop(-1)
    gs = mpl.gridspec.GridSpec(nr, nc, top=.98, bottom=.02, left=.05, right=.99, hspace=0,
                               wspace=0.1, height_ratios=hr)
    pre = r'$\mathcal{M}^2'
    suf = '/10^{' + str(lnorm) + '}$' if lnorm else '$'
    _s = s
    for ns, s in enumerate(sims):
        with BLsim(s['name']) as sim:
            _nm = s.get('nm', nm)
            rlim = np.empty(2)
            rlim[0] = sim.r[0] if rmin is None else rmin
            rlim[1] = sim.r[-1] if rmax is None else rmax
            rin = sim.rloc(max(1, rlim[0]))
            rout = sim.rloc(min(3.9, rlim[1]))
            row = 0
            ax = None
            msqr = sim.mach**2
            for nt, tlim in enumerate(s['ts']):
                axs = []
                for i in range(3):
                    ax = plt.subplot(gs[row + i, ns], sharex=ax)
                    axs.append(ax)
                plt.sca(axs[0])
                time = r"$t/2\pi={:d}-{:d}$".format(*tlim)
                if row == 0:
                    plt.title(sim.name + ' ' + time)
                else:
                    plt.title(time)
                sim.paper_flux_plot(o=o, s=_s, t0=tlim[0], tf=tlim[1], axs=axs, norm=msqr,
                                    lnorm=lnorm, use_txt=False,  plt_ydp= plt_ydp)
                if nt < 2:
                    plt.xlabel('')
                for i, ax in enumerate(axs):
                    tmp = (row - nt if spacer else row) + i
                    lbl = chr(ord('A') + ns) + chr(ord('a') + tmp) + ')'
                    bbox = dict(facecolor='w', alpha=0.5, edgecolor='none')
                    x = .03 if i < 2 else .1
                    ax.text(x, .95, lbl, c='k', transform=ax.transAxes, ha='left',
                            va='top', fontsize=6, bbox=bbox)
                    if ns == 0:
                        axs[0].set_ylabel(pre +'C_S' + suf)
                        axs[1].set_ylabel(pre +'C_i' + suf)
                        axs[2].set_ylabel(pre + r'\left[{\rm AM\; terms}\right]' + suf)
                    else:
                        [ax.set_ylabel('') for ax in axs]
                row += 3
                if spacer:
                    row += 1
    if save:
        plt.savefig(fn)
        plt.close()
    return


def am_subpanels(sims=None, save=False, figsize=None, dpi=300, fopt=None, fn=None, sdir=None, txt=True, lbl=True, spacer=True, rmin=None, rmax=None, nm=5, lopt=None, lnorm=-3, popt=None, overwrite=True, gsopt=None):
    if sims is None:
        sims = 1
    if sims in [1 , 2]:
        _sims = [dict(name='M07.FR.r.a', ts=[[100, 200, 125], [200, 300, 225], [550, 600, 575]]),
                 dict(name='M09.FR.r.lc.a', ts=[[100, 200, 150], [250, 350, 275], [400, 500, 450]]),
                 dict(name='M12.FR.mix.lc.a', ts=[[100, 200, 150], [250, 350, 300], [500, 600, 550]]),
                 dict(name='M15.FR.r.a', ts=[[100, 200, 150], [300, 400, 350], [500, 600, 550]]),
                 ]
        if save and not fn:
            fn = 'am_subpanels_{}.png'.format(sims)
        if sims == 1:
            sims = _sims[0:2]
        elif sims == 2:
            sims = _sims[2:]
    nsim = len(sims)
    if not lnorm:
        lnorm = 0
    if save or fn:
        save = True
        if not fn:
            fn = 'am_subpanels.png'
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if parse_not_overwrite(overwrite, fn):
        return None
    nc = nsim
    nvar = 3
    _popt = dict(lw=1)
    if popt is not None:
        _popt.update(popt)
    if lopt is None:
        lopt = dict(handlelength=1, fontsize=8, handletextpad=.4, columnspacing=.7)
    if figsize is None:
        figsize = np.array([8.5, 11]) * 2
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt:
        _fopt.update(fopt)
    nt = len(sims[0]['ts'])
    assert np.all([len(s['ts']) == nt for s in sims])
    fig = plt.figure(**_fopt)
    _gsopt = dict(top=.98, bottom=.02, left=.05, right=.99, hspace=0.1, wspace=0.1)
    if gsopt is not None:
        _gsopt.update(gsopt)
    gs = mpl.gridspec.GridSpec(nt, nsim, **_gsopt)
    for col, args in enumerate(sims):
        sim = BLsim(args['name'])
        opt = args.get('opt', dict())
        if type(opt) == dict:
            opt = [opt] * nt
        for row in range(nt):
            ts = args['ts'][row]
            sim.am_subpanel(*ts, gs0=gs[row, col], save=False, ylbl=(col==0),
                             xlbl=(row==nt-1), hdf5=args.get('hdf5'), **opt[row])
    if save:
        plt.savefig(fn)
        plt.close()
    return


def mode_hist(sims=None, fn=None, save=None, data=None, ll=True, overwrite=True):
    if fn is None and save:
        fn = 'mode_hist.pdf'
    if parse_not_overwrite(overwrite, fn):
        return None
    if data is None:
        if hasattr(sims, 'lower'):
            if os.path.isfile(sims):
                with open(sims) as f:
                    sims = [l.split('#')[0].strip() for l in f.readlines()]
                sims = [s for s in sims if s]
        data = dict()
        tmp = comp_wrapper('dict_sdg_modes', sims, ll=ll)
        for i in tmp:
            data.update(i)

    nmodes = 3
    hist = dict()
    for i in data:
        mach = int(i[1:3])
        tmp = data[i]
        nsets = len(tmp) + 1
        if not mach in hist:
            hist[mach] = np.zeros((nsets, 32))
        for j in range(nsets - 1):
            for m in tmp[j][:nmodes]:
                hist[mach][j, m] += 1
        for m in tmp[-1]:
            hist[mach][2, m] += 1

    dx = .05
    offset = 2.5 * dx
    hmax = 0
    for i in hist:
        hmax = max(hmax, hist[i].max())
    names = ['Reds', 'Blues', 'Greys', 'Greens']
    cmaps = []
    for i in range(nsets):
        cm = plt.cm.get_cmap(names[i], hmax + 1)
        colors = cm(np.linspace(0, 1,  hmax + 1))
        colors[0, :] = np.array([1,1,1,0])
        cmaps.append(ListedColormap(colors))
    #cmaps = [plt.cm.get_cmap(cmaps[i], hmax + 1) for i in range(nsets)]
    ims = [None,] * nsets
    imopt = dict(vmin=-.5, vmax=hmax+.5, interpolation='nearest')

    fig = plt.figure(figsize=(4.5, 3), dpi=300)
    gs = mpl.gridspec.GridSpec(1, nsets + 2, width_ratios=[1, .05] + [.05,] * nsets, top=.99,
                               bottom=.15, left=.1, right=.93, wspace=0)
    ax = plt.subplot(gs[0])
    for i in hist:
        for j in range(nsets):
            extent = [i - dx + offset * (j - 1), i + dx + offset * (j - 1), 0, 31]
            ims[j] = plt.imshow(np.array([hist[i][j]]).T, extent=extent, cmap=cmaps[j], **imopt)
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    yl = plt.ylim()
    xl = plt.xlim(4.5, 15.5)
    plt.plot(xl, xl, c='.5', lw=1, ls=':', zorder=-10)
    plt.ylim(*yl)
    plt.xlim(*xl)
    plt.xlabel(r'$\mathcal{M}$')
    plt.ylabel(r'$m$')
    names = ['star', 'disk', 'global']
    for j in range(nsets - 1):
        cax = plt.subplot(gs[2 + j])
        t = []
        if j ==  nsets - 1:
            t = np.arange(hmax+1)
        cb = plt.colorbar(ims[j], cax=cax, ticks=t)
        cax.set_xlabel(names[j], fontsize=8, rotation='vertical')

    if fn or save:
        plt.savefig(fn)
        plt.close()

    return data


def multi_mdot_split(sims=None, save=False, figsize=None, dpi=300, fopt=None, fn=None,
                     sdir=None, lbl=True, lnorm=True, overwrite=True, lloc=4):
    if save or fn:
        save = True
        if not fn:
            fn = 'multi_mdot_split.pdf'
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if parse_not_overwrite(overwrite, fn):
        return None
    _opt = dict(title=False, lnorm=lnorm, lloc=lloc)
    if sims is None:
        sims = [dict(name='M06.HR.r.lc.a', args=[400, 500], kwargs=_opt),
                dict(name='M09.FR.r.lc.a', args=[100, 200], kwargs=_opt),
                dict(name='M12.FR.r.lc.a', args=[500, 600], kwargs=_opt),
                ]
    nr = len(sims)
    nc = 1

    if figsize is None:
        figsize = np.array([4, 7])
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)
    fig = plt.figure(**_fopt)
    gs = mpl.gridspec.GridSpec(nr, nc, top=.99, bottom=.06, left=.15, right=.97,
                               hspace=0)
    ax = None
    for ns, s in enumerate(sims):
        ax = plt.subplot(gs[ns], sharex=ax)
        fd = BLsim(s['name']).load_flux_data()
        fd.mdot_split(*s['args'], ax=ax, legend=not ns, **s['kwargs'])
        lbl = fd.time_slice(*s['args'])['title']
        tmp = lbl.split(' ')
        lbl = tmp[0] + '\n' + ' '.join(tmp[1:]).replace('.0', '')
        lbl = chr(ord('a') + ns) + ') ' + lbl
        ax.text(.98, .94, lbl, c='k', transform=ax.transAxes, ha='right', va='top',
                fontsize=6)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        yl, yu = plt.ylim()
        if yu - yl > 3:
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.5))
        else:
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
        ax.tick_params(axis='both', which='both', direction='in', zorder=10)
        ax.set_axisbelow(False)
        if ns < nr - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.xlabel('')

    if fn or save:
        plt.savefig(fn)
        plt.close()
    return


def multi_vortensity_prof(sims=None, save=False, figsize=None, dpi=300, fopt=None,
                          fn=None, sdir=None, lbl=True, lnorm=True, overwrite=True,
                          lopt=None):
    if save or fn:
        save = True
        if not fn:
            fn = 'multi_vortensity_prof.pdf'
        if sdir:
            sdir = os.path.expanduser(sdir)
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            fn = os.path.join(sdir, fn)
    if parse_not_overwrite(overwrite, fn):
        return None
    if lopt is None:
        lopt = dict(handlelength=1, fontsize=7, handletextpad=.4, columnspacing=.7,
                    ncol=2, loc=7)
    if sims is None:
        F = False
        sims = [
            dict(name='M06.HR.r.lc.a', kwargs=dict(rmax=3, ylim=[0,10.5])),
            dict(name='M09.FR.r.lc.a', kwargs=dict(rmax=1.8, ylim=[0,13.5], legend=F)),
            dict(name='M12.FR.r.lc.a', kwargs=dict(rmax=1.8, ylim=[0,2.1], legend=F)),
            dict(name='M15.FR.r.a', kwargs=dict(rmax=1.5, ylim=[0,3], legend=F)),
        ]
    nr = len(sims)
    nc = 1

    if figsize is None:
        figsize = np.array([3, 7.5])
    _fopt = dict(dpi=dpi, figsize=figsize)
    if fopt is None:
        fopt = {}
    _fopt.update(fopt)
    fig = plt.figure(**_fopt)
    gs = mpl.gridspec.GridSpec(nr, nc, top=.99, bottom=.05, left=.16, right=.97,
                               hspace=.15)
    ax = None
    for ns, s in enumerate(sims):
        ax = plt.subplot(gs[ns])
        sim = BLsim(s['name'])
        sim.alt_vortensity_profiles(ax=ax, **s['kwargs'], title=False, lopt=lopt)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='both', direction='in', zorder=10)
        lbl = chr(ord('a') + ns) + ') ' + 'M{:d}'.format(int(np.round(sim.mach)))
        ax.text(.83, .97, lbl, c='k', transform=ax.transAxes, ha='left', va='top',
                fontsize=8)
        if ns < nr - 1:
            plt.xlabel('')
    if fn or save:
        plt.savefig(fn)
        plt.close()
    return
