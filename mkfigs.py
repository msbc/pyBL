#! /usr/bin/env python

import argparse

# setup and parse command line options
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--overwrite',
                    default=False,
                    action='store_true',
                    help='Overwrite existing figs')
parser.add_argument('-f', '--figs',
                    default=[],
                    metavar='N',
                    type=str,
                    nargs='+',
                    help='What figures to make.')
parser.add_argument('-a', '--all',
                    default=False,
                    action='store_true',
                    help='Make all figs')
parser.add_argument('-d', '--diagnostics',
                    default=False,
                    action='store_true',
                    help='Make all diagnostic figs')
parser.add_argument('-g', '--git',
                    default=False,
                    action='store_true',
                    help='Git pull and exit')
args = parser.parse_args()


import warnings
warnings.filterwarnings('ignore')
sims = 'sims.txt'
import numpy as np
t0 = np.arange(6) * 100
_overwrite = args.overwrite
_all = args.all
dopt = dict(save=True, add_modes=True, add_max=1, tmark=True, ext='pdf',
        overwrite=_overwrite, title_y=0.9, suptitle=False)
_diagnostics = [4, 6, 9, 11, 18, 19]
_figs = args.figs
if _all:
    _figs.append('all')
if args.diagnostics:
    #_figs = _diagnostics
    _figs.append('diag')
print("figs:", _figs)

import gc
import os
import matplotlib as mpl
if __name__ == '__main__':
    mpl.use('agg')
try:
    import bl_reader as blc
except ImportError:
    import sys_pyBL as blc
import matplotlib.pyplot as plt

if args.git:
    import sys_pyBL as blc
    blc.git_pull()
    import sys
    sys.exit()

blc.rc['rename_lc'] = True


class SimHolder(dict):
    def __init__(self, *args, **kwargs):
        super(SimHolder, self).__init__(*args, **kwargs)
        self._dict = dict(m6='M06.HR.r.lc.a', m7='M07.FR.r.a', m9='M09.FR.r.lc.a',
                          m12='M12.FR.r.lc.a', m15='M15.FR.r.a')

    def __getitem__(self, item):
        try:
            return super(SimHolder, self).__getitem__(item)
        except KeyError:
            self[item] = blc.BLsim(self._dict.get(item, item))
            print('Loading ' + repr(self[item]) + ' from ' + self[item].path)
        return super(SimHolder, self).__getitem__(item)

_sims = SimHolder()


class BLholder(blc.DataContainer):
    def __init__(self, fn='bl_stats.npz', sim_list=sims):
        if hasattr(sim_list, 'lower'):
            if os.path.isfile(sim_list):
                with open(sim_list) as f:
                    sim_list = [l.split('#')[0].strip() for l in f.readlines()]
                sim_list = [s for s in sim_list if s]
        self.sim_list = sim_list
        super().__init__(fn=fn, allow_pickle=True)

    def _gen_data(self, key):
        if key is None:
            for sim in self.sim_list:
                if sim not in self.data:
                    self._gen_data(sim)
        else:
            print(key)
            sim = _sims[key]
            data = sim.bl_stats()
            data['omax']
            self.data[key] = data.data

    def plot_key(self, key, fn=None, diff=None, save=False, overwrite=True, ext='pdf',
                 figsize=None, dpi=300, fig=None, ax=None):
        data = self.data
        name = dict(bl=r'$\delta_{\rm bl}$', plateau=r'$\delta_{\rm plateau}$').get(key, key)
        if fn:
            save = True
        if save and fn is None:
            fn = key + '_vs_mach.' + ext.lstrip('.')

        machLR = [int(i[1:3]) for i in data if 'LR' in i]
        machFR = [int(i[1:3]) for i in data if 'FR' in i]
        machHR = [int(i[1:3]) for i in data if 'HR' in i]
        if diff is None:
            if np.array(self[self.keys()[0]][key]).size > 1:
                diff = True
        if diff:
            yLR = [np.diff(data[i][key]) for i in data if 'LR' in i]
            yFR = [np.diff(data[i][key]) for i in data if 'FR' in i]
            yHR = [np.diff(data[i][key]) for i in data if 'HR' in i]
        else:
            yLR = [data[i][key] for i in data if 'LR' in i]
            yFR = [data[i][key] for i in data if 'FR' in i]
            yHR = [data[i][key] for i in data if 'HR' in i]

        if fig is None and ax is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        if ax:
            plt.sca(ax)
        plt.scatter(machLR, yLR, label='LR')
        plt.scatter(machFR, yFR, label='FR')
        plt.scatter(machHR, yHR, label='HR')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.xlabel('$\mathcal{M}$')
        plt.ylabel(name)
        if save:
            plt.savefig(fn)

bl_stats = BLholder()

class FigMaker:
    aliases = {}

    def __init__(self):
        return

    def _sanitize(self, key):
        for i in '-. ':
            key = key.replace(i, '_')
        return key

    def __call__(self, plots):
        plots = list(np.atleast_1d(plots))
        if 'all' in plots:
            plots.remove('all')
            plots += self._all_plots
        if 'diag' in plots:
            plots.remove('diag')
            plots += self._all_diag
        for plot in plots:
            attr = self._sanitize(plot)
            if not hasattr(self, attr):
                if plot in self.aliases:
                    attr = self.aliases[plot]
                elif attr in self.aliases:
                    attr = self.aliases[attr]
            print('plotting', attr)
            getattr(self, attr)()
            gc.collect()

    @property
    def _all_plots(self):
        return [i for i in dir(self) if i[0] != '_' and callable(getattr(self, i))]

    @property
    def _all_diag(self):
        return [i for i in self._all_plots if i.endswith('_diag')]


class Paper1(FigMaker):
    def sonic_examples(self):
        blc.multi_stripe(save=True, overwrite=_overwrite)

    def M09_FR_r_a(self):
        _sims['m9'].map_stripe2(np.arange(50, 601, 50), save=True, rlim=[-1.45, 1.45],
                                overwrite=_overwrite, sdir='./', llim=[-9.5, 9.5])

    def M09_FR_r_a_diag(self):
        _sims['m9'].compact_diag(**dopt)

    def M09_FR_r_a_evo_prof(self):
        _sims['m9'].evo_prof(var_list=['omega', 'dens'], cb=True, save=True,
                            overwrite=_overwrite)

    def M06_HR_r_a(self):
        _sims['m6'].map_stripe2(t0 + 25, save=True, sdir='./', overwrite=_overwrite)

    def M06_HR_r_a_diag(self):
        _sims['m6'].compact_diag(**dopt)

    def M12_FR_r_a(self):
        _sims['m12'].map_stripe2(t0 + 50, save=True, sdir='./', llim=[-1.4, 1.4],
                                 overwrite=_overwrite)

    def M12_FR_r_a_diag(self):
        _sims['m12'].compact_diag(**dopt)

    def M15_FR_r_a(self):
        _sims['m15'].map_stripe2(t0 + 75, save=True, sdir='./', rlim=[-4.9, 4.9],
                                 overwrite=_overwrite, lxlim=[.95, 1.29])

    def M15_FR_r_a_diag(self):
        _sims['m15'].compact_diag(**dopt)

    def vortex_types(self):
        blc.vortex_types(save=True)

    def one_armed(self):
        with blc.BLsim('M12.FR.random.a').loadfile('cons', 400) as df:
            gsopt = dict(left=.08, right=.92, top=.95, bottom=.08)
            fig, ax = plt.subplots(figsize=(6.3, 4.9), dpi=300, gridspec_kw=gsopt)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(axis='both', which='both', direction='in', zorder=10)
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
            df.plot2d(fn='M12_spiral.png', overwrite=_overwrite, minmax=False, rp=1.15,
                      phi0=.8 * np.pi, axis_labels=True, fig=fig, ax=ax)

    def vort_profiles(self):
        blc.multi_vortensity_prof(save=True, overwrite=_overwrite)

    def old_multi_dispersion(self):
        blc.multi_dispersion(save=True, overwrite=_overwrite)

    def gen_dispersion_data(self):
        blc.gen_dispersion_data(overwrite=_overwrite)

    def multi_dispersion(self):
        blc.plot_dispersion_data(save=True, overwrite=_overwrite, skip_gen=True)

    def res_modes(self, q=2):
        from matplotlib.colors import ListedColormap

        mach = np.array([5.0, 6.0, 6.0, 8.0, 8.0, 9.0, 9.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                         14.0, 9.0, 9.0])
        m = np.array([2.0, 5.0, 2.0, 4.0, 3.0, 4.0, 6.0, 5.0, 7.0, 6.0, 9.0, 7.0, 7.0,
                      7.0, 3.0])
        omega = np.array([0.11, 0.36, 0.15, 0.35, 0.21, 0.42, 0.49, 0.47, 0.56, 0.54,
                          0.62, 0.6, 0.61, 0.53, 0.34])
        nc = int(np.round(mach.max() - mach.min() + 1))
        cm = plt.cm.get_cmap('plasma', nc)
        colors = cm(np.linspace(0, 1, nc))
        cm = plt.cm.get_cmap('viridis', nc)
        colors[::2] = cm(np.linspace(0, 1, nc))[::2]
        cmap = ListedColormap(colors)
        fig = plt.figure(dpi=300, figsize=(5, 3.5))
        plt.scatter(mach * m, omega, c=mach, norm=None, cmap=cmap, vmin=mach.min()-.5,
                    vmax=mach.max()+.5)
        ax = plt.gca()
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='both', direction='in')
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.025))
        xlim = plt.xlim()
        ylim = plt.ylim()
        y = np.linspace(0, 1, 400)
        x = q * np.pi / (2 + y - 3 * y**(1./3.))
        plt.plot(x, y, lw=1, zorder=-1)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.xlabel(r'$m\mathcal{M}$')
        plt.ylabel(r'$\Omega_{\rm p}$')
        cb = plt.colorbar(pad=0)
        cb.ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        cb.set_label(r'$\mathcal{M}$')
        plt.savefig('res_modes.pdf')
        plt.close()

    # paper 2
class Paper2(FigMaker):
    def multi_st(self):
        blc.multi_st(save=True, overwrite=_overwrite)

    def am_plot_data(self, overwrite=False):
        _sims['m7'].am_plot_data(100, 200, overwrite=overwrite)
        _sims['m7'].am_plot_data(200, 300, overwrite=overwrite)
        _sims['m7'].am_plot_data(550, 600, overwrite=overwrite)
        _sims['m9'].am_plot_data(100, 200, overwrite=overwrite)
        _sims['m9'].am_plot_data(250, 350, overwrite=overwrite)
        _sims['m9'].am_plot_data(400, 500, overwrite=overwrite)
        _sims['M12.FR.mix.a'].am_plot_data(100, 200, overwrite=overwrite)
        _sims['M12.FR.mix.a'].am_plot_data(250, 350, overwrite=overwrite)
        _sims['M12.FR.mix.a'].am_plot_data(500, 600, overwrite=overwrite)
        _sims['m15'].am_plot_data(100, 200, overwrite=overwrite)
        _sims['m15'].am_plot_data(300, 400, overwrite=overwrite)
        _sims['m15'].am_plot_data(500, 600, overwrite=overwrite)


print('Done with initialization')

#Paper1()(_figs)
Paper2()(_figs)

# old figs
if 0:
    # fig 1
    if _all or 1 in _figs:
        blc.multi_stripe(save=True, overwrite=_overwrite)
        gc.collect()

    # fig 2
    if _all or 2 in _figs:
        _sims['m9'].thumbnails(save=True, overwrite=_overwrite)
        gc.collect()

    # fig 3
    if _all or 3 in _figs:
        _sims['m9'].thumbnails(save=True, stripes=True, overwrite=_overwrite)
        gc.collect()

    # fig 4
    if _all or 4 in _figs:
        _sims['m9'].compact_diag(**dopt)
        gc.collect()

    # fig 5
    if _all or 5 in _figs:
        _sims['m6'].map_stripe2(t0 + 25, save=True, sdir='./', overwrite=_overwrite)
        gc.collect()

    # fig 6
    if _all or 6 in _figs:
        _sims['m6'].compact_diag(**dopt)
        gc.collect()

    # fig 7
    if _all or 7 in _figs:
        blc.multi_dispersion(save=True, overwrite=_overwrite)
        gc.collect()

    # fig 8
    if _all or 8 in _figs:
        _sims['m12'].map_stripe2(t0 + 50, save=True, sdir='./', llim=[-1.4, 1.4],
                                 overwrite=_overwrite)
        gc.collect()

    # fig 9
    if _all or 9 in _figs:
        _sims['m12'].compact_diag(**dopt)
        gc.collect()

    # fig 10
    if _all or 10 in _figs:
        _sims['m15'].map_stripe2(t0 + 75, save=True, sdir='./', rlim=[-4.9, 4.9],
                                 overwrite=_overwrite, lxlim=[.95, 1.29])
        gc.collect()

    # fig 11
    if _all or 11 in _figs:
        _sims['m15'].compact_diag(**dopt)
        gc.collect()

    # fig 12
    if _all or 12 in _figs:
        print(12)
        blc.multi_omega(save=True, use_maps=True, overwrite=_overwrite)
        gc.collect()

    # fig 13
    if _all or 13 in _figs:
        blc.multi_mdot_split(save=True, overwrite=_overwrite)
        gc.collect()

    # fig 14
    if _all or 14 in _figs:
        blc.AM_plot(save=True, overwrite=_overwrite)
        gc.collect()

    # fig 15
    if _all or 15 in _figs:
        blc.multi_st(save=True, overwrite=_overwrite)
        gc.collect()

    # fig 16
    if _all or 16 in _figs:
        _opt = dict(lxlim=[.95, 1.35], llim=[-1.4, 1.4], rlim=[-4.3,4.3], dv=0, lrat=.5, sdir='./', save=True)
        blc.BLsim('M12.FR.mix.lc.a').map_stripe2([75, 175, 300, 326, 398, 499], overwrite=_overwrite, **_opt)
        #blc.BLsim('M10.FR.r.a').vortex_evo([225, 275, 301, 351, 402, 427, 452], save=True, mlim=[-14,14],
         #                                  overwrite=_overwrite)
                                           #overwrite=True)
        gc.collect()

    # fig 17
    if _all or 17 in _figs:
        blc.mode_hist(sims=sims, overwrite=_overwrite, save=True)
        gc.collect()

    # fig 18
    if _all or 18 in _figs:
        blc.BLsim('M09.FR.mix.a').compact_diag(**dopt)
        gc.collect()

    # fig 19
    if _all or 19 in _figs:
        blc.BLsim('M09.FR.mix.b').compact_diag(**dopt)
        gc.collect()

    # fig 20
    if _all or 20 in _figs:
        blc.CS_both(save=True, data='paper_cs_eff.npz', sims=sims, overwrite=_overwrite)
        gc.collect()

    # fig 21
    if _all or 21 in _figs:
        bl_stats.plot_key('bl', save=True, overwrite=_overwrite)
        gc.collect()

    # fig 22
    if _all or 22 in _figs:
        bl_stats.plot_key('plateau', save=True, overwrite=_overwrite)
        gc.collect()

    # fig 23
    if _all or 23 in _figs:
        bl_stats.plot_key('d1omega', save=True, overwrite=_overwrite)
        gc.collect()

    # fig 24
    if _all or 24 in _figs:
        blc.multi_vortensity_prof(save=True, overwrite=_overwrite)
        gc.collect()

    # fig 25
    if _all or 25 in _figs:
        _sims['m9'].map_stripe2(np.arange(50, 601, 50), save=True, rlim=[-1.45, 1.45],
                                overwrite=_overwrite, sdir='./', llim=[-9.5, 9.5])
        gc.collect()

    # fig 26
    if _all or 26 in _figs:
        _sims['m9'].evo_prof(var_list=['omega', 'dens'], cb=True, save=True,
                             overwrite=_overwrite)
        gc.collect()


    # fig 27
    if _all or 27 in _figs:
        with blc.BLsim('M12.FR.random.a').loadfile('cons', 400) as df:
            df.plot2d(fn='M12_spiral.png', overwrite=_overwrite, minmax=False, rp=1.15, phi0=.8*np.pi)
        gc.collect()

