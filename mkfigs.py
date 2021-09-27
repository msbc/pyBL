#! /usr/bin/env python

if __name__ == "__main__":
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
    parser.add_argument('-p', '--paper',
                        type=int,
                        default=2,
                        help='Choose paper number to make figs for ')
    args = parser.parse_args()


import warnings
warnings.filterwarnings('ignore')
sims = 'sims.txt'
import numpy as np
t0 = np.arange(6) * 100
_overwrite = True
_figs = ["all"]

if __name__ == "__main__":
    _overwrite = args.overwrite
    _figs = args.figs
    if args.all:
        _figs.append('all')
    if args.diagnostics:
        _figs.append('diag')
    print("figs:", _figs)

dopt = dict(save=True, add_modes=True, add_max=1, tmark=True, ext='pdf',
        overwrite=_overwrite, title_y=0.9, suptitle=False)

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
    def __init__(self, fn=None, sim_list=sims):
        if fn is None:
            fn = [i + 'bl_stats.npz' for i in ['', '../']]
            fn = [i for i in fn if os.path.isfile(i)][0]
        if hasattr(sim_list, 'lower'):
            if not os.path.isfile(sim_list):
                sim_list = '../' + sim_list
            assert os.path.isfile(sim_list)
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
                 figsize=None, dpi=300, fig=None, ax=None, legend=True, log=True,
                 pl=None, lbl=None):
        data = self.data
        name = dict(bl=r'$\delta_{\rm bl}\,\left[R_\star\right]$',
                    plateau=r'$\delta_{\rm plateau}\,\left[R_\star\right]$',
                    d1omega=r'$\Omega_K-\Omega\,\left[{\rm cycles}/{\rm orbit}\right]$'
                    ).get(key, key)
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
        if legend:
            plt.legend(loc='lower left')
        if pl is not None:
            xlim = np.array(plt.xlim())
            ylim = np.array(plt.ylim())
            a, b = pl
            if a is None:
                if log:
                    a = np.sqrt(ylim.prod()) / np.sqrt(np.prod(xlim**b))
                else:
                    a = np.mean(ylim) / np.mean(xlim**b)
                print("coef", a)
            x = np.linspace(xlim[0], xlim[1], 100)
            y = a * x**b
            plt.plot(x, y, c='.5', lw=1, ls=':', zorder=-1)
            plt.xlim(*xlim)
            plt.ylim(*ylim)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='both', direction='in')
        if log:
            plt.yscale('log')
            plt.xscale('log')
        if lbl:
            lbl = lbl + ')'
            topt = dict(c='k', ha='right', va='top', fontsize=8)
            ax.text(.98, .94, lbl, transform=ax.transAxes, **topt)
        plt.xlabel('$\mathcal{M}$')
        plt.ylabel(name)
        if save:
            plt.savefig(fn)


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
            try:
                getattr(self, attr)()
            except Exception as e:
                print('!!!!  Plotting {} failed!  !!!!'.format(attr))
                print(e)
                import traceback
                traceback.print_tb(e.__traceback__)
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
        blc.plot_dispersion_data(save=True, overwrite=_overwrite, skip_gen=True, T=True)

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

    def _am_plot_data(self, overwrite=False):
    #def am_plot_data(self, overwrite=True):
        _sims['m6'].am_plot_data(100, 200, overwrite=overwrite)
        _sims['m6'].am_plot_data(200, 300, overwrite=overwrite)
        _sims['m6'].am_plot_data(550, 600, overwrite=overwrite)
        _sims['m7'].am_plot_data(100, 200, overwrite=overwrite)
        _sims['m7'].am_plot_data(200, 300, overwrite=overwrite)
        _sims['m7'].am_plot_data(550, 600, overwrite=overwrite)
        _sims['m9'].am_plot_data(100, 200, overwrite=overwrite)
        _sims['m9'].am_plot_data(250, 350, overwrite=overwrite)
        _sims['m9'].am_plot_data(400, 500, overwrite=overwrite)
        _sims['M12.FR.mix.lc.a'].am_plot_data(100, 200, overwrite=overwrite)
        _sims['M12.FR.mix.lc.a'].am_plot_data(250, 350, overwrite=overwrite)
        _sims['M12.FR.mix.lc.a'].am_plot_data(500, 600, overwrite=overwrite)
        _sims['m15'].am_plot_data(100, 200, overwrite=overwrite)
        _sims['m15'].am_plot_data(300, 400, overwrite=overwrite)
        _sims['m15'].am_plot_data(500, 600, overwrite=overwrite)

    def am_subpanels_1(self):
        blc.am_subpanels(1, save=True, overwrite=_overwrite)

    def am_subpanels_2(self):
        blc.am_subpanels(2, save=True, overwrite=_overwrite)

    def am_terms(self):
        #self._am_plot_data()
        blc.am_terms(save=True, overwrite=_overwrite)

    def mdot_cs(self):
        blc.Mdot_CS(save=True, overwrite=_overwrite)

    def bl_properties(self):
        fn = 'bl_properties.pdf'
        if not _overwrite and os.path.isfile(fn):
            return
        bl_stats = BLholder()
        plt.figure(figsize=[4, 6], dpi=300)
        gsopt = dict(top=.95, bottom=.1, left=.2, right=.98, hspace=0, wspace=0.1)
        gs = mpl.gridspec.GridSpec(3, 1, **gsopt)
        for i, var in enumerate(['bl', 'd1omega', 'plateau']):
            ax = plt.subplot(gs[i])
            opt = dict(ax=ax, save=False, overwrite=_overwrite, legend=i==2,
                       lbl='abc'[i], pl=[(None, -2), (None, -1), (None, -2./3.)][i])
            bl_stats.plot_key(var, **opt)
            if i != 2:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.xlabel('')
        plt.savefig(fn)
        plt.close()

    def CS_both(self):
        blc.CS_both(save=True)

    def mode_hist(self):
        import os
        import cmocean
        from matplotlib.colors import ListedColormap


        data = blc.modes.data
        fn = os.path.abspath(os.path.join(os.path.dirname(__file__), 'bl.csv'))
        #fn = os.path.expanduser(
        #    '~/Dropbox/Research/IAS/rrr/bl_shared/simulation_results/Production/bl.csv')
        gdata = np.genfromtxt(fn, skip_header=1, delimiter=', ', dtype=int)[:, (0, 2)]

        nmodes = 3
        hist = {}
        for i in data:
            mach = int(i[1:3])
            tmp = data[i]
            nsets = len(tmp) + 1
            if not mach in hist:
                hist[mach] = np.zeros((nsets, 32))
            for j in range(nsets - 1):
                for m in tmp[j][:nmodes]:
                    hist[mach][j, m] += 1
        for i in gdata:
            hist[i[1]][2, i[0]] += 1

        dx = .05
        os = 2.5 * dx
        hmax = 0
        for i in hist:
            hmax = int(max(hmax, hist[i].max()) + .5)
        print(hmax)
        names = ['Reds', 'Blues', 'Greys', 'Greens']
        names = ['cmo.solar_r', 'cmo.dense', 'cmo.matter']
        cmaps = []
        for i in range(nsets):
            cm = plt.cm.get_cmap(names[i], hmax + 1)
            colors = cm(np.linspace(.05, 1, hmax + 1))
            colors[0, :] = np.array([1, 1, 1, 0])
            cmaps.append(ListedColormap(colors))
        # cmaps = [plt.cm.get_cmap(cmaps[i], hmax + 1) for i in range(nsets)]
        ims = [None, ] * nsets
        imopt = dict(vmin=-.5, vmax=hmax + .5, interpolation='nearest')

        fig = plt.figure(figsize=(4.5, 3), dpi=300)
        gs = mpl.gridspec.GridSpec(1, nsets + 2, width_ratios=[1, .05] + [.05, ] * nsets,
                                   top=.99,
                                   bottom=.15, left=.1, right=.93, wspace=0)
        ax = plt.subplot(gs[0])
        for i in hist:
            for j in range(nsets):
                extent = [i - dx + os * (j - 1), i + dx + os * (j - 1), 0, 31]
                ims[j] = plt.imshow(np.array([hist[i][j]]).T, extent=extent,
                                    cmap=cmaps[j], **imopt)
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='both', direction='in')
        yl = plt.ylim()
        xl = plt.xlim(4.5, 15.5)
        plt.plot(xl, xl, c='.5', lw=1, ls=':', zorder=-10)
        plt.ylim(*yl)
        plt.xlim(*xl)
        plt.xlabel(r'$\mathcal{M}$')
        plt.ylabel(r'$m$')
        names = ['star', 'disk', 'global']
        for j in range(nsets):
            cax = plt.subplot(gs[2 + j])
            t = []
            if j == nsets - 1:
                t = np.arange(hmax + 1)
            cb = plt.colorbar(ims[j], cax=cax, ticks=t)
            cax.set_xlabel(names[j], fontsize=8, rotation='vertical')

        plt.savefig('mode_hist.pdf')
        plt.close()


print('Done with initialization')

if __name__ == "__main__":
    if args.paper == 1:
        Paper1()(_figs)
    elif args.paper == 2:
        Paper2()(_figs)
    else:
        print("No figures have been configured for paper number {:d}.".format(args.paper))
