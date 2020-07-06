#! /usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from string import ascii_lowercase as abc
import os
from glob import glob

tau = np.pi * 2

_r0 = {5: .82, 6: .85, 7: .86, 8: .88, 9: .89, 10: .93, 11: .87, 12: .92, 13: .86, 14:.85,  15: .85}
_cmap = mpl.cm.get_cmap()
_c0 = plt.rcParams['axes.prop_cycle'].by_key()['color'] + [_cmap(i) for i in np.linspace(.1,1,10)]

def predict(M, m, r0=.9):
    return np.sqrt(M**-2 + (M / (2 * m * r0))**2) / r0

def mopt(name, M=None):
    tmp = name.split('.')
    if M is None:
        M = int(tmp[0][1:])
    res = tmp[1]
    seed = tmp[-2]
    suf = tmp[-1]
    suf = abc.index(suf)
    if res == '2048':
        res = 'LR'
        suf = 3
    if seed == 'random':
      suf += 2

    ec = {'r': 'k', 'mix': 'k', 'random': 'k', 'prime': 'k'}
    markers = {'LR': ['o', 'o', 'o', 'o'], 'FR': ['o', 'o', 'o'], 'HR': ['o', 'o', 'o', 'o']}

    return {'marker': markers[res][suf], 'edgecolors': ec[seed]}

def consolidate(dir='./', sufix='.csv', of='bl.csv'):
    tables = []
    files = glob(os.path.join(dir, 'M*'))
    n = len(sufix)
    while files:
        f = files.pop(0)
        if os.path.isdir(f):
            files.extend(glob(os.path.join(f, 'M*')))
        if f[-n:] == sufix:
            tables.append(f)

    out = ['mode number, Omega_p, Mach, Time/orb observed, Simulation']
    for fn in sorted(tables):
        sim = os.path.split(fn)[-1].split('_')[0]
        mach = int(sim.split('.')[0][1:])
        with open(fn, 'r') as f:
            text = f.read()
        text = text.split('# Global Modes')[-1].strip()
        for line in filter(None, text.split('\n')):
            tmp = line.split(',')
            out.append('{0:02d}, {1:}, {2:02d}, {3:03d}, {4:}'.format(int(tmp[0]), tmp[3].strip(), mach, int(float(tmp[1]) / tau + .5), sim))

    with open(of, 'w') as f:
        f.write('\n'.join(out))


def plot_disp(csv='tmp.csv', save=False, compile=None):
    if compile is None:
        compile = (os.path.split(os.getcwd())[-1] != 'paper1')
    if compile:
        consolidate()
    data = np.genfromtxt(csv, dtype=None, names=True, delimiter=',', autostrip=True)
    machs = sorted(set(data['Mach']))
    lbls = [r'$\mathcal{M}' + r'\!=\!{:d}, r_0\!=\!{:0.2g}$'.format(M, _r0.get(M, .85)) for M in machs]
    head = []
    colors = {}
    lopt = {'loc': 3, 'frameon': True, 'markerscale': 1, 'prop': {'size':8}, 'ncol': 2, 'handletextpad': .1,
            'scatteryoffsets': [.75]}
    fig = plt.figure()
    cindex = 0

    if 0:
        for M in machs:
            loc = np.where(data['Mach'] == M)
            paths = plt.scatter(data['mode_number'][loc], data['Omega_p'][loc])
            colors[M] = paths.get_facecolor()[0]
            head.append(paths)
    else:
        for M in machs:
            loc = np.where(data['Mach'] == M)
            add_path = True
            for d in data[loc]:
                #print(d)
                opt = {}
                try:
                    opt['c'] = colors[M]
                except KeyError:
                    opt['c'] = _c0[cindex]
                    cindex += 1
                    colors[M] = opt['c']
                opt.update(mopt(d['Simulation']))
                #print(opt)
                paths = plt.scatter(d['mode_number'], d['Omega_p'], **opt)
                #colors[M] = paths.get_facecolor()[0]

                if add_path:
                    if opt['marker'] == 'o':
                        head.append(paths)
                        add_path = False
    plt.axes().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    plt.legend(head, lbls, **lopt)
    xlim = plt.xlim()
    ylim = plt.ylim()
    ylim = plt.ylim(min(.1, ylim[0]),None)
    ms = np.linspace(xlim[0], xlim[1], 200)
    for M in machs:
        plt.plot(ms, predict(float(M), ms, _r0[M]), lw=1, c=colors[M], zorder=-M)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('$m$')
    plt.ylabel(r'$\Omega_{\rm p}$')
    ylim = plt.ylim()
    plt.ylim(max(ylim[0], 0), min(ylim[1], 1))
    if save:
        fig.savefig('dispersion.pdf')
        plt.close()


if __name__ == '__main__':
    plot_disp(save=True, compile=False)
