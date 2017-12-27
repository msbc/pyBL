#! /usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pyBL.simBLclass as bl

tau = 2 * np.pi

def wave(m, nphi, nr=None, phase=0, amp=1):
    if nr is None:
        nr = nphi
    phi = (np.arange(nphi) + .5) * 2 * np.pi / (nphi + 1)

class _base(object):
    def __init__(self, nphi=512, nr=None, dt=tau, rmin=.9, rmax=3):
        self.nphi = nphi
        if nr is None:
            nr = nphi
        self.nr = nr
        self.dt = dt
        self.phi = np.arange(nphi + 1) * tau / nphi
        self.phic = .5 * (self.phi[:-1] + self.phi[1:])
        tmp = (rmax / rmin)**(1. / nr)
        self.r = rmin * tmp**np.arange(nr + 1)
        self.rc = .5 * (self.r[:-1] + self.r[1:])
        self.gridc = np.meshgrid(self.rc, self.phic)[::-1]
        self.grid = np.meshgrid(self.r, self.phi)[::-1]

    def unitwave(self, m, phase=0):
        return np.cos(self.gridc[0] * m - phase)

class _time_step(_base):
    def __init__(self, base, t, modes=[0,1], phases=0, speeds=.8, noise=0, dt=None):
        self.t = t
        self.dt = dt
        self.modes = np.atleast_1d(modes)
        one = np.ones(self.modes.shape)
        try:
            self.phases = phases * one
        except ValueError:
            self.phases = phases
        try:
            self.speeds = speeds * one
        except ValueError:
            self.speeds = speeds
        self.noise = noise
        self.__dict__.update(base.__dict__)
        self.patern = self._gen_patern()

    def _gen_patern(self):
        out = 0
        for i in xrange(self.modes.size):
            out += self.modes[i] * self.unitwave(i, self.phases[i] + self.t * self.speeds[i] / float(i))
        if self.noise:
            out += self.noise * np.random.randn(self.nphi, self.nr)
        return out

    def save(self, fn=None):
        fmt = 'mock.out1.' + '{0:05d}' + '.npy'
        if fn == 0:
            tmp = np.array([np.arange(self.modes.size), self.modes, self.speeds])
            head = 'noise = %.2e' % self.noise
            if not self.dt is None:
                head +=', dt = %.4f' % self.dt
            head += '\nmode, amplitude, speed'
            opt = dict(header=head, fmt=['%02d', '%.3f', '%.3f'])
            np.savetxt('summary.txt', tmp.T, **opt)
        if fn is None:
            fn = 0
            while os.path.isfile(fmt.format(fn)):
                fn += 1
            fn = fmt.format(fn)
        try:
            if fn == int(fn):
                fn = fmt.format(fn)
        except ValueError:
            pass
        one = np.ones((1,) + self.patern.shape)
        data = {i: one.copy() for i in ['dens', 'mom1', 'mom2', 'mom3']}
        data['mom1'] *= self.patern
        data['x1f'] = self.r
        data['x2f'] = self.phi
        data.update({i: getattr(self, i) for i in ['t', 'modes', 'speeds']})
        np.save(fn, data)

    def save_as_athdf(self, fn=None):
        if fn is None:
            fn = 'time-'
        with h5py.File(fn, 'w') as f:
            f.attrs['Coordinates'] = 'cylindrical'
            f.attrs['MeshBlockSize'] = [self.nr, self.nphi, 1]
            f.attrs['RootGridSize'] = [self.nr, self.nphi, 1]
            f.attrs['MaxLevel'] = 0
            f.attrs['VariableNames'] = ['dens', 'mom1', 'mom2', 'mom3']
            f.attrs['NumMeshBlocks'] = 1
            f.create_dataset('Levels', [0], dtype='i')
            f.create_dataset('LogicalLocations', [[[0,0,0]]], dtype='i')

class gen_mock(_base):
    def time_step(self, t, modes=None, phases=None, speeds=None, noise=None, save=True, fn=None, dt=None):
        _opt = {}
        if not modes is None:
            _opt['modes'] = modes
        if not phases is None:
            _opt['phases'] = phases
        if not speeds is None:
            _opt['speeds'] = speeds
        if not noise is None:
            _opt['noise'] = noise
        if not dt is None:
            _opt['dt'] = dt
        out = _time_step(self, t, **_opt)
        if save:
            out.save(fn)
        return out

    def time_seq(self, dt=None, tmax=None, tmin=0, opts={}):
        if dt is None:
            dt = self.dt
        if tmax is None:
            tmax = tmin + 10 * dt
        print dt
        t = np.arange(tmin, tmax + dt, dt)
        opts = np.resize(np.atleast_1d(opts), t.size)
        for i in xrange(t.size):
            self.time_step(t[i], fn=i, dt=dt, **opts[i])

    def spiral(self, dphidr=1):
        return (self.gridc[1] - 1) * dphidr

    #def fixed_patern(self, dt=tau, tmax=100, tmin=0, **opt):

class mock_sim(bl.BLsim):
    def cross_corr(self, t1, t2=None, smooth=4, save=False, ext='png', **kwargs):
        if not kwargs.get('plot', False):
            return super(mock_sim, self).cross_corr(t1, t2, **kwargs)
        if t2 is None:
            t2 = t1 + 1
        t1 = self.loadfile(t1)
        t2 = self.loadfile(t2)
        dt = t2.data['t'] - t1.data['t']
        out = super(mock_sim, self).cross_corr(t1, t2, **kwargs)
        th = self.prop_speed(t1, t2, cc=out, smooth=smooth) * dt
        r = self.rc
        plt.plot(r * np.cos(th), r * np.sin(th), 'w', lw=1)
        r = self.r
        one = np.ones(r.shape)
        for i in xrange(t1.data['modes'].size):
            if t1.data['modes'][i]:
                th = dt * t1.data['speeds'][i] * one
                plt.plot(r * np.cos(th), r * np.sin(th), 'k:', lw=1)
        if save:
            plt.savefig(self.name + '_cc_{:d}-{:d}.'.format(t1.t, t2.t) + ext)
            plt.close()
        return out

    def r_phase(self, r):
        modes = super(mock_sim, self).r_phase(r, ret_m=True)
        bf = self.loadfile(1)
        speeds = bf.data['speeds']
        ti = self.times
        dt = self.dt
        t = dt * ti
        yl = plt.ylim()
        for m in modes:
            plt.plot(ti, t * speeds[m], 'k:', lw=1)
        plt.ylim(*yl)
        plt.xlabel('Time / $%.4f$' % dt)

    def r_speed(self, r):
        modes = super(mock_sim, self).r_speed(r, ret_m=True)
        bf = self.loadfile(1)
        speeds = bf.data['speeds']
        ti = self.times
        dt = self.dt
        t = dt * ti
        yl = plt.ylim()
        plt.gca().set_color_cycle(None)
        cs = plt.rcParams['axes.prop_cycle']()
        for m in modes:
            try:
                plt.axhline(speeds[m], ls=':', lw=1, **cs.next())
            except IndexError:
                pass
        plt.ylim(*yl)
        plt.xlabel('Time / $%.4f$' % dt)
