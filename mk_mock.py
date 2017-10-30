#! /usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

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
    def __init__(self, base, t, modes=[0,1], phases=0, speeds=.8, noise=None):
        self.t = t
        self.modes = np.atleast_1d(modes)
        one = np.ones(self.modes.shape)
        self.phases = phases * one
        self.speeds = speeds * one
        self.noise = noise
        self.__dict__.update(base.__dict__)
        self.patern = self._gen_patern()

    def _gen_patern(self):
        out = 0
        for i in xrange(self.modes.size):
            out += self.modes[i] * self.unitwave(i, self.phases[i] + self.t * self.speeds[i])
        if self.noise:
            out += self.noise * numpy.random.randn(self.nphi, self.nr)
        return out

    def save(self, fn=None):
        fmt = 'mock.out1.' + '{0:05d}' + '.npy'
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
    def time_step(self, t, modes=None, phases=None, speeds=None, noise=None, save=True):
        _opt = {}
        if not modes is None:
            _opt['modes'] = modes
        if not phases is None:
            _opt['phases'] = phases
        if not speeds is None:
            _opt['speeds'] = speeds
        if not noise is None:
            _opt['noise'] = noise
        out = _time_step(self, t, **_opt)
        if save:
            out.save()
        return out

    def time_seq(self, dt=tau, tmax=None, tmin=0, opts={}):
        if tmax is None:
            tmax = tmin + 10 * dt
        t = np.arange(tmin, tmax + dt, dt)
        opts = np.resize(np.atleast_1d(opts), t.size)
        for i in xrange(t.size):
            self.time_step(t[i], **opts[i])

    #def fixed_patern(self, dt=tau, tmax=100, tmin=0, **opt):
