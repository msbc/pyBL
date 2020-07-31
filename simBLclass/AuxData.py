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
from scipy.interpolate import CubicSpline
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
from ..helpers import grad, atleast_4d
from ..parmap import parmap
from ._local_helpers import *
from .defaults import rc
from .BLDataFiles import BLFT

class _FileBuffer(object):
    def __init__(self, filenames, ghost=1, sim=None, var='FT'):
        self.filenames = filenames
        self.len = len(filenames)
        self.size = 2 * ghost + 1
        self.sim = sim
        self.ghost = ghost
        self.index = None
        self._var = var
        self._phase = None
        self._amp = None
        self._speed = None
        self._t = None
        self._offset = range(-ghost, ghost + 1)
        self.set_index(0)

    def _load(self, f):
        try:
            if int(f) == f:
                f = self.filenames[f]
        except ValueError:
            pass
        # print(f)
        bf = BLFT(f, sim=self.sim)
        return [bf.t, np.abs(bf[self._var]), np.angle(bf[self._var])]

    def _cap_index(self, i):
        return min(max(i, 0), self.len - 1)

    def set_index(self, i):
        i %= self.len
        self._t, self._amp, self._phase = zip(
            *[self._load(self._cap_index(i + j)) for j in self._offset])
        self._t = list(self._t)
        self._amp = list(self._amp)
        self._phase = list(self._phase)
        self._speed = [None] * self.size
        self._comp()
        self.index = i

    def _set(self, i):
        i %= self.len
        i0 = i - self.ghost
        tmp = self._load(self._cap_index(i0))
        self._t = [tmp[0]] * self.size
        self._amp = [tmp[1]] * self.size
        self._phase = [tmp[2]] * self.size
        self.index = i0
        while self.index < i:
            self._increment()

    def _comp(self, shift=False):
        # unwrap
        phi = np.array(self._phase)
        # coarse correct
        dphi = np.diff(phi, axis=0)
        tmp = np.pad(dphi, ((1, 0), (0, 0), (0, 0)), 'constant')
        tmp = np.minimum(np.floor(tmp / tau), 0)
        phi -= tmp.cumsum(axis=0) * tau
        del tmp
        # fine correct
        dphi = np.diff(phi, axis=0)
        t = np.array(self._t)
        dt = np.diff(t)
        fdphi = np.pad(dphi, ((0, 1), (0, 0), (0, 0)), 'constant')
        bdphi = np.pad(dphi, ((1, 0), (0, 0), (0, 0)), 'constant')
        fdt = np.pad(dt, (0, 1), 'constant')[:, np.newaxis, np.newaxis]
        bdt = np.pad(dt, (1, 0), 'constant')[:, np.newaxis, np.newaxis]
        shift = np.rint(np.maximum(fdphi * bdt / fdt - bdphi, 0) / tau)
        bshift = np.rint(
            np.roll(np.maximum(bdphi * fdt / bdt - fdphi, 0) / tau, 1, axis=0))
        shift[shift != bshift] = 0
        shift *= tau
        phi += shift.cumsum(axis=0)
        self._phase = list(phi)

        # speed
        self._speed = list(grad(t, phi))

    def _increment(self):
        if self.index >= self.len - 1:
            raise IndexError('Cannot increment past last time-step.')
        tmp = self._load(self.index + self.ghost + 1)
        self._t.append(tmp.pop(0))
        self._amp.append(tmp.pop(0))
        self._phase.append(tmp.pop(0))
        # coarse unwrap
        fdphi = self._phase[-1] - self._phase[-2]
        self._phase[-1] -= tau * np.minimum(np.floor(fdphi / tau), 0)
        # fine unwrap
        fdphi = self._phase[-1] - self._phase[-2]
        bdphi = self._phase[-2] - self._phase[-3]
        fdt = self._t[-1] - self._t[-2]
        bdt = self._t[-2] - self._t[-3]
        shift = np.rint(np.maximum(fdphi * bdt / fdt - bdphi, 0) / tau)
        bshift = np.rint(
            np.roll(np.maximum(bdphi * fdt / bdt - fdphi, 0) / tau, 1, axis=0))
        shift[shift != bshift] = 0
        shift *= tau
        self._phase[-2] += shift
        self._phase[-1] += shift

        # speed
        self._speed.append(None)
        dl = self._t[-2] - self._t[-3]
        dr = self._t[-1] - self._t[-2]
        Dinv = 1. / (dl + dr)
        rat = dr / dl
        self._speed[-2] = (dl ** -1 - dr ** -1) * self._phase[-2] \
                          + Dinv / rat * self._phase[-1] \
                          - rat * Dinv * self._phase[-3]

        # update index and arrays
        self._phase.pop(0)
        self._t.pop(0)
        self._amp.pop(0)
        self._speed.pop(0)
        self.index += 1

    def increment(self):
        try:
            self._increment()
            return True
        except IndexError:
            return False

    @property
    def t(self):
        return self._t[self.ghost]

    @property
    def amp(self):
        return self._amp[self.ghost]

    @property
    def phase(self):
        return self._phase[self.ghost]

    @property
    def speed(self):
        return self._speed[self.ghost]

    @property
    def state(self):
        return (self.t, self.amp, self.phase, self.speed)

    def mean(self, data):
        return np.array([i for i in data if i is not None]).mean(axis=0)

    def std(self, data):
        return np.std(np.array([i for i in data if i is not None]), axis=0)


class IncrementalFFT(object):
    def __init__(self, filenames, sim=None, store_data=False, fine_out=None,
                 coarse_out=None, var='FT', quiet=None):
        self.filenames = filenames
        self._ghost = 10
        self.buffer = _FileBuffer(filenames, ghost=self._ghost, sim=sim, var=var)
        self.sim = sim
        self._store_data = store_data
        if fine_out is None:
            fine_out = os.path.join(sim.path, 'FFT_fine_' + var + '.npy')
        if coarse_out is None:
            coarse_out = os.path.join(sim.path, 'FFT_coarse_' + var + '.npy')
        self.fine_out = fine_out
        self.coarse_out = coarse_out
        self._t = None
        self._amp = None
        self._phase = None
        self._speed = None
        if quiet is None:
            quiet = rc('quiet')
        self.quiet = quiet
        self._cd = None
        self.var = var

    def process(self, store_data=None):
        mode = 'wb'
        _ft = -1
        _ct = -1
        if os.path.isfile(self.fine_out):
            _ft = self.get_last_time(self.fine_out)
            mode = 'ab'
            _ct = self.get_last_time(self.coarse_out)
            mode = 'ab'
        if store_data is None:
            store_data = self._store_data
        if store_data:
            self._t = []
            self._amp = []
            self._phase = []
            self._speed = []
            self._cd = {'t': [], 'amp': [], 'phase': [], 'speed': [], 'amp_std': [],
                        'phase_std': [], 'speed_std': []}
        test = True
        if not self.quiet:
            # print(self.coarse_out)
            print('Compiling FFT ({:}) data.'.format(self.var))
            helpers.update_progress(0)
        with open(self.fine_out, mode) as fine, open(self.coarse_out, mode) as coarse:
            while test:
                data = self.buffer.state
                if store_data:
                    self._t.append(data[0])
                    self._amp.append(data[1])
                    self._phase.append(data[2])
                    self._speed.append(data[3])
                if data[0] > _ft:
                    for i in data:
                        np.array(i).astype('float32').tofile(fine)
                name = os.path.split(self.buffer.filenames[self.buffer.index])[-1].split(
                    '.')
                if data[0] > _ct:
                    if name[1] == 'FT' and name[2][-1] == '0':
                        # print(self.buffer.index, data[0], self.buffer.filenames[self.buffer.index])
                        np.array(data[0]).astype('float32').tofile(coarse)
                        a = self.buffer._amp
                        if store_data:
                            self._cd['t'].append(data[0])
                            self._cd['amp'].append(self.buffer.mean(a))
                            self._cd['amp_std'].append(self.buffer.std(a))
                        self.buffer.mean(a).astype('float32').tofile(coarse)
                        self.buffer.std(a).astype('float32').tofile(coarse)
                        a = self.buffer._phase
                        if store_data:
                            self._cd['phase'].append(self.buffer.mean(a))
                            self._cd['phase_std'].append(self.buffer.std(a))
                        self.buffer.mean(a).astype('float32').tofile(coarse)
                        self.buffer.std(a).astype('float32').tofile(coarse)
                        a = self.buffer._speed
                        if store_data:
                            self._cd['speed'].append(self.buffer.mean(a))
                            self._cd['speed_std'].append(self.buffer.std(a))
                        self.buffer.mean(a).astype('float32').tofile(coarse)
                        self.buffer.std(a).astype('float32').tofile(coarse)
                test = self.buffer.increment()
                if not self.quiet:
                    helpers.update_progress(float(self.buffer.index) / self.buffer.len)
        helpers.update_progress(1)
        if store_data:
            for i in self._cd:
                self._cd[i] = np.array(i)

    def get_last_time(self, fn=None, nvar=None, nphi=None):
        if fn is None:
            fn = self.coarse_out
        if nvar is None:
            if fn == self.coarse_out:
                nvar = 6
            elif fn == self.fine_out:
                nvar = 3
            else:
                raise RuntimeError('"navr" cannot be determined.')
        first = BLFT(self.filenames[0], sim=self.sim)
        if nphi is None:
            nphi = self.sim.inputs['meshblock']['nx2']
        nr = first.rc.size
        del first
        with open(fn, 'rb') as f:
            offset = - 4 * (nvar * nphi * nr + 1)
            f.seek(offset, os.SEEK_END)
            t = np.fromfile(f, 'float32', 1)[0]
        return t


class FTdataFile(object):
    def __init__(self, filename, nr=None, nphi=None, sim=None, coarse=True):
        if nr is None:
            nr = sim.rc.size
        if nphi is None:
            nphi = sim.inputs['meshblock']['nx2']
        self.filename = filename
        # print(filename)
        self.nr = nr
        self.nphi = nphi
        self.modes = np.arange(nphi)[np.newaxis, :, np.newaxis]
        self.sim = sim
        self.coarse = coarse
        self._t = None
        self._amp = None
        self._phase = None
        self._speed = None
        self._amp_std = None
        self._phase_std = None
        self._speed_std = None

    def existsQ(self):
        return os.path.isfile(self.filename)

    def updateQ(self):
        if not self.existsQ():
            print('FFT ' + self.filename + ' does not exist')
            return 1
        if os.path.getsize(self.filename) == 0:
            print('FFT ' + self.filename + ' has size 0')
            return 2
        if self.sim is not None:
            tmp = [0]
            files = [i for i in glob(os.path.join(self.sim.path, '*.athdf'))]
            tmp.extend([os.path.getctime(i) for i in files])
            if max(tmp) > os.path.getmtime(self.filename):
                try:
                    loc = np.array(tmp[1:]).argmax()
                    print(loc, files[loc], tmp[loc - 1], os.path.getmtime(self.filename))
                except:
                    print("IDK:", sys.exc_info()[0], loc, len(files), len(tmp))
                return False
        return False

    def generate(self):
        self.sim.gen_fft_file()

    def _read_data(self):
        tmp = self.updateQ()
        if tmp:
            print("UpdateQ: {:}".format(tmp))
            self.generate()
        t = []
        amp = []
        phase = []
        speed = []
        nvar = 3
        if self.coarse:
            amp_std = []
            phase_std = []
            speed_std = []
            nvar += 3
        with open(self.filename, 'rb') as f:
            while True:
                try:
                    t.append(np.fromfile(f, 'float32', 1)[0])
                    data = list(
                        np.reshape(np.fromfile(f, 'float32', nvar * self.nphi * self.nr),
                                   (nvar, self.nphi, self.nr)))
                    amp.append(data.pop(0))
                    if self.coarse: amp_std.append(data.pop(0))
                    phase.append(data.pop(0))
                    if self.coarse: phase_std.append(data.pop(0))
                    speed.append(data.pop(0))
                    if self.coarse: speed_std.append(data.pop(0))
                except (EOFError, IndexError, ValueError):
                    break
        self._t = np.array(t)
        self._amp = np.array(amp)
        self._phase = np.array(phase)
        self._speed = np.array(speed) / self.modes
        if self.coarse:
            self._amp_std = np.array(amp_std)
            self._phase_std = np.array(phase_std)
            self._speed_std = np.array(speed_std) / self.modes

    @property
    def t(self):
        if self._t is None:
            self._read_data()
        return self._t

    @property
    def amp(self):
        if self._amp is None:
            self._read_data()
        return self._amp

    @property
    def amp_std(self):
        if self._amp_std is None:
            self._read_data()
        return self._amp_std

    @property
    def phase(self):
        if self._phase is None:
            self._read_data()
        return self._phase

    @property
    def speed(self):
        if self._speed is None:
            self._read_data()
        return self._speed

    @property
    def speed_std(self):
        if self._speed_std is None:
            self._read_data()
        return self._speed_std

    @property
    def FT(self):
        return self.amp * np.exp(1j * self.phase)


class IncrementalFFThdf5(object):
    def __init__(self, filenames, sim=None, store_data=False, fine_out=None,
                 coarse_out=None, var='FT', quiet=None):
        self.filenames = filenames
        self._ghost = 10
        self.buffer = _FileBuffer(filenames, ghost=self._ghost, sim=sim, var=var)
        self.sim = sim
        self._store_data = store_data
        if fine_out is None:
            fine_out = os.path.join(sim.path, 'FFT_fine_' + var + '.hdf5')
        if coarse_out is None:
            coarse_out = os.path.join(sim.path, 'FFT_coarse_' + var + '.hdf5')
        self.fine_out = fine_out
        self.coarse_out = coarse_out
        self._t = None
        self._amp = None
        self._phase = None
        self._speed = None
        if quiet is None:
            quiet = rc('quiet')
        self.quiet = quiet
        self._cd = None
        self.var = var

    def process(self, store_data=None):
        mode = 'w'
        _ft = -1
        _ct = -1
        if os.path.isfile(self.fine_out):
            _ft = self.get_last_time(self.fine_out)
            _ct = self.get_last_time(self.coarse_out)
            mode = 'a'
        if store_data is None:
            store_data = self._store_data
        if store_data:
            self._t = []
            self._amp = []
            self._phase = []
            self._speed = []
            self._cd = {'t': [], 'amp': [], 'phase': [], 'speed': [], 'amp_std': [],
                        'phase_std': [], 'speed_std': []}
        test = True
        if not self.quiet:
            # print(self.coarse_out)
            print('Compiling H5FFT ({:}) data.'.format(self.var))
            helpers.update_progress(0)
        with h5py.File(self.fine_out, mode) as fine, \
                h5py.File(self.coarse_out, mode) as coarse:
            shape = (0, self.sim.inputs['meshblock']['nx2'], self.sim.rc.size)
            mshape = (None, self.sim.inputs['meshblock']['nx2'], self.sim.rc.size)
            names = ['t', 'amp', 'phase', 'speed']
            dt = 'float32'
            try:
                # fine data sets
                fine.create_dataset('t', (0,), maxshape=(None,), dtype=dt)
                fine.create_dataset('amp', shape, maxshape=mshape, dtype=dt)
                fine.create_dataset('phase', shape, maxshape=mshape, dtype=dt)
                fine.create_dataset('speed', shape, maxshape=mshape, dtype=dt)
                # coarse data sets
                coarse.create_dataset('t', (0,), maxshape=(None,), dtype=dt)
                coarse.create_dataset('amp', shape, maxshape=mshape, dtype=dt)
                coarse.create_dataset('phase', shape, maxshape=mshape, dtype=dt)
                coarse.create_dataset('speed', shape, maxshape=mshape, dtype=dt)
                coarse.create_dataset('amp_std', shape, maxshape=mshape, dtype=dt)
                coarse.create_dataset('phase_std', shape, maxshape=mshape, dtype=dt)
                coarse.create_dataset('speed_std', shape, maxshape=mshape, dtype=dt)
            except RuntimeError:
                pass
            while test:
                data = self.buffer.state
                if store_data:
                    self._t.append(data[0])
                    self._amp.append(data[1])
                    self._phase.append(data[2])
                    self._speed.append(data[3])
                if data[0] > _ft:
                    for i, d in zip(names, data):
                        fine[i].resize(fine[i].shape[0] + 1, axis=0)
                        fine[i][fine[i].shape[0] - 1] = d
                name = os.path.split(self.buffer.filenames[self.buffer.index])[-1].split(
                    '.')
                if data[0] > _ct:
                    if name[1] == 'FT' and name[2][-1] == '0':
                        # print(self.buffer.index, data[0], self.buffer.filenames[self.buffer.index])
                        for i in names + [j + '_std' for j in names[1:]]:
                            coarse[i].resize(coarse[i].shape[0] + 1, axis=0)
                        coarse['t'][coarse['t'].shape[0] - 1] = data[0]
                        a = self.buffer._amp
                        if store_data:
                            self._cd['t'].append(data[0])
                            self._cd['amp'].append(self.buffer.mean(a))
                            self._cd['amp_std'].append(self.buffer.std(a))
                        coarse['amp'][coarse[i].shape[0] - 1] = self.buffer.mean(a)
                        coarse['amp_std'][coarse[i].shape[0] - 1] = self.buffer.std(a)
                        a = self.buffer._phase
                        if store_data:
                            self._cd['phase'].append(self.buffer.mean(a))
                            self._cd['phase_std'].append(self.buffer.std(a))
                        coarse['phase'][coarse[i].shape[0] - 1] = self.buffer.mean(a)
                        coarse['phase_std'][coarse[i].shape[0] - 1] = self.buffer.std(a)
                        a = self.buffer._speed
                        if store_data:
                            self._cd['speed'].append(self.buffer.mean(a))
                            self._cd['speed_std'].append(self.buffer.std(a))
                        coarse['speed'][coarse[i].shape[0] - 1] = self.buffer.mean(a)
                        coarse['speed_std'][coarse[i].shape[0] - 1] = self.buffer.std(a)
                test = self.buffer.increment()
                if not self.quiet:
                    helpers.update_progress(float(self.buffer.index) / self.buffer.len)
        if not self.quiet:
            helpers.update_progress(1)
        if store_data:
            for i in self._cd:
                self._cd[i] = np.array(i)

    def get_last_time(self, fn=None, nvar=None, nphi=None):
        if fn is None:
            fn = self.coarse_out
        if nvar is None:
            if fn == self.coarse_out:
                nvar = 6
            elif fn == self.fine_out:
                nvar = 3
            else:
                raise RuntimeError('"navr" cannot be determined.')
        #first = BLFT(self.filenames[0], sim=self.sim)
        if nphi is None:
            nphi = self.sim.inputs['meshblock']['nx2']
        #nr = first.rc.size
        #del first
        try:
            with h5py.File(fn, 'r') as f:
                t = float(f['t'][-1])
        except (KeyError, ValueError):
            t = -1.0
        return t


class FTdataHDF5File(object):
    def __init__(self, filename, nr=None, nphi=None, sim=None, coarse=True, var=None):
        if nr is None:
            nr = sim.rc.size
        if nphi is None:
            nphi = sim.inputs['meshblock']['nx2']
        self.filename = filename
        # print(filename)
        if var is None:
            var = os.path.split(filename)[-1].split('.')[0].split('_')[-1]
        self.var = var
        self.nr = nr
        self.nphi = nphi
        self.modes = np.arange(nphi)[np.newaxis, :, np.newaxis]
        self.sim = sim
        self.coarse = coarse
        self._file = None

    def existsQ(self):
        return os.path.isfile(self.filename)

    def updateQ(self):
        if not self.existsQ():
            print('FFT ' + self.filename + ' does not exist')
            return 1
        if os.path.getsize(self.filename) == 0:
            print('FFT ' + self.filename + ' has size 0')
            return 2
        if self.sim is not None:
            tmp = [0]
            files = [i for i in glob(os.path.join(self.sim.path, '*.athdf'))]
            tmp.extend([os.path.getctime(i) for i in files])
            if max(tmp) > os.path.getmtime(self.filename):
                try:
                    loc = np.array(tmp[1:]).argmax()
                    print(
                    loc, files[loc], tmp[loc - 1], os.path.getmtime(self.filename))
                except:
                    print("IDK:", sys.exc_info()[0], loc, len(files), len(tmp))
                return False
        return False

    def generate(self, quiet=False):
        f = IncrementalFFThdf5(self.sim.sortedFFT(), var=self.var, sim=self.sim,
                               quiet=quiet)
        f.process()

    def _read_data(self):
        tmp = self.updateQ()
        if tmp:
            print("UpdateQ: {:}".format(tmp))
            self.generate()
        self._file = h5py.File(self.filename, 'r')

    @property
    def t(self):
        if self._file is None:
            self._read_data()
        return self._file['t'][:]

    @property
    def amp(self):
        if self._file is None:
            self._read_data()
        return self._file['amp'][:]

    @property
    def amp_std(self):
        if self._file is None:
            self._read_data()
        return self._file['amp_std'][:]

    @property
    def phase(self):
        if self._file is None:
            self._read_data()
        return self._file['phase'][:]

    @property
    def speed(self):
        if self._file is None:
            self._read_data()
        return self._file['speed'][:] / self.modes

    @property
    def speed_std(self):
        if self._file is None:
            self._read_data()
        return self._file['speed_std'][:] / self.modes

    @property
    def FT(self):
        return self.amp * np.exp(1j * self.phase)


class FFTset(object):
    def __init__(self, filenames, sim=None, athinput=None, fft_data=None,
                 fft_time=None):
        self.filenames = filenames
        self.sim = sim
        self._fft_data = fft_data
        self._fft_time = fft_time

    def _load_fft_data(self):
        data = []
        t = []
        for fn in self.filenames:
            if self.sim is None:
                f = BLFT(fn)
            else:
                f = self.sim.loadfile(fn)
            data.append(f['FT'])
            t.append(f.t)
        data = np.array(data)
        t = np.array(t)
        if self._fft_data is None:
            self._fft_data = data
        if self._fft_time is None:
            self._fft_time = t
        return data

    @property
    def data(self):
        if self._fft_data is None:
            return self._load_fft_data()
        return self._fft_data

    @property
    def time(self):
        if self._fft_time is None:
            self._load_fft_data()
        return self._fft_time


class CompositeFFTSet(object):
    def __init__(self, fftsets, fft_data=None, fft_time=None, phase_angle=None,
                 phase_speed=None):
        self.fftsets = fftsets
        self._fft_time = fft_time
        self._fft_data = fft_data
        self._phase_angle = phase_angle
        self._phase_speed = phase_speed

    @property
    def data(self):
        if self._fft_data is None:
            return self._collect_fft_data()
        return self._fft_data

    @property
    def time(self):
        if self._fft_time is None:
            self._collect_fft_data()
        return self._fft_time

    def _collect_fft_data(self):
        ffts = sorted(self.fftsets, key=lambda x: x.time[1])
        nt = sum([i.time.size for i in ffts])
        shape = (nt,) + ffts[0].data.shape[1:]
        data = np.empty(shape, dtype='complex64')
        time = np.empty(nt)
        n = len(ffts)
        for i in range(n):
            data[i::n] = ffts[i].data
            time[i::n] = ffts[i].time
        while time[1] == 0:
            data = data[1:]
            time = time[1:]
        phase = self._unwrap(time, np.angle(data))
        if self._fft_data is None:
            self._fft_data = data
        if self._fft_time is None:
            self._fft_time = time
        if self._phase_angle is None:
            self._phase_angle = phase
        return time, data, phase

    def _unwrap(self, time=None, phase=None, limit=None, mNorm=False, fine_correct=False):
        if phase is None:
            phase = np.angle(self.data)
        else:
            phase = phase.copy()
        if time is None:
            time = self.time
        if limit is None:
            limit = 0
            # limit = - .1 / np.arange(phase.shape[1])[np.newaxis,:,np.newaxis]
        if mNorm:
            limit /= np.arange(phase.shape[1])[np.newaxis, :, np.newaxis]
        # coarse unwrap
        d = crudeDiff(time, phase, axis=0)
        shift = np.zeros_like(phase)
        shift[np.where(d < limit)] += tau
        # shift = np.roll(shift, -1, axis=0)
        # shift[-1] = 0
        self._shift = shift.copy()
        phase += shift.cumsum(axis=0)
        # fine-course unwrap
        if fine_correct:
            dphi = np.diff(phase, axis=0)
            fdphi = np.pad(dphi, ((0, 1), (0, 0), (0, 0)), 'constant')
            bdphi = np.pad(dphi, ((1, 0), (0, 0), (0, 0)), 'constant')
            dt = np.diff(time)
            dtmax = dt.max()
            fdt = np.pad(dt, (0, 1), 'constant')[:, np.newaxis, np.newaxis]
            bdt = np.pad(dt, (1, 0), 'constant')[:, np.newaxis, np.newaxis]
            ashift = np.maximum(fdphi * bdt / fdt - bdphi, 0) / tau
            bshift = np.maximum(bdphi * fdt / bdt - fdphi, 0) / tau
            shift = np.maximum(fdphi * bdt / fdt - bdphi + bdphi * fdt / bdt - fdphi,
                               0) * .5 / tau
            shift = np.rint(shift) * tau
            phase += shift.cumsum(axis=0)
        return phase

    def _fine_correct(self, time=None, phase=None):
        if phase is None:
            phase = self._unwrap()
        if time is None:
            time = self.time
        dphi = np.diff(phase, axis=0)
        fdphi = np.pad(dphi, ((0, 1), (0, 0), (0, 0)), 'constant')
        bdphi = np.pad(dphi, ((1, 0), (0, 0), (0, 0)), 'constant')
        dt = np.diff(time)
        dtmax = dt.max()
        fdt = np.pad(dt, (0, 1), 'constant')[:, np.newaxis, np.newaxis]
        bdt = np.pad(dt, (1, 0), 'constant')[:, np.newaxis, np.newaxis]
        shift = np.rint(np.maximum(fdphi * bdt / fdt - bdphi, 0) / tau)
        bshift = np.rint(
            np.roll(np.maximum(bdphi * fdt / bdt - fdphi, 0) / tau, 1, axis=0))
        shift[shift != bshift] = 0
        # shift = np.minimum(shift * tau, np.maximum(dtmax - fdphi, 0))
        shift *= tau
        self._fshift = shift.copy()
        return phase + shift.cumsum(axis=0)

    def phase(self):
        if self._phase_angle is None:
            self._phase_angle = self._unwrap()
        return self._phase_angle

    def prop_speed(self, phase=None):
        if self._phase_speed is None:
            if phase is None:
                phase = self._unwrap()
            m = np.arange(phase.shape[1])
            m = m[np.newaxis, :, np.newaxis]
            dphi = grad(self.time, phase)
            self._phase_speed = dphi / m
        return self._phase_speed


class npz_wrapper(object):
    def __init__(self, npz, rc):
        self.npz = npz
        self.rc = rc[None, :]

    def __getitem__(self, item):
        if item == 'mom1':
            return self.npz['Mdot']
        out = self.npz[item]
        if item == 'Mdot':
            out *= tau * self.rc
        return out


class FluxData(object):
    def __init__(self, npz, rc, sim, ns=None, option=2, smoothing=None):
        self.npz = npz
        self.rc = rc
        self.sim = sim
        self.ns = ns
        self.dopt = option
        self.dsmooth = smoothing
        self.r2 = tau * sim.rc ** 2
        self._load_or_make()
        t = self.npz['t']
        i0 = 0
        while t[i0] == t[i0 + 1]:
            i0 += 1
        jf = t.size - 1
        while t[jf] == t[jf - 1]:
            jf -= 1
        self._sl = slice(i0, jf + 1)

    def _load(self):
        if not hasattr(self.npz, 'lower'):
            return True
        try:
            self.npz =  np.load(self.npz)
            return True
        except IOError:
            return False

    def _load_or_make(self):
        if not self._load():
            #tmp = ['CS', 'CA', 'CL', 'Mdot', 'dd', 'dens', 'vr', 'vphi', 'pseudo',
            #       'v1v2', 'vortensity', 'mom2']
            print('Generating flux data')
            tmp = self.sim.loadfile('FT', 0).flux_variables()
            fn = self.sim._get_flux_fn(1, 2, 0, check=False)
            data = self.sim._mk_flux_data(ll=True)
            out = dict(zip(tmp, np.swapaxes(data, 0, 1)))
            gc.collect()
            try:
                out['t'] = self.sim.gen_fft_times()
            except (MemoryError, OSError):
                out['t'] = self.sim.gen_fft_times(ll=False)
            np.savez(fn, **out)
            self.npz = np.load(fn)

    def __getitem__(self, item):
        if item == 'mom1':
            return self.npz['Mdot'][self._sl]
        if item == 'Mdot':
            return self.Mdot()
        if item == 'mdot':
            raise KeyError("Key 'mdot' does not exist, but 'Mdot' does.")
        try:
            return self.npz[item][self._sl]
        except KeyError:
            pass
        tmp = {'vr': 'vel1', 'vphi': 'vel2', 'mom2': 'rhov2'}
        if item in tmp:
            return self[tmp[item]]
        if item == 'CS':
            return self.T12(3)
        if item == 'CA':
            return self.CA(3)
        if item == 'dd':
            return self['dens**2'] / self['dens'] - 1.0
        raise KeyError("FluxData does not contain {:}.".format(item))

    @property
    def csm(self):
        return self.sim.CS_RRR_data()[self._sl]

    def Mdot(self):
        return tau * self.rc * self['mom1']

    def mdot(self):
        raise AttributeError("Did you mean 'Mdot' instead of 'mdot'?")

    def hann(self, f, ns=None, dt=False):
        try:
            if ns < 0:
                ns = 0
        except TypeError:
            pass
        if ns is None:
            ns = self.ns
        if not ns:
            return f
        kern = scipy.signal.windows.hann(ns)
        if f.ndim == 2:
            kern = kern[:, np.newaxis]
        kern /= kern.sum()
        if dt:
            if ns > 2:

                kern /= self['t'][ns] * rc('ff')
                f = np.vstack((f, f[-1:-ns-2:-1, :]))
        #print(kern)
        return scipy.signal.fftconvolve(np.real(f), kern, mode='same', axes=0)
        #return scipy.signal.oaconvolve(np.real(f), kern, mode='same', axes=0)

    def vphi1(self):
        return self.sim.rc ** -.5

    def vphi2(self):
        return self['mom2'] / self['dens']

    def vphi3(self):
        return self['vphi']

    def flux_est(self, i, p, rpow=-3):
        return phi_visable(self.rc, i) * self['dens']**p * self.rc**rpow * np.diff(self.r)

    def dt(self, f, ns=None):
        if not ns:
            return grad(self['t'], self.hann(f, ns=ns, dt=True), axis=0)
        t = self['t']
        loc = slice(0, t.size)
        t = np.hstack([t, t[1:-ns+2] + t[-1]])
        if ns < 0:
            hf = f
        else:
            hf = self.hann(f, ns=ns, dt=True)
        return grad(t, hf, axis=0)[loc]


    def dr(self, f):
        return grad(self.sim.rc, f, axis=-1)

    def _option(self, option):
        if option is None:
            option = self.dopt
        if option not in [1, 2, 3]:
            msg = "Option value {:} is not valid. Must be in [1, 2, 3]."
            raise ValueError(msg.format(option))
        return option

    def _smoothing(self, s):
        if s is None:
            s = self.dsmooth
        if s not in [1, 2]:
            msg = "Smooth value {:} is not valid. Must be in [1, 2]."
            raise ValueError(msg.format(s))
        return s

    def _vphi(self, vphi=None):
        if vphi is None:
            vphi = self.dopt
        try:
            if vphi == int(vphi):
                vphi = getattr(self, 'vphi' + str(vphi))()
        except TypeError:
            pass
        return vphi

    def _dm2(self, vphi=None):
        if np.all(vphi == 2):
            return 0
        vphi = self._vphi(vphi)
        return self['mom2'] - vphi * self['dens']

    def CA(self, vphi=None):
        vphi = self._vphi(vphi)
        return self.r2 * self['mom1'] * vphi

    def yCA(self, vphi=None):
        dlogl = self.dr(np.log(self._vphi(vphi) * self.rc))
        return self.CA(vphi) * dlogl

    def T12(self, vphi=None):
        vphi = self._vphi(vphi)
        return self['CL'] - self.CA(vphi)

    def dl(self, vphi=None):
        return self.dr(self._vphi(vphi) * self.rc)

    def yMdot(self, vphi=None, ns=None):
        out = -self['Mdot'] * self.dl(vphi=vphi)
        if ns:
            out = self.hann(out, ns=ns)
        return out

    def ycs(self, vphi=None, ns=None):
        out = self.dr(self.T12(vphi))
        if ns:
            out = self.hann(out, ns=ns)
        return out

    def time_slice(self, t0=None, tf=None, ts=0, tnorm=None, **ignore):
        if tnorm is None:
            tnorm = tau
        if not tnorm or tnorm < 0:
            tnorm = 1
        title = ''
        try:
            title = self.sim.name + ' '
        except AttributeError:
            pass
        if tnorm == tau:
            title += '$t/ 2 \pi={:.1f}-{:.1f}$'.format(t0, tf)
        t0 *= tnorm
        tf *= tnorm
        ts *= tnorm
        if tnorm != tau:
            title += '$t/ 2 \pi={:.1f}-{:.1f}$'.format(t0 / tau, tf / tau)
        t = self['t']
        invdt = 1 / (tf - t0 - ts)
        i0, il = np.searchsorted(t, [t0, tf])
        if il < t.size - 1:
            il += 1
        its = int(np.round(np.searchsorted(t, ts))) + 1
        loc = (slice(i0, il), slice(None))
        loc1 = (slice(i0, i0 + its + 1), slice(None))
        loc2 = (slice(il - its - 1, il), slice(None))
        return dict(t0=t0, tf=tf, ts=ts, tnorm=tnorm, loc=loc, loc1=loc1, loc2=loc2,
                    i0=i0, its=its, il=il, invdt=invdt, title=title)

    def smooth_dt(self, f, ns=None, **kwargs):
        if 'loc1' not in kwargs:
            kwargs = self.time_slice(**kwargs)
        loc1 = kwargs['loc1']
        loc2 = kwargs['loc2']
        out = (self.hann(f[loc2], ns=ns) - self.hann(f[loc1], ns=ns)).mean(axis=0)
        return out * kwargs['invdt']

    def coef_p(self):
        return np.pi * self.sim.rc ** 3.5 * self.sim.mach**-2

    def ydp(self, o=None, s=None, ns=None, sd=None, **kwargs):
        o = self._option(o)
        s = self._smoothing(s)
        vphi = self._vphi(o)
        dm2 = self._dm2(o)
        if sd is None:
            sd = self.time_slice(**kwargs)

        if o == 1:
            if s == 1:
                ns = 0
            dr_rho = self.dr(self.hann(self['dens'], ns=ns))
        elif o == 2:
            if s == 1:
                dr_rho = self.hann(self.dr(self['dens']) / self['dens'], ns=ns, dt=True)
            else:
                hd = self.hann(self['dens'], ns=ns, dt=True)
                dr_rho = self.dr(hd) / hd
        elif o == 3:
            if 'ln_rho' in self.npz:
                dr_rho = self.dr(self.hann(self['ln_rho'], ns=ns, dt=True))
            else:
                dr_rho = self.dr(self.hann(np.log(self['dens']), ns=ns, dt=True))

        coef = self.coef_p()
        if o != 1:
            if coef.ndim == 1:
                coef = coef[np.newaxis, :] * self['dens']
            else:
                coef *= self['dens']

        if o == 1:
            return coef * self.smooth_dt(dr_rho, **sd)
        return coef * self.dt(dr_rho, ns=-ns if ns else ns)

    def ydv2(self, vphi=None, s=None, ns=None):
        vphi = self._vphi(vphi)
        dm2 = self.dt(self._dm2(vphi), ns=ns)
        if vphi.ndim < 2:
            dv2dt = 0
        else:
            dv2dt = self.dt(vphi, ns=ns)
        d = self['dens'] if s == 1 else self.hann(self['dens'], ns=ns)
        return self.r2 * (d * dv2dt + dm2)

    def R(self, vphi=None):
        vphi = self._vphi(vphi)
        return self['rhoDvrDt'] - self['rhov2v2'] + 2 * vphi * self['rhov2'] \
               - vphi**2 * self['dens']

    def mdot_split(self, t0, tf, save=False, fn=False, sdir=None, overwrite=True,
                   figsize=None, dpi=300, popt=None, fig=None, ax=None, title=True,
                   lnorm=True, ext='pdf', lloc=None, legend=True):
        if save or fn:
            save = True
            if fn is None:
                fn = '_mdot_split_{:04d}_{:04d}.'.format(t0, tf)
                fn = os.path.join(sdir, self.sim.name + fn + '.' + ext.lstrip('.'))
                if sdir:
                    if not os.path.isdir(sdir):
                        os.mkdir(sdir)
        if parse_not_overwrite(overwrite, fn):
            return
        ts = self.time_slice(t0=t0, tf=tf)
        loc = ts['loc']
        lntxt = '$'
        if lnorm:
            if lnorm is True:
                lnorm = -3
            lnorm = int(lnorm)
            lntxt = '/10^{' + str(lnorm) + '}$'
        if not lnorm:
            lnorm = 0
        if popt is None:
            popt = dict()

        ys = [tau * self.rc * (self['vr'] * self['dens'])[loc].mean(axis=0) * 10**-lnorm]
        mdot = (self['Mdot'])[loc].mean(axis=0)
        ys.extend([mdot * 10**-lnorm - ys[0], mdot * 10**-lnorm])

        if fig is None and ax is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        if ax is not None:
            plt.sca(ax)
        for i in range(3):
            plt.plot(self.rc, -ys[i], c=('k' if i == 2 else None), **popt)
        lbls = [r'$-2\pi r\left<v_r\right>\left<\Sigma\right>$',
                r'$-2\pi r\left<v_r\delta\Sigma\right>$',
                r'$\dot{M}$']
        plt.axhline(0, c='.5', lw=1, ls=':', zorder=-10)
        rin = self.sim.rloc(1.01)
        rout = -10
        ys = np.array(ys)
        yl, yu = ys[:, rin:rout].min(), ys[:, rin:rout].max()
        dy = (yu - yl) * .05
        plt.ylim(yl - dy, yu + dy)
        plt.xlim(1, self.sim.r[-1])
        ax = plt.gca()
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.1))
        if legend:
            plt.legend(lbls, loc=lloc)
        plt.xlabel('$r$')
        plt.ylabel(r'$\dot{M}' + lntxt)
        if title is True:
            title = ts['title']
        if title:
            plt.title(title)

        if save or fn:
            plt.savefig(fn)
            plt.close()
        return

    def lines(self, o=None, s=None, ns=None, pns=None, mean=True, norm=1, yCA=False,
              **kwargs):
        if pns is None:
            pns = ns
        vphi = self._vphi(o)
        out = [self.ycs(vphi, ns=ns), self.ydv2(vphi, s, ns=ns), self.yMdot(vphi, ns=ns),
               self.ydp(o, s, ns=pns, **kwargs)]
        if yCA:
            out.append(self.yCA(vphi))
        if norm and norm != 1:
            out = [i * norm for i in out]
        if mean:
            sd = self.time_slice(**kwargs)
            for i, var in enumerate(out):
                if var.ndim == 2:
                    out[i] = var[sd['loc']].mean(axis=0)
        out.append(out[0] + out[1])
        return out

    def plot_lines(self, o=None, s=None, ns=None, extras=True, ax=None, lopt=None,
                   plt_ydp=True, save=False, fn=None, overwrite=True, **kwargs):
        if save or fn:
            save = True
            if fn is None:
                t0 = kwargs.get('t0', 0)
                tf = kwargs.get('tf', 0)
                fn = '_flux_{:04d}_{:04d}.'.format(t0, tf)
                sdir = kwargs.get('sdir', '')
                ext = kwargs.get('ext', 'pdf')
                fn = os.path.join(sdir, self.sim.name + fn + '.' + ext.lstrip('.'))
                if sdir:
                    if not os.path.isdir(sdir):
                        os.mkdir(sdir)
        if parse_not_overwrite(overwrite, fn):
            return
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        lines = self.lines(o=o, s=s, ns=ns, **kwargs)
        plt.plot(self.rc, lines[0], label=r'$\partial_r C_{\rm S}$', lw=1)
        plt.plot(self.rc, lines[1], label=r'$\partial_t v_\phi$', lw=1)
        plt.plot(self.rc, lines[2], label=r'$\dot{M}\partial_r\ell$', c='k')
        if plt_ydp:
            plt.plot(self.rc, lines[3], label=r'$\partial_t\partial_rP$', ls=':', lw=1)
        plt.plot(self.rc, lines[4], label=r'$\partial_r C_{\rm S}\! +\! \partial_t v_\phi$',
                 c='.5', lw=1)
        if len(lines) > 5:
            plt.plot(self.rc, -lines[5], label=r'$-C_{\rm A}\partial_r \ln\ell$', lw=1)
        if extras:
            plt.axhline(0, c='.5', ls=':', lw=1)
            plt.axvline(1, c='.5', ls=':', lw=1)
            plt.xlabel('$R$')
            plt.ylabel(r'$\left[\dot{M}\partial_r\ell\right]$')
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis='both', which='both', direction='in', zorder=10)
            ax.set_axisbelow(False)
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.1))
            # legend
            if lopt is None:
                lopt = kwargs.get('lopt', dict(handlelength=1, fontsize=8,
                                  handletextpad=.4, columnspacing=.7))
            plt.legend(loc=9, ncol=5, **lopt)
            # set_ylim
            ri = self.sim.rloc(1.01)
            ri2 = self.sim.rloc(3)
            yu = np.maximum(np.nan_to_num(lines[0]), np.nan_to_num(lines[2]))
            yu = yu[ri:ri2].max() * 1.05
            yl = np.minimum(np.nan_to_num(lines[1]), np.nan_to_num(lines[2]))
            if len(lines) > 5:
                yl = np.minimum(yl, np.nan_to_num(-lines[5]))
            # yl = -v('Mdot')
            yl = yl[ri:ri2].min() - .1 * yu
            _yl, _yu = plt.ylim(yl, yu)
            plt.xlim(1, self.sim.r[-1])
        if save or fn:
            plt.savefig(fn)
            plt.close()
        return lines

    def R_plot(self, o=None, s=None, ns=None, extras=True, ax=None, save=None, fn=None,
               **kwargs):
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        v0 = self._vphi(o)
        sd = self.time_slice(**kwargs)
        rho = self['dens']
        lines = [-self.dr(rho / self.sim.mach**2)[sd['loc']].mean(axis=0),
                 -self.R(o)[sd['loc']].mean(axis=0),
                 (2*v0/self.rc*(self['rhov2']-rho*v0))[sd['loc']].mean(axis=0),
                 (rho * (v0**2 / self.rc-self.rc**-2))[sd['loc']].mean(axis=0)
        ]
        plt.plot(self.rc, lines[0], label=r'$-\partial_r P$')
        plt.plot(self.rc, lines[1], label=r'$-R$')
        plt.plot(self.rc, lines[2], label='\#2')
        plt.plot(self.rc, lines[3], label='\#1-\#4')
        plt.plot(self.rc, np.asarray(lines).sum(axis=0), lw=1, c='k', label='Sum')
        if extras:
            plt.axhline(0, c='.5', ls=':', lw=1)
            plt.xlabel('$r$')
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(axis='both', which='both', direction='in', zorder=10)
            ax.set_axisbelow(False)
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.1))
            # legend
            plt.legend()
            # set_ylim
            plt.ylim(-5e-2, 5e-2)
            plt.xlim(1, self.sim.r[-1])
        if save or fn:
            plt.savefig(fn)
            plt.close()
        return lines

    def flux_compare(self, s=None, ns=None, figsize=None, dpi=300, fn=None, save=None,
                     **kwargs):
        if figsize is None:
            figsize = np.array((11*.5, 8.5)) * .8
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ts = self.time_slice(**kwargs)
        ax = None
        gs = mpl.gridspec.GridSpec(3, 1, top=.93, left=.08, right=.98, bottom=.08,
                                   hspace=0)
        for i in range(3):
            ax = plt.subplot(gs[i], sharex=ax, sharey=ax)
            self.plot_lines(i + 1, s=s, ns=ns, extras=True, **kwargs)
            plt.xlabel('')
            if i == 2:
                plt.xlabel('$r$')
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            lbl = chr(ord('a') + i) + ') Option ' + str(i + 1)
            ax.text(.96, .8, lbl, c='k', transform=ax.transAxes, ha='right', fontsize=8)

        fig.suptitle(ts['title'])

        if save or fn:
            plt.savefig(fn)
            plt.close()
        return

    def paper_flux_plot(self, o=None, s=None, ns=None, nm=5, figsize=None, dpi=300,
                        fn=None, save=None, lnorm=True, norm=1, lopt=None, axs=None,
                        use_txt=True, **kwargs):
        ts = self.time_slice(**kwargs)
        lntxt = '$'
        if lnorm:
            if lnorm is True:
                lnorm = int(np.round(-2 * np.log10(self.sim.mach) - 3))
            lntxt = '/10^{' + str(lnorm) + '}$'
        else:
            lnorm = 0
        # choose modes to plot
        csm = self.csm[ts['loc']].mean(axis=0) * 10**(-lnorm) * norm
        csm[0, :] = 0
        window = np.ones_like(self.rc)
        window[:self.sim.rloc(1.0)] = 0
        window[-5:] = 0
        _norm = self.sim.intr(np.abs(csm * window[None, :]))
        modes = sorted(range(_norm.shape[0]), key=lambda x: -_norm[x])

        # setup plotting
        if lopt is None:
            lopt = dict(handlelength=1, fontsize=8, handletextpad=.4, columnspacing=.7)
        if figsize is None:
            figsize = np.array((11*.5, 8.5)) * .8
        ax = None
        if axs is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            gs = mpl.gridspec.GridSpec(3, 1, top=.92, left=.15, right=.98, bottom=.05,
                                       wspace=.15, hspace=0)
            axs = []
            for i in range(3):
                ax = plt.subplot(gs[i], sharex=ax)
                axs.append(ax)
        tx, ty = .98, .94
        topt = dict(c='k', ha='right', fontsize=8)
        ri = self.sim.rloc(1.02)

        # C_S, C_S,m plot
        ax = axs[0]
        plt.sca(ax)
        cs = self.T12(o)[ts['loc']].mean(axis=0) * 10**(-lnorm) * norm
        plt.plot(self.rc, cs, 'k-', label='$C_S$')
        ym = []
        for m in modes[:nm]:
            ym.append(csm[m])
            plt.plot(self.rc, csm[m], label=str(m), lw=1)
        plt.plot(self.rc, csm[1:].sum(axis=0), c='.5', ls='-', label='sum', lw=1)
        ym = np.array(ym)
        plt.legend(loc=4, ncol=nm + 2, **lopt)
        plt.axhline(0, c='.5', ls=':', lw=1)
        plt.axvline(1, c='.5', ls=':', lw=1)
        # set ylim
        yl = np.minimum(csm[1:].sum(axis=0), cs)
        yl = np.minimum(yl, ym.min(axis=0))[ri:-5].min()
        yl = min(0, yl)
        yu = np.maximum(csm[1:].sum(axis=0), cs)
        yu = np.maximum(yu, ym.max(axis=0))[ri:-5].max()
        yu = max(0, yu)
        dy = (yu - yl) * .05
        plt.ylim(min(yl - dy, -2.5 * dy), max(yu + dy, 2.5 * dy))
        # labels
        plt.ylabel('$C_S' + lntxt)
        if use_txt:
            ax.text(tx, ty, 'a)', transform=ax.transAxes, **topt)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='both', direction='in', zorder=10)
        ax.set_axisbelow(False)

        # C_L, C_A, C_S
        ax = axs[1]
        plt.sca(ax)
        yu = []
        yl = []
        cl = self['CL'][ts['loc']].mean(axis=0) * 10**(-lnorm) * norm
        ca = cl - cs
        plt.plot(self.rc, cs, label='$C_S$', c='k')
        plt.plot(self.rc, ca, label='$C_A$', lw=1)
        plt.plot(self.rc, cl, label='$C_L$', lw=1)
        plt.legend(loc=4, ncol=3, **lopt)
        plt.axhline(0, c='.5', ls=':', lw=1)
        plt.axvline(1, c='.5', ls=':', lw=1)
        y = np.array([cs, ca, cl])[:, ri:-5]
        yl, yu = y.min(), y.max()
        dy = (yu - yl) * .05
        plt.ylim(yl - dy, max(yu + dy, 2.5 * dy))
        plt.ylabel('$C_i' + lntxt)
        if use_txt:
            ax.text(tx, ty, 'b)', transform=ax.transAxes, **topt)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='both', direction='in', zorder=10)
        ax.set_axisbelow(False)

        # Mdot, new!
        ax = axs[2]
        plt.sca(ax)
        self.plot_lines(o=o, s=s, ax=ax, norm=10**(-lnorm) * norm, **kwargs)
        plt.ylabel(r'$\left[\dot{M}\partial_r\ell\right]' + lntxt)
        if use_txt:
            ax.text(tx, ty, 'c)', transform=ax.transAxes, **topt)

        if save or fn:
            plt.savefig(fn)
            plt.close()
        return

    def bl_width(self, rl=.25, ru=.75, o=None):
        omega = self._vphi(o) / self.sim.rc
        ri = omega.argmax(axis=-1)
        out = np.empty((ri.size, 2))
        for i in range(ri.size):
            omax = omega[i, ri[i]]
            tmp = omega[i].copy()
            tmp[ri[i]:] = 0
            out[i, 0] = self.sim.rc[np.abs(tmp - rl * omax).argmin()]
            out[i, 1] = self.sim.rc[np.abs(tmp - ru * omax).argmin()]
        return out

    def plateau(self, frac=.9, o=None):
        omega = self._vphi(o) / self.sim.rc
        ri = omega.argmax(axis=-1)
        out = np.empty((ri.size, 2))
        for i in range(ri.size):
            omax = omega[i, ri[i]]
            inner = omega[i].copy()
            outer = omega[i].copy()
            inner[ri[i]:] = 0
            outer[:ri[i]] = 0
            out[i, 0] = self.sim.rc[np.abs(inner - frac * omax).argmin()]
            out[i, 1] = self.sim.rc[np.abs(outer - frac * omax).argmin()]
        return out


class Lightcurves(object):
    def __init__(self, filenames, sim=None, path=None, detect_npz=True, auto_export=True,
                 tunit=18.4, skip=50, old=None):
        self.sim = sim
        if path is None:
            if sim is None:
                path = ''
            else:
                path = sim.path
        self.path = path
        if sim and detect_npz:
            fn = 'lightcurve.npz'
            if os.path.isfile(os.path.join(path, fn)):
                self._source_files = np.atleast_1d(filenames)
                filenames = fn
        self.filenames = np.atleast_1d(filenames)
        self._auto_export = auto_export
        self.tunit = tunit
        self.skip = skip
        self._old = old

    def _extract(self):
        export = self._auto_export
        if self.filenames.size > 1 or self.filenames[0].split('.')[-1] == 'hdf5':
            for fn in self.filenames:
                with h5py.File(os.path.join(self.path, fn)) as f:
                    if 'time' in f:
                        if f['time'].size:
                            try:
                                flux = np.concatenate([flux, f['flux'][:]], axis=0)
                                time = np.concatenate([time, f['time'][:]], axis=0)
                            except NameError:
                                flux = f['flux'][:]
                                time = f['time'][:]
                                self._views = f.attrs['views'][:]
                                self._powers = f.attrs['powers'][:]
                                try:
                                    self._radii = f.attrs['radii'][:]
                                except KeyError:
                                    self._radii = np.array([1.0, self.sim.r[-1]])
        else:
            try:
                with np.load(os.path.join(self.path, self.filenames[0])) as f:
                    flux = f['flux']
                    time = f['time']
                    self._views = f['views']
                    self._powers = f['powers']
                    try:
                        self._radii = f['radii']
                    except KeyError:
                        self._radii =  np.array([1.0, self.sim.r[-1]])
                    export = False
            except OSError as e:
                print('Issue with npz file, reverting to source files.')
                print(e)
                self.filenames = self._source_files
                self._extract()
                return
        self._flux = atleast_4d(flux)
        self._time = time
        if export:
            self.export()

    @property
    def views(self):
        try:
            return self._views
        except AttributeError:
            self._extract()
            return self._views

    @property
    def powers(self):
        try:
            return self._powers
        except AttributeError:
            self._extract()
            return self._powers

    @property
    def radii(self):
        try:
            return self._radii
        except AttributeError:
            self._extract()
            return self._radii

    @property
    def flux(self):
        try:
            return self._flux
        except AttributeError:
            self._extract()
            return self._flux

    @property
    def time(self):
        try:
            return self._time
        except AttributeError:
            self._extract()
            return self._time

    def export(self, fn=None):
        if fn is None:
            fn = os.path.join(self.sim.path, 'lightcurve.npz')
        np.savez(fn, flux=self.flux, time=self.time, views=self.views, powers=self.powers)

    def _interpolate(self):
        dtimes = np.diff(self.time)
        dt = dtimes.min()
        newtime = np.arange(0, self.time[-1] + dt, dt)
        if newtime[-1] > self.time[-1]:
            newtime = newtime[:-1]
        i = 0
        remap = np.empty((newtime.size, self.views.size, self.powers.size,
                          self.radii.size - 1))
        for ti, t in enumerate(newtime):
            while t > self.time[i]:
                i += 1
            if i == dtimes.size:
                res = (t - self.time[i]) / dtimes[i - 1]
                i -= 1
            else:
                res = (t - self.time[i]) / dtimes[i]
            remap[ti] = self.flux[i] * (1.0 - res) + self.flux[i + 1] * res
        self._newtime = newtime
        self._remap = remap

    @property
    def newtime(self):
        try:
            return self._newtime
        except AttributeError:
            self._interpolate()
            return self._newtime

    @property
    def remap(self):
        try:
            return self._remap
        except AttributeError:
            self._interpolate()
            return self._remap

    @property
    def fine_est(self):
        try:
            return self._fine_est
        except AttributeError:
            self._fine_est = self.interp_est(self.newtime)
            return self._fine_est

    @property
    def coarse_est(self):
        try:
            return self._coarse_est
        except AttributeError:
            self._coarse_est = self.interp_est(self.newtime)
            return self._coarse_est

    def tloc(self, time, new=True):
        t = self.newtime if new else self.time
        return np.abs(t - time).argmin()

    def detrend(self, data, t=None, tloc=None):
        if t is None:
            t = self.newtime
            if tloc is not None:
                t = t[tloc]
        m, b = lintrend(t, data)
        return data - m * t + b

    def ft(self, t0=None, t1=None, n=None, window=None, detrend=None):
        if t0 is not None:
            t0 = self.tloc(t0)
        if t1 is not None:
            t1 = self.tloc(t1)
        loc = slice(t0, t1)
        if n is None:
            n = self.newtime[loc].size
        data = self.remap[loc]
        if window is not None:
            if detrend is None:
                detrend = True
            if hasattr(window, 'lower'):
                window = scipy.signal.get_window(window, n)
            tmp = tuple([slice(None)] + (data.ndim - window.ndim) * [np.newaxis])
            window = window[tmp]
        else:
            window = 1
        if detrend is True:
            m, b = lintrend(self.newtime[loc], data)
            detrend = m * self.newtime[loc, None, None] + b
        if np.any(detrend):
            data -= detrend
        data -= data.mean(axis=0)
        data *= window
        fourier = np.fft.fft(data, axis=0, n=n) / (.5 * n)
        freq = np.fft.fftfreq(n, d=self.newtime[1] / tau)
        return freq, fourier

    def plot_ft(self, vi=None, pi=None, ri=None, data=None, tloc=True, xlim=True, ylim=True, dpi=300,
                figsize=None, nufit=10, detrend=False, fig=None, ax=None, tmin=None,
                nu_dno=None, save=False, omax=None, oharm=None):
        if data is None:
            if pi is None or vi is None:
                raise ValueError('If data not specified, then vi and pi must be.')
            if tloc is None:
                if tmin:
                    tloc = slice(self.tloc(tmin * tau), None)
                tloc = slice(None)
            if tloc == True:
                tloc = slice(self.tloc(self.skip), None)
            data = self.remap[tloc, vi, pi, ri]
            if ri is None:
                data = data.sum(axis=-1)
        if detrend:
            d = self.detrend(data, tloc=tloc)
        else:
            d = data
        while d.ndim > 1:
            if d.shape[-1] == 1:
                loc = tuple((d.ndim-1) * [slice(None)] + [0])
                d = d[loc]
            else:
                raise ValueError("Data must be 1D.")
        ft = np.fft.fft(d / d.mean() - 1)
        nt = len(d)
        freq = np.fft.fftfreq(nt, d=self.newtime[1] / tau)
        x = freq[:nt//2]
        iu = np.searchsorted(x, nufit)
        loc = slice(1, iu)
        y = np.abs(ft[:nt//2])
        m, b, _, _, _ = scipy.stats.linregress(np.log(x[loc]), np.log(y[loc]))
        #print(m)
        pl = np.exp(m * np.log(x) + b)

        if fig is None and ax is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        plt.plot(x, np.abs(y / pl - 1))
        if nu_dno:
            plt.axvline(nu_dno, zorder=-1, lw=1, ls=':', c='.5')
        yl = plt.ylim()
        if np.any(omax):
            omax = np.mean(omax)
            if oharm is None:
                oharm = 1
            for h in np.atleast_1d(oharm):
                print(omax * h, h)
                plt.axvline(omax * h, zorder=-1 - h, lw=1, ls='-.', c='.5')
        if xlim:
            if xlim is True:
                xlim = [0, 6.5]
            plt.xlim(*xlim)
        if ylim:
            if ylim is True:
                xl, xu = np.searchsorted(x, [.1, plt.xlim()[1]])
                ylim = [0, np.abs(y / pl - 1).max() * 1.1]
            plt.ylim(*ylim)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
        plt.xlabel('freq. (per orbit)')
        plt.ylabel(r'$\left|A_\nu\right|$')
        def fwd(x):
            return x / self.tunit
        def bak(x):
            return x * self.tunit
        ax2 = ax.secondary_xaxis('top', functions=(fwd, bak))
        ax2.set_xlim(*(np.array(ax.get_xlim()) / self.tunit))
        ax2.set_xlabel('Est. freq. (Hz)')
        ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.01))
        if save:
            plt.savefig(self.sim.name + '_lc_ft.pdf')
            plt.close()

    def periodogram(self, vi=None, pi=None,ri=None, data=None, tloc=True, xlim=True, ylim=True,
                    dpi=300, figsize=None, nufit=20, window='hann', detrend=False,
                    nu0=None, fig=None, ax=None, ylog=True, rel=False, tmin=None,
                    nu_dno=None, save=False, omax=None, oharm=None):
        if data is None:
            if pi is None or vi is None:
                raise ValueError('If data not specified, then vi and pi must be.')
            if tloc is None:
                if tmin:
                    tloc = slice(self.tloc(tmin * tau), None)
                tloc = slice(None)
            if tloc == True:
                tloc = slice(self.tloc(self.skip), None)
            data = self.remap[tloc, vi, pi, ri]
            if ri is None:
                data = data.sum(axis=-1)
        if detrend:
            d = self.detrend(data, tloc=tloc)
        else:
            d = data
        if rel:
            d = d / d.mean() - 1
        while d.ndim > 1:
            if d.shape[-1] == 1:
                loc = tuple((d.ndim-1) * [slice(None)] + [0])
                d = d[loc]
            else:
                raise ValueError("Data must be 1D.")
        if nu0 is None:
            nu0 = tau / (self.newtime[1])
        f, Pxx_den = scipy.signal.periodogram(d, nu0, window, detrend='linear')

        if fig is None and ax is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        if ylog:
            plt.semilogy(f, Pxx_den, zorder=0)
        else:
            plt.plot(f, Pxx_den, zorder=0)
        if nu_dno:
            plt.axvline(nu_dno, zorder=-1, lw=1, ls=':', c='.5')
        if np.any(omax):
            omax = np.mean(omax)
            if oharm is None:
                oharm = 1
            for h in np.atleast_1d(oharm):
                plt.axvline(omax * h, zorder=-1 - h, lw=1, ls='-.', c='.5')
        if xlim:
            if xlim is True:
                xlim = [0, 6.5]
            plt.xlim(*xlim)
        if ylim:
            if ylim is True:
                xl, xu = np.searchsorted(f, [.1, plt.xlim()[1]])
                ylim = [0, Pxx_den[xl:xu].max() * 2]
                if ylog:
                    ylim[0] = 0.5 * Pxx_den[xl:xu].min()
            plt.ylim(*ylim)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
        plt.xlabel('freq. (per orbit)')
        plt.ylabel(r'periodogram')
        def fwd(x):
            return x / self.tunit
        def bak(x):
            return x * self.tunit
        ax2 = ax.secondary_xaxis('top', functions=(fwd, bak))
        ax2.set_xlim(*(np.array(ax.get_xlim()) / self.tunit))
        ax2.set_xlabel('Est. freq. (Hz)')
        ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.01))
        plt.sca(ax)
        if save:
            plt.savefig(self.sim.name + '_periodogram.pdf')
            plt.close()

    def spectrogram(self, vi=None, pi=None, ri=None, data=None, tloc=True, xlim=True, ylim=True,
                    dpi=300, figsize=None, nufit=20, window='hann', detrend=False,
                    nu0=None, fig=None, ax=None, log=True, rel=False, nperseg=None,
                    tperseg=20, vmin=1e-8, vmax=True, norm=None, cmap=None, sdata=None,
                    cb=True, cbl=True, cax=None, nu_dno=None, save=False, omax=None,
                    oharm=None, ot=None, fd=None):
        if sdata is None:
            if data is None:
                if pi is None or vi is None:
                    raise ValueError('If data not specified, then vi and pi must be.')
                if tloc is None:
                    tloc = slice(None)
                if tloc is True:
                    tloc = slice(self.tloc(self.skip), None)
                data = self.remap[tloc, vi, pi, ri]
                if ri is None:
                    data = data.sum(axis=-1)
            if detrend:
                d = self.detrend(data, tloc=tloc)
            else:
                d = data
            if rel:
                d = d / d.mean() - 1
            while d.ndim > 1:
                if d.shape[-1] == 1:
                    loc = tuple((d.ndim - 1) * [slice(None)] + [0])
                    d = d[loc]
                else:
                    raise ValueError("Data must be 1D.")
            if nu0 is None:
                nu0 = tau / (self.newtime[1])
            if nperseg is None:
                nperseg = self.tloc(tperseg * tau)
        if norm is None and log:
            norm = mpl.colors.LogNorm()
        if sdata is None:
            f, t, Sxx = scipy.signal.spectrogram(d, nu0, window, nperseg=int(nperseg),
                                                 detrend='linear')
        else:
            f, t, Sxx, nu0, nperseg = sdata
        if fig is None and ax is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
        elif not fig:
             fig = plt.gcf()
        if ax:
            plt.sca(ax)
        else:
            ax = plt.gca()
        if vmax is True:
            vmax = Sxx[1:, 1:].max()
        im = plt.pcolormesh(t, f, Sxx, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap)
        if nu_dno:
            plt.axhline(nu_dno, lw=1, ls=':', c='.5')
        xl = plt.xlim()
        yl = plt.ylim()
        if np.any(omax):
            if omax is True or ot is True:
                if fd is None:
                    fd = self.sim.load_flux_data()
                if omax is True:
                    omax = (fd.vphi2() / self.sim.rc).max(axis=-1)
                if ot is True:
                    ot = fd['t'] / tau
            if oharm is None:
                oharm = range(1, 9)
            print(ot, omax)
            for h in np.atleast_1d(oharm):
                plt.plot(ot, omax * h, lw=1, ls=':', c='w', alpha=.4)
        plt.xlim(*xl)
        plt.ylim(*yl)
        ax = plt.gca()
        plt.ylabel('freq. (per orbit)')
        plt.xlabel(r'$t/2\pi$')
        if ylim:
            if ylim is True:
                ylim = [0, 6.5]
                plt.ylim(*ylim)
        if cb:
            offset = False
            if cax is None:
                divider = make_axes_locatable(ax)
                if cax is None:
                    cax = divider.append_axes("top", size="5%", pad=0.05)
                    offset = True
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
            if offset:
                cb.ax.xaxis.set_label_position('top')
                cb.ax.xaxis.set_ticks_position('top')
            if cbl:
                if cbl is True:
                    cbl = r'$\left|A_\nu\right|$'
                cb.set_label(cbl)
        #ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        #ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.25))
        def fwd(x):
            return x / self.tunit
        def bak(x):
            return x * self.tunit
        ax2 = ax.secondary_yaxis('right', functions=(fwd, bak))
        ax2.set_ylim(*(np.array(ax.get_ylim()) / self.tunit))
        ax2.set_ylabel('Est. freq. (Hz)')
        #ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.01))
        plt.sca(ax)
        if save:
            plt.savefig(self.sim.name + '_spectrogram.png')
            plt.close()
        return f, t, Sxx, nu0, nperseg

    def plot_lc(self, vi, pi, save=False, dpi=300):
        plt.figure(dpi=dpi)
        plt.plot(self.newtime / tau, self.remap[:,vi, pi])
        if save:
            plt.savefig(self.sim.name + '_full_lightcurve.pdf')
            plt.close()

    def zoom_lc(self, vi, pi, tl, tu, save=False, dpi=300):
        plt.figure(dpi=dpi)
        il = self.tloc(tl * tau)
        iu = self.tloc(tu * tau) + 1
        d = self.remap[:,vi, pi]
        plt.plot(self.newtime / tau, d)
        yl = d[il:iu].min()
        yu = d[il:iu].max()
        dy = (yu - yl) * .025
        plt.xlim(tl, tu)
        plt.ylim(yl - dy, yu + dy)
        if save:
            plt.savefig(self.sim.name + '_zoom_lc.pdf')
            plt.close()

    def several_plots(self, vi=2, pi=4, tl=275, tu=300, sdir=None, save=True, tmin=100):
        sim = self.sim
        if not save:
            sdir = None
        if sdir is None:
            sdir = ''
        if sdir is True:
            sdir = os.path.split(sim.path)[-1] + '_plots'
            sdir = os.path.join(rc('fig_base_dir'), sdir, 'lc')
        if not os.path.isdir(sdir):
            os.mkdir(sdir)
        pwd = os.getcwd()
        try:
            os.chdir(sdir)
            sim.compact_diag(save=save)
            nu_dno = sim.cc_op_plots(tu, save=save, rmin=1.6)
            print('nu = {:.3g} per orbit, {:.3g} mHz'.format(nu_dno, nu_dno / self.tunit * 1e3))
            self.plot_lc(vi, pi, save=save)
            self.zoom_lc(vi, pi, tl, tu, save=save)
            self.plot_ft(vi, pi, tmin=tmin, save=save, nu_dno=nu_dno)
            self.periodogram(vi, pi, tmin=tmin, save=save, nu_dno=nu_dno)
            self.spectrogram(vi, pi, save=save, nu_dno=nu_dno)
        finally:
            os.chdir(pwd)

    def flux_est(self, fd=None, bins=False, ro=1):
        if fd is None:
            fd = self.sim.load_flux_data()
        na = np.newaxis
        rho = fd['dens'][:, na, na, :]
        i = self.views[na, :, na, na]
        p = self.powers[na, na, :, na]
        r = self.sim.rc[na, na, na, :]
        dr = np.diff(self.sim.r)[na, na, na, :]
        flux = rho**p * r**(ro-3) * phi_visable(r, i) * dr
        if not bins:
            return flux
        out = np.empty(flux[:,:,:,0].shape + (self.radii.size - 1,))
        for ri in range(self.radii.size - 1):
            a = np.argmin(np.abs(self.sim.r - self.radii[ri]))
            b = np.argmin(np.abs(self.sim.r - self.radii[ri + 1]))
            out[:,:,:,ri] = flux[:, :, :, a:b].sum(axis=-1)
        return out

    @property
    def interp_est(self):
        try:
            return self._interp_est
        except AttributeError:
            fd = self.sim.load_flux_data()
            data = self.flux_est(fd=fd, bins=True)
            self._interp_est = CubicSpline(fd['t'], data)
            return self._interp_est

    def fine_normalized(self, vi=None, pi=None, ri=None, rsum=None):
        if vi is None:
            vi = slice(None)
            if rsum is None:
                rsum = False
        if pi is None:
            if rsum is None:
                rsum = False
        if ri is None:
            ri = slice(None)
            if rsum is None:
                rsum = True
        loc = slice(None), vi, pi, ri
        top = self.remap[loc]
        bot = self.fine_est[loc]
        if rsum:
            top = top.sum(axis=-1)
            bot = bot.sum(axis=-1)
        return top / bot


class modeData(object):
    def __init__(self, data, sim=None, dw=.1, overlap=9.9, nbin=3):
        self.sim = sim
        self.dw = dw
        self.nbin = nbin
        self._dt = overlap
        mask = np.logical_not(data['mask'])
        run = data['run']
        r = data['r']
        fits = data['fits']
        tlist = data['tlist']
        self.r = r
        out = [[] for i in r]
        for z in zip(*np.where(run == nbin)):
            # weighted mean speed
            w = fits[z[0], z[1], z[2], 0]
            t0 = z[1]
            t1 = t0
            tmp = [i for i in out[z[0]] if (i[0] == z[2]) and (i[1] <= tlist[t0] <= i[2])]
            # if not tmp:
            if (not tmp) and (t0 < fits.shape[1] - 1):
                while (run[z[0], t1 + 1, z[2]] == nbin) and (
                        abs(fits[z[0], t1 + 1, z[2], 0] - w) < dw * w):
                    t1 += 1
                    w = fits[z[0], t0:t1, z[2], 0].mean()
                    if t1 >= run.shape[1] - 1:
                        break
                if t1 - t0 >= nbin:
                    t0 = tlist[t0] / tau
                    t1 = tlist[t1 + 1] / tau
                    out[z[0]].append([z[2], t0, t1, w])
        self.mode_data = [np.array(i) for i in out]
        # [[mode, ti, tf, speed] * modes_detected(r) for r in rlist]

    def filter(self, data=None):
        if data is None:
            try:
                return self._filter[:]
            except AttributeError:
                self._filter = [self.filter(i) for i in self.md]
                return self._filter[:]
        # kludge: sort first by mode then by duration
        tmp = sorted(data, key=lambda x: x[0] - 1e-6 * (x[2] - x[1]))
        out = []
        for mode in tmp:
            similar = [i for i in out if
                       (mode[0] == i[0]) and (abs(mode[3] - i[3]) < self.dw * i[3])]
            if not similar:
                out.append(mode)
        return np.array(out)

    def write(self, fn=None):
        if fn is None:
            fn = self.sim.name + '_modes.csv'
        rs = ["R = " + repr(i) for i in self.r]
        rs.append('Global Modes')
        data = self.filter()
        data.append(self.filter(self.g_modes()))
        out = ['# m, t_start, t_end, speed', '']
        for i, r in enumerate(rs):
            out.append('# ' + r)
            out += ['{0:d}, {1:.2f}, {2:.2f}, {3:.3f}'.format(int(j[0]), *j[1:]) for j in
                    data[i]]
            out.append('')
        with open(fn, 'w') as f:
            f.write('\n'.join(out))
        return

    def plot(self, save=False, fn=None, ext='pdf', inc_global=True, show_pl=True,
             mklbls=True, legend=True, ymax=-1, use_ymax=False, cap=1, lopt=None):
        markers = 'o', '+', 'x', '.'
        if lopt is None:
            lopt=dict()
        lbls = []
        xmax = 0
        for i, r in enumerate(self.r):
            try:
                xmax = max(xmax, self.filter()[i][:, 0].max())
                ymax = max(ymax, self.filter()[i][:, 3].max())
                plt.scatter(self.filter()[i][:, 0], self.filter()[i][:, 3],
                            marker=markers[i])
                lbls.append(r'$r={0:.2f}$'.format(r))
            except IndexError:
                pass
        if inc_global:
            data = self.g_modes()
            if np.any(data):
                plt.scatter(data[:, 0], data[:, 3], marker='*')
                lbls.append('Global')
        if legend:
            plt.legend(lbls, **lopt)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='both', direction='in')
        if mklbls:
            plt.xlabel('mode')
            plt.ylabel(r'$\Omega_{\rm p}$')
        if self.sim is not None:
            if mklbls:
                plt.title(helpers.sanitize_lbl(self.sim.name))
            if show_pl:
                xlim = plt.xlim(None)
                ylim = plt.ylim()
                _x = np.linspace(0, 32, 360)
                if xlim[1] > 40:
                    _x = np.linspace(0, 64, 720)
                M = int(self.sim.mach + .1)
                yl = np.sqrt(M ** -2 + (M / (2 * rc('rl')[M] * _x)) ** 2) / rc('rl')[M]
                plt.plot(_x, yl, c='.5', ls='-.', lw=1, zorder=-1)
                plt.plot(2 * _x, yl, c='.6', ls='-.', lw=1, zorder=-1)
                plt.plot(3 * _x, yl, c='.7', ls='-.', lw=1, zorder=-1)
                if M in rc('ru'):
                    yu = self.sim.upper_omega(_x, r=rc('ru')[M])
                    plt.plot(_x, yu, c='.5', ls=':', lw=1, zorder=-1)
                    plt.plot(2 * _x, yu, c='.6', ls=':', lw=1, zorder=-1)
                    plt.plot(3 * _x, yu, c='.7', ls=':', lw=1, zorder=-1)
                plt.xlim(*xlim)
                plt.ylim(*ylim)
                sim = self.sim
                omega = np.nan_to_num(sim.flux_data['vphi'] / sim.rc)
                omax = np.max(omega[2000:].mean(axis=0))
                plt.axhline(omax, c='.5', lw=1, ls='--', zorder=-2)
                xmax = max(xmax + 1, xlim[1])
                plt.xlim(None, xmax)
                ymax = max(ymax, omax)
                ymax = max(ymax, ylim[1])
        if use_ymax:
            ymax = min(cap, 1.05 * ymax)
            plt.ylim(None, ymax)
        if save or fn:
            if fn is None:
                fn = self.sim.name + '_dispersion.' + ext
            plt.savefig(fn)
            plt.close()
        if use_ymax:
            return ymax
        return

    def __getitem__(self, item):
        return self.mode_data[item]

    @property
    def md(self):
        return self.mode_data

    def g_modes(self):
        try:
            return self._g_modes
        except AttributeError:
            out = []
            for mode in self[0]:
                keep = True
                t0 = mode[1]
                t1 = mode[2]
                for r in self[1:]:
                    try:
                        mlist = [i for i in r if (i[0] == mode[0]) and (
                                abs(i[3] - mode[3]) < self.dw * mode[3])]
                    except IndexError:
                        mlist = []
                    while mlist:
                        if min(mlist[0][2], mode[2]) - max(mlist[0][1],
                                                           mode[1]) < self._dt:
                            mlist.pop(0)
                        else:
                            break
                    if not mlist:
                        keep = False
                        break
                    t0 = max(t0, mlist[0][1])
                    t1 = min(t1, mlist[0][2])
                if keep:
                    out.append([mode[0], t0, t1, mode[3]])
            self._g_modes = self.filter(np.array(out))
        return self._g_modes


class DataContainer:
    def __init__(self, fn, sim=None, df=None, prefix='cons', allow_pickle=False,
                 strip_objects=True):
        self._allow_pickle = allow_pickle
        self.fn = fn
        self.data = dict()
        self.sim = sim
        self._df = df
        self._prefix = prefix
        self._strip_objects = strip_objects
        self._file_data = self._load_own_data()
        if self._file_data is not None:
            for i in self._file_data:
                self.data[i] = self._strip(self._file_data[i])

    def _load_own_data(self):
        try:
            return np.load(self.fn, allow_pickle=self._allow_pickle)
        except IOError:
            self._gen_data(None)
            self._save()

    def _strip(self, item):
        if not self._strip_objects:
            return item
        if type(item) == np.ndarray:
            #print('strip:', item)
            if item.dtype == np.dtype(object):
                #print('obj')
                if item.shape == tuple():
                    #print('shape = ()')
                    return item.item()
        return item

    @property
    def df(self):
        if hasattr(self._df, 'sim'):
            return self._df
        if type(self._df) == int:
            self._df = self.sim.loadfile(self._prefix, self._df)
        else:
            self._df = self.sim.loadfile(self._df)
        return self._df

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            pass
        try:
            out = self._strip(self._file_data[key])
            self.data[key] = out
            return out
        except KeyError:
            pass
        func = getattr(self, str(key), getattr(self, '_' + str(key)))
        if callable(func):
            self.data[key] = func()
            self._save()
            return self.data[key]
        try:
            self._gen_data(key)
            out = self.data[key]
            self._save()
            return out
        except (NotImplementedError, KeyError):
            pass
        raise KeyError('Could not find or generate "{0}".'.format(key))

    def _save(self):
        np.savez(self.fn, **self.data)
        self._file_data = self._load_own_data()

    def _gen_data(self, *args):
        raise NotImplementedError

    def keys(self):
        return list(self.data.keys()) + list(self._file_data.keys())


class BLstats(DataContainer):
    def __init__(self, sim, df=600, fn=None):
        if fn is None:
            fn = os.path.join(sim.path, 'bl_stats_{0:04d}.npz'.format(df))
        super().__init__(fn, sim=sim, df=df)

    def _gen_data(self, *args):
        print("_gen_data")
        self.data.update(self.df.bl_stats())
