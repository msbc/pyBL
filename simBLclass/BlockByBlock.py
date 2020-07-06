import h5py
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from glob import glob
import sys
import sympy as sp
from .. import helpers
from ..NCcmap import NCcmap
from .defaults import rc


def _save_fig(fn, def_fn, sdir=None):
    if fn is None:
        fn = def_fn
    if sdir is not None:
        if not os.path.isdir(sdir):
            os.mkdir(sdir)
        fn = os.path.join(sdir, fn)
    plt.savefig(fn)
    plt.close()
    return fn


class Coord(object):
    def __init__(self, xf, xv, block_shape):
        na = np.newaxis
        self.one = np.ones(block_shape)
        self.x3f, self.x2f, self.x1f = xf
        self.x3v, self.x2v, self.x1v = xv
        self.face = [xf[0][:, na, na], xf[1][na, :, na], xf[2][na, na, :]]
        self.center = [xv[0][:, na, na], xv[1][na, :, na], xv[2][na, na, :]]
        self._keys = ['x' + i + j for i in '123' for j in 'vf']
        self._mapper = dict()

    def __getitem__(self, key):
        if key in self._mapper:
            key = self._mapper[key]
        if key in self._keys:
            try:
                return getattr(self, key)
            except AttributeError:
                pass
        raise KeyError('Unable to parse {0:}'.format(key))

    def __contains__(self, item):
        return item in self.keys()

    def cart_face(self):
        raise NotImplementedError

    def plot_cart_face(self, dim=None):
        return self.cart_face()

    def keys(self):
        return tuple(self._keys + list(self._mapper.keys()))

    def faces(self):
        return [self.one * i for i in self.face[::-1]]


class Cartesian(Coord):
    def __init__(self, xf, xv, block_shape):
        super(Cartesian, self).__init__(xf, xv, block_shape)
        self.zf, self.yf, self.xf = self.face
        self.zc, self.yc, self.xc = self.center
        tmp = 'xyz'
        self._keys += [i + j for i in tmp for j in 'fc']
        self._mapper = {i: i + 'c' for i in tmp}

    def cart_face(self):
        return self.faces()

class Cylindrical(Coord):
    def __init__(self, xf, xv, block_shape):
        super(Cylindrical, self).__init__(xf, xv, block_shape)
        self.zf, self.phif, self.rf = self.face
        self.zc, self.phic, self.rc = self.center
        tmp = ['z', 'phi', 'r']
        self._keys += [i + j for i in tmp for j in 'fc']
        self._mapper = {i: i + 'c' for i in tmp}

    def cart_face(self):
        z, phi, r = self.face
        return [r * np.cos(phi), r * np.cos(phi), z]

    def plot_cart_face(self, dim=None):
        z, phi, r = self.face
        if dim == 1:
            phi = 0
        out = [r * np.cos(phi), r * np.cos(phi), z]
        if dim == 1:
            out = [out[0], out[2], out[1]]
        return out

class SphericalPolar(Coord):
    def __init__(self, xf, xv, block_shape):
        super(SphericalPolar, self).__init__(xf, xv, block_shape)
        self.phif, self.thetaf, self.rf = self.face
        self.phic, self.thetac, self.rc = self.center
        tmp = ['phi', 'theta', 'r']
        self._keys += [i + j for i in tmp for j in 'fc']
        self._mapper = {i: i + 'c' for i in tmp}

    def cart_face(self):
        phi, th, r = self.face
        st = np.sin(th)
        return [r * st * np.cos(phi), r * st * np.cos(phi), r * np.cos(th)]

    def plot_cart_face(self, dim=None):
        phi, th, r = self.face
        if dim == 0:
            phi = 0.5 * np.pi * np.ones_like(phi)
        if dim == 1:
            th = 0.5 * np.pi * np.ones_like(th)
        st = np.sin(th)
        out = [r * st * np.cos(phi), r * st * np.sin(phi),
                r * np.cos(th) * np.ones_like(phi)]
        if dim == 1:
            out = [out[0], out[2], out[1]]
        return out


class Block(object):
    def __init__(self, index, data, var_names, xf, xv, logloc,
                 coord=None, expr_expand=None):
        self.index = index
        self.data = data
        self.block_shape = data.shape[1:]
        self.one = np.ones(self.block_shape)
        self.logical_location = logloc
        na = np.newaxis
        try:
            coord = coord.lower()
        except AttributeError:
            if coord is None:
                coord = 'cartesian'
            else:
                raise TypeError('coord must have string-like type or be None.')
        if coord == "spherical_polar":
            self.coord = SphericalPolar(xf, xv, self.block_shape)
        elif coord == "cylindrical":
            raise NotImplementedError
            self.coord = Cylindrical(xf, xv, self.block_shape)
        elif coord in ["cartesian", None]:
            self.coord = Cartesian(xf, xv, self.block_shape)
        else:
            raise ValueError('coord "{:}" not understood.'.format(coord))
        try:
            self.var_names = [i.decode("utf-8") for i in var_names]
        except AttributeError:
            self.var_names = [str(i) for i in var_names]
        if expr_expand is None:
            expr_expand = dict()
        self.expr_expand = expr_expand

    def __contains__(self, item):
        if item in self.var_names:
            return True
        if item in self.expr_expand:
            return True
        return False

    def _special_keys(self, key):
        if key == 'rho':
            return self['dens']
        if key == 'dens':
            return self['rho']
        if key[:3] == 'vel' and len(key) == 4:
            if 'mom' + key[3] in self:
                return self['mom' + key[3]] / self['dens']
        if key[:3] == 'mom' and len(key) == 4:
            if 'vel' + key[3] in self:
                return self['vel' + key[3]] * self['rho']
        if key in self.coord:
            return self.coord[key]
        if key in self.expr_expand:
            return self.expr_eval(self.expr_expand[key])
        raise KeyError

    def _parse_self(self, key):
        try:
            return self._special_keys(key)
        except (NotImplementedError, KeyError):
            pass
        if hasattr(self, key):
            try:
                out = getattr(self, key)()
                if out.shape == self.block_shape:
                    return out
            except (AttributeError, TypeError):
                pass
        raise KeyError

    def _getitem(self, key):
        try:
            return self.data[self.var_names.index(key)]
        except (IndexError, ValueError):
            raise KeyError

    def __getitem__(self, key):
        try:
            return self._getitem(key)
        except KeyError:
            pass
        try:
            return self._parse_self(key)
        except KeyError:
            pass
        if hasattr(self.coord, key):
            return getattr(self.coord, key)
        raise KeyError('Unable to parse {0:}'.format(key))

    def expr_eval(self, expr):
        if expr in self:
            return self[expr]
        expr = sp.sympify(expr)
        sym = [i for i in expr.atoms() if isinstance(i, sp.symbol.Symbol)]
        data = [self[str(i)] for i in sym]
        #subs = {i: self[str(i)] for i in expr.atoms() if type(i) == sp.symbol.Symbol}
        return sp.lambdify(sym, expr, "numpy")(*data)

    def locator(self, x3=None, x2=None, x1=None):
        snone = slice(None)
        if x3 is None:
            x3 = snone
        else:
            try:
                self.coord.x3v[x3]
            except IndexError:
                if self.coord.x3f[0] <= x3 <= self.coord.x3f[-1]:
                    x3 = np.abs(self.coord.x3v - x3).argmin()
                else:
                    x3 = snone
        if x2 is None:
            x2 = snone
        else:
            try:
                self.coord.x2v[x2]
            except IndexError:
                if self.coord.x2f[0] <= x2 <= self.coord.x2f[-1]:
                    x2 = np.abs(self.coord.x2v - x2).argmin()
                else:
                    x2 = snone
        if x1 is None:
            x1 = snone
        else:
            try:
                self.coord.x1v[x1]
            except IndexError:
                if self.coord.x1f[0] <= x1 <= self.coord.x1f[-1]:
                    x1 = np.abs(self.coord.x1v - x1).argmin()
                else:
                    x1 = snone
        return x3, x2, x1

    def imshow(self, var, loc, opt=None, use_cart=True):
        if opt is None:
            opt = dict()
        loc = self.locator(*loc)
        snone = slice(None)
        if loc.count(snone) == 3:
            return None
        data = self.expr_eval(var)[loc]
        tmp = [loc[i] == snone for i in range(3)]
        dim = tmp.index(False)
        if use_cart:
            coord = self.coord.plot_cart_face(dim)
        else:
            coord = self.coord.faces()
        coord = [i[loc] for i in coord]
        coord.pop(dim)
        x, y = coord
        return plt.pcolormesh(x, y, data, **opt)


class _AxisMean(object):
    def __init__(self, owner, axis, var):
        self.owner = owner
        self.axis = axis
        self.var = var
        self._xf = np.array([owner.xf[axis].min(), owner.xf[axis].max()])
        self._xv = np.array([self._xf.mean()])
        self.logloc = owner.logloc
        sequence = list(self.logloc.max(axis=0))
        sequence.pop(axis)
        axes = [0, 1, 2]
        axes.pop(axis)
        self._axes = axes
        a, b = owner.xf[axes[0]], owner.xf[axes[1]]
        self._extent = [[a[i, 0], a[i, -1], b[i, 0], b[i, -1]] for i in range(a.shape[0])]
        extent = []
        for i in self._extent:
            if i not in extent:
                extent.append(i)
        self.extent = extent
        self.nb = self.logloc.shape[0]
        self.ns = len(sequence)
        self._index = 0

    def __getitem__(self, index):
        loc = [i for i in range(len(self._extent)) if self._extent[i] == self.extent[index]]
        data = 0
        n = 0
        for i in loc:
            data += self.owner[i].expr_eval(self.var).mean(axis=self.axis)
            n += 1
        if not n:
            return None
        data /= n
        data = np.expand_dims(np.expand_dims(data, self.axis), 0)
        xf = [i[loc[0]] for i in self.owner.xf]
        xf[self.axis] = self._xf
        xv = [i[loc[0]] for i in self.owner.xv]
        xv[self.axis] = self._xv
        logloc = self.logloc[loc[0]]
        logloc[self.axis] = 0
        out = Block(index, data, [self.var], xf, xv, logloc,
                    coord=self.owner.coord, expr_expand=self.owner.expr_expand)
        out.nmean = n
        return out

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        out = None
        try:
            while out is None:
                out = self[self._index]
                self._index += 1
        except (IndexError, ValueError):
            raise StopIteration
        return out


class BlockByBlock(object):
    def __init__(self, fn, sim_path=None, ai_data=None, sim=None, x2_face=None,
                 num_ghost=0, coord=None, expr_expand=None, time_unit=None):
        if x2_face is None and sim is not None:
            x2_face = sim.x2_face
        if sim_path:
            if not os.path.isfile(fn):
                fn = os.path.join(sim_path, fn)
        self.fn = fn
        self._prefix = '.'.join((os.path.split(fn)[-1]).split('.')[:-1])
        hdf5 = h5py.File(fn)
        self.hdf5 = hdf5
        self.t = hdf5.attrs['Time']
        self.logloc = hdf5['LogicalLocations'][:]
        keys = list(hdf5.keys())
        key = [i for i in ['hydro', 'cons', 'prim'] if i in keys][0]
        self.data = hdf5[key]
        self.xf = [hdf5['x' + i + 'f'][:] for i in '321']
        self.xv = [hdf5['x' + i + 'v'][:] for i in '321']
        na = np.newaxis
        # self._xf = [self.x3f[:, :, na, na], self.x2f[:, na, :, na], self.x1f[:, na, na, :]]
        # self._xv = [self.x3v[:, :, na, na], self.x2v[:, na, :, na], self.x1v[:, na, na, :]]
        self.var_names = list(hdf5.attrs['VariableNames'])
        self.nb = hdf5.attrs['NumMeshBlocks']
        self._index = 0
        self.coord = coord
        self.expr_expand = expr_expand
        if time_unit is None:
            time_unit = [1.0, 'sim units']
        self.time_unit = time_unit

    def __getitem__(self, index):
        args = [index, self.data[:, index], self.var_names, [i[index] for i in self.xf],
                [i[index] for i in self.xv], self.logloc[index]]
        kwargs = dict(coord=self.coord, expr_expand=self.expr_expand)
        return Block(*args, **kwargs)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        try:
            out = self[self._index]
        except (IndexError, ValueError):
            raise StopIteration
        self._index += 1
        return out

    def _default_title(self):
        try:
            out = self.sim.name
        except AttributeError:
            pass
        out = os.path.split(self.fn)[-1]
        out += r' $t = {0:g}$ ({1:s})'.format(self.t * self.time_unit[0],
                                              self.time_unit[1])
        return helpers.sanitize_lbl(out)

    def mean(self, axis, var):
        return _AxisMean(self, axis, var)

    def plot_slice(self, var, pos, axis=3, save=False, fn=None, fig=None, ax=None,
                   fig_opt=None, dpi=None, figsize=None, log=False, popt=None, cb=True,
                   cbl=None, aspect=1, title=None, lbls=True, vmin=None, vmax=None,
                   ext='png', sdir=None, use_cart=True, ax_lbl_add='', cmap=None,
                   zerocent=False):
        if pos is None:
            pos = 'mean'
        _axes = [2, 1, 0]
        _i = _axes[axis - 1]
        _axes.remove(3 - axis)
        loc = [None] * 3
        loc[3 - axis] = pos
        if popt is None:
            popt = dict()
        norm = None
        if log:
            norm = LogNorm()
        if zerocent:
            if cmap is None:
                cmap = NCcmap
            if vmin is not None or vmax is not None:
                if vmin is None:
                    vmin = vmax
                if vmax is None:
                    vmax = vmin
                vmax = np.abs([vmin, vmax]).max()
                vmin = -vmax
        if 'cmap' not in popt:
            popt['cmap'] = cmap

        _popt = {'norm': norm, 'vmin': vmin, 'vmax': vmax}
        _popt.update(popt)
        if title is None:
            title = self._default_title()

        if fig is None and ax is None:
            _fopt = {'dpi': dpi, 'figsize': figsize}
            if fig_opt is None:
                fig_opt = {}
            _fopt.update(fig_opt)
            fig = plt.figure(**_fopt)
        if ax is not None:
            plt.sca(ax)
        im = None
        vlim = [np.inf, -np.inf]
        do_lim = None in [vmin, vmax]
        images = []
        if pos == 'mean':
            iterator = self.mean(3 - axis, var)
            loc[3 - axis] = self.xf[3 - axis].mean()
        else:
            iterator = self
        for block in iterator:
            tmp = block.imshow(var, loc, _popt, use_cart=use_cart)
            if tmp is not None:
                im = tmp
                images.append(im)
                if do_lim:
                    tmp = im.get_clim()
                    vlim[0] = min(tmp[0], vlim[0])
                    vlim[1] = max(tmp[1], vlim[1])
        ax = plt.gca()
        if do_lim:
            if zerocent:
                tmp = np.abs(vlim).max()
                vlim = [-tmp, tmp]
            for im in images:
                im.set_clim(*vlim)
        if aspect is not None:
            ax.set_aspect(aspect)
        if cb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            if cbl is None:
                try:
                    if str(var) == var:
                        cbl = helpers.sanitize_lbl(var)
                except TypeError:
                    pass
            if cbl:
                cb.set_label(cbl)
            # helpers.constrain_cb(ax, cb)
            plt.sca(ax)
        if title:
            tmp = dict()
            tmp.update(self.__dict__)
            plt.title(helpers.sanitize_lbl(title.format(**tmp)))
        if lbls:
            if use_cart:
                plt.xlabel(rc('coord_lbls')["cartesian"][_axes[0]] + ax_lbl_add)
                plt.ylabel(rc('coord_lbls')["cartesian"][_axes[1]] + ax_lbl_add)
            else:
                plt.xlabel(rc('coord_lbls')[self.coord][_axes[0]] + ax_lbl_add)
                plt.ylabel(rc('coord_lbls')[self.coord][_axes[1]] + ax_lbl_add)

        if save or fn:
            def_fn = self._prefix
            try:
                if str(var) == var:
                    def_fn += '_' + var
            except TypeError:
                pass
            def_fn += '_plot.' + ext
            _save_fig(fn, def_fn, sdir=sdir)
        return ax, cb
