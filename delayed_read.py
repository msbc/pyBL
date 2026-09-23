"""
Read Athena++ output data files.
"""

# Python modules
import re
import struct
import sys
import warnings
from io import open  # Consistent binary I/O from Python 2 and 3

# Other Python modules
import numpy as np

# Load HDF5 reader
try:
    import h5py
except ImportError:
    pass


class athdf(dict):
    """Lazily read variables from an Athena++ ``.athdf`` file.

    The object behaves like a dictionary of field arrays, but delays reading
    individual quantities until they are requested. Coordinate arrays and
    file metadata are loaded during initialization; call :meth:`load_all` to
    materialize every selected quantity.

    Parameters
    ----------
    filename : path-like
        Path to an Athena++ ``.athdf`` file.
    data : dict, optional
        Existing mapping to populate instead of creating a new result.
    quantities : sequence of str, optional
        Field names to make available. If omitted, all file quantities are
        available for lazy loading.
    dtype : numpy.dtype, default=numpy.float32
        Data type used for loaded field arrays.
    level : int, optional
        AMR level to load. Defaults to the maximum level in the file.
    return_levels : bool, default=False
        Include AMR level information in the result.
    subsample : bool, default=False
        Select data at the requested level without restriction averaging.
    fast_restrict : bool, default=False
        Use the faster AMR restriction path.
    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max : float, optional
        Bounds of the region to load in each coordinate direction.
    vol_func : callable, optional
        Cell-volume function used for volume-weighted restriction.
    vol_params : tuple, optional
        Additional parameters for ``vol_func``.
    face_func_1, face_func_2, face_func_3 : callable, optional
        Coordinate-face transformation functions.
    center_func_1, center_func_2, center_func_3 : callable, optional
        Cell-center functions for each coordinate direction.

    Notes
    -----
    ``h5py`` is required when constructing an instance. Accessing a field
    through ``obj[name]`` loads that field and caches it in the mapping.
    """

    # Initialization
    def __init__(
            self,
            filename,
            data=None,
            quantities=None,
            dtype=np.float32,
            level=None,
            return_levels=False,
            subsample=False,
            fast_restrict=False,
            x1_min=None,
            x1_max=None,
            x2_min=None,
            x2_max=None,
            x3_min=None,
            x3_max=None,
            vol_func=None,
            vol_params=None,
            face_func_1=None,
            face_func_2=None,
            face_func_3=None,
            center_func_1=None,
            center_func_2=None,
            center_func_3=None):

        # Import necessary module for reading HDF5 files
        try:
            h5py
        except NameError:
            raise ImportError(
                    'athdf could not be executed because h5py could not be imported.')

        # Prepare dictionary for results
        if data is None:
            data = {}
            self.new_data = True
            self._existing_keys = []
        else:
            self.new_data = False
            self._existing_keys = list(data.keys())
        super(athdf, self).__init__(data)
        self.filename = filename
        self.quantities = quantities
        self.dtype = dtype
        self.level = level
        self.return_levels = return_levels
        self.subsample = subsample
        self.fast_restrict = fast_restrict
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_min = x2_min
        self.x2_max = x2_max
        self.x3_min = x3_min
        self.x3_max = x3_max
        self.vol_func = vol_func
        self.vol_params = vol_params
        self.face_func_1 = face_func_1
        self.face_func_2 = face_func_2
        self.face_func_3 = face_func_3
        self.center_func_1 = center_func_1
        self.center_func_2 = center_func_2
        self.center_func_3 = center_func_3

        # Open file
        with h5py.File(filename, 'r') as f:

            # Extract size information
            self.max_level = f.attrs['MaxLevel']
            if self.level is None:
                self.level = self.max_level
            self.block_size = f.attrs['MeshBlockSize']
            root_grid_size = f.attrs['RootGridSize']
            self.levels = f['Levels'][:]
            self.logical_locations = f['LogicalLocations'][:]

            # Determine sums and slices
            nx_vals = []
            for d in range(3):

                # Sum or slice
                if self.block_size[d] == 1 and root_grid_size[d] > 1:
                    other_locations = [
                            location for location in zip(
                                self.levels,
                                self.logical_locations[:, (d+1) % 3],
                                self.logical_locations[:, (d+2) % 3])]

                    # Effective slice
                    if len(set(other_locations)) == len(other_locations):
                        nx_vals.append(1)

                    # Nontrivial sum
                    else:
                        num_blocks_this_dim = 0
                        for (level_this_dim, loc_this_dim) in zip(
                                self.levels, self.logical_locations[:, d]):
                            if level_this_dim <= self.level:
                                num_blocks_this_dim = max(
                                        num_blocks_this_dim, (loc_this_dim + 1)
                                        * 2 ** (self.level-level_this_dim))
                            else:
                                num_blocks_this_dim = max(
                                        num_blocks_this_dim, (loc_this_dim + 1)
                                        / 2 ** (level_this_dim-self.level))
                        nx_vals.append(num_blocks_this_dim)

                # Singleton dimension
                elif self.block_size[d] == 1:
                    nx_vals.append(1)

                # Normal case
                else:
                    nx_vals.append(root_grid_size[d] * 2 ** self.level)

            # Set dimensions
            self.nx1 = nx_vals[0]
            self.nx2 = nx_vals[1]
            self.nx3 = nx_vals[2]
            self.lx1 = self.nx1 / self.block_size[0]
            self.lx2 = self.nx2 / self.block_size[1]
            self.lx3 = self.nx3 / self.block_size[2]
            self.num_extended_dims = 0
            for nx in nx_vals:
                if nx > 1:
                    self.num_extended_dims += 1

            # Set volume function for preset coordinates if needed
            coord = f.attrs['Coordinates'].decode('ascii', 'replace')
            if (self.level < self.max_level and not self.subsample
                    and not self.fast_restrict and self.vol_func is None):
                x1_rat = f.attrs['RootGridX1'][2].decode('ascii', 'replace')
                x2_rat = f.attrs['RootGridX2'][2].decode('ascii', 'replace')
                x3_rat = f.attrs['RootGridX3'][2].decode('ascii', 'replace')
                if (coord == 'cartesian' or coord == 'minkowski' or coord == 'tilted'
                        or coord == 'sinusoidal'):
                    if (
                            (self.nx1 == 1 or x1_rat == 1.0)
                            and (self.nx2 == 1 or x2_rat == 1.0)
                            and (self.nx3 == 1 or x3_rat == 1.0)):
                        self.fast_restrict = True
                    else:
                        self.vol_func = lambda xm, xp, ym, yp, zm, zp: (
                                xp - xm) * (yp - ym) * (zp - zm)
                elif coord == 'cylindrical':
                    if (self.nx1 == 1 and (self.nx2 == 1 or x2_rat == 1.0)
                            and (self.nx3 == 1 or x3_rat == 1.0)):
                        self.fast_restrict = True
                    else:
                        self.vol_func = lambda rm, rp, phim, phip, zm, zp: (
                                rp**2 - rm**2) * (phip - phim) * (zp - zm)
                elif coord == 'spherical_polar' or coord == 'schwarzschild':
                    if self.nx1 == 1 and self.nx2 == 1 and (
                            self.nx3 == 1 or x3_rat == 1.0):
                        self.fast_restrict = True
                    else:
                        self.vol_func = lambda rm, rp, thetam, thetap, phim, phip: ((
                                rp**3 - rm**3) * abs(np.cos(thetam) - np.cos(thetap))
                                * (phip - phim))
                elif coord == 'kerr-schild':
                    if self.nx1 == 1 and self.nx2 == 1 and (
                            self.nx3 == 1 or x3_rat == 1.0):
                        self.fast_restrict = True
                    else:
                        a = vol_params[0]

                        def vol_func(rm, rp, thetam, thetap, phim, phip):
                            cosm = np.cos(thetam)
                            cosp = np.cos(thetap)
                            return ((rp**3 - rm**3) * abs(cosm - cosp) + a**2 *
                                    (rp - rm) * abs(cosm**3 - cosp**3)) * (phip - phim)
                else:
                    raise AthenaError('Coordinates not recognized')

            # Set cell center functions for preset coordinates
            if center_func_1 is None:
                if (coord == 'cartesian' or coord == 'minkowski' or coord == 'tilted'
                        or coord == 'sinusoidal' or coord == 'kerr-schild'):
                    def center_func_1(xm, xp): return 0.5 * (xm + xp)
                elif coord == 'cylindrical':
                    def center_func_1(xm, xp): return (
                            2.0/3.0 * (xp**3 - xm**3) / (xp**2 - xm**2))
                elif coord == 'spherical_polar':
                    def center_func_1(xm, xp): return (
                            3.0/4.0 * (xp**4 - xm**4) / (xp**3 - xm**3))
                elif coord == 'schwarzschild':
                    def center_func_1(xm, xp): return (0.5 * (xm**3 + xp**3)) ** (1.0/3.0)
                else:
                    raise AthenaError('Coordinates not recognized')
            if center_func_2 is None:
                if (coord == 'cartesian' or coord == 'cylindrical' or coord == 'minkowski'
                        or coord == 'tilted' or coord == 'sinusoidal'
                        or coord == 'kerr-schild'):
                    def center_func_2(xm, xp): return 0.5 * (xm + xp)
                elif coord == 'spherical_polar':
                    def center_func_2(xm, xp):
                        sm = np.sin(xm)
                        cm = np.cos(xm)
                        sp = np.sin(xp)
                        cp = np.cos(xp)
                        return (sp - xp*cp - sm + xm*cm) / (cm-cp)
                elif coord == 'schwarzschild':
                    def center_func_2(xm, xp): return np.arccos(
                            0.5 * (np.cos(xm) + np.cos(xp)))
                else:
                    raise AthenaError('Coordinates not recognized')
            if center_func_3 is None:
                if (coord == 'cartesian' or coord == 'cylindrical'
                        or coord == 'spherical_polar' or coord == 'minkowski'
                        or coord == 'tilted' or coord == 'sinusoidal'
                        or coord == 'schwarzschild' or coord == 'kerr-schild'):
                    def center_func_3(xm, xp): return 0.5 * (xm + xp)
                else:
                    raise AthenaError('Coordinates not recognized')

            # Check output level compared to max level in file
            if (self.level < self.max_level and not self.subsample
                    and not self.fast_restrict):
                warnings.warn(
                        'Exact restriction being used: performance severely affected;' +
                        ' see documentation',
                        AthenaWarning)
                sys.stderr.flush()
            if self.level > self.max_level:
                warnings.warn(
                        'Requested refinement level higher than maximum level in file:' +
                        ' all cells will be prolongated',
                        AthenaWarning)
                sys.stderr.flush()

            # Check that subsampling and/or fast restriction will work if needed
            if self.level < self.max_level and (self.subsample or self.fast_restrict):
                max_restrict_factor = 2 ** (self.max_level - self.level)
                for current_block_size in self.block_size:
                    if (current_block_size != 1 and current_block_size %
                            max_restrict_factor != 0):
                        raise AthenaError(
                                'Block boundaries at finest level must be cell' +
                                ' boundaries at desired level for subsampling or fast' +
                                ' restriction to work')

            # Create list of all quantities if none given
            var_quantities = np.array([x.decode('ascii', 'replace')
                                       for x in f.attrs['VariableNames'][:]])
            coord_quantities = ('x1f', 'x2f', 'x3f', 'x1v', 'x2v', 'x3v')
            attr_quantities = [key for key in f.attrs]
            other_quantities = ('Levels',)
            if not self.new_data:
                self.quantities = self.values()
            elif self.quantities is None:
                self.quantities = var_quantities
            else:
                for q in self.quantities:
                    if q not in var_quantities and q not in coord_quantities:
                        possibilities = '", "'.join(var_quantities)
                        possibilities = '"' + possibilities + '"'
                        error_string = (
                                'Quantity not recognized: file does not include "{0}" but'
                                ' does include {1}')
                        raise AthenaError(error_string.format(q, possibilities))
            self.quantities = [str(q) for q in self.quantities
                               if q not in coord_quantities
                               and q not in attr_quantities
                               and q not in other_quantities]

            # Store file attribute metadata
            for key in attr_quantities:
                self[str(key)] = f.attrs[key]

            # Get metadata describing file layout
            self.num_blocks = f.attrs['NumMeshBlocks']
            dataset_names = np.array([x.decode('ascii', 'replace')
                                      for x in f.attrs['DatasetNames'][:]])
            dataset_sizes = f.attrs['NumVariables'][:]
            dataset_sizes_cumulative = np.cumsum(dataset_sizes)
            variable_names = np.array([x.decode('ascii', 'replace')
                                       for x in f.attrs['VariableNames'][:]])
            self.quantity_datasets = {}
            self.quantity_indices = {}
            for q in self.quantities:
                var_num = np.where(variable_names == q)[0][0]
                dataset_num = np.where(dataset_sizes_cumulative > var_num)[0][0]
                if dataset_num == 0:
                    dataset_index = var_num
                else:
                    dataset_index = var_num - dataset_sizes_cumulative[dataset_num-1]
                self.quantity_datasets[q] = dataset_names[dataset_num]
                self.quantity_indices[q] = dataset_index

            # Locate fine block for coordinates in case of slice
            fine_block = np.where(self.levels == self.max_level)[0][0]
            self.x1m = f['x1f'][fine_block, 0]
            self.x1p = f['x1f'][fine_block, 1]
            self.x2m = f['x2f'][fine_block, 0]
            self.x2p = f['x2f'][fine_block, 1]
            self.x3m = f['x3f'][fine_block, 0]
            self.x3p = f['x3f'][fine_block, 1]

            # Populate coordinate arrays
            face_funcs = (face_func_1, face_func_2, face_func_3)
            center_funcs = (center_func_1, center_func_2, center_func_3)
            for d, nx, face_func, center_func in zip(
                    range(1, 4), nx_vals, face_funcs, center_funcs):
                xf = 'x' + repr(d) + 'f'
                xv = 'x' + repr(d) + 'v'
                if nx == 1:
                    xm = (self.x1m, self.x2m, self.x3m)[d-1]
                    xp = (self.x1p, self.x2p, self.x3p)[d-1]
                    self[xf] = np.array([xm, xp])
                else:
                    xmin = f.attrs['RootGridX'+repr(d)][0]
                    xmax = f.attrs['RootGridX'+repr(d)][1]
                    xrat_root = f.attrs['RootGridX'+repr(d)][2]
                    if xrat_root == -1.0 and face_func is None:
                        raise AthenaError(
                                'Must specify user-defined face_func_{0}'.format(d))
                    elif face_func is not None:
                        self[xf] = face_func(xmin, xmax, xrat_root, nx + 1)
                    elif xrat_root == 1.0:
                        try:
                            if np.all(self.levels == self.level):
                                self[xf] = np.empty(nx + 1)
                                for n_block in range(int(nx / self.block_size[d-1])):
                                    sample_block = np.where(self.logical_locations[:, d-1]
                                                            == n_block)[0][0]
                                    index_low = n_block * self.block_size[d-1]
                                    index_high = index_low + self.block_size[d-1] + 1
                                    self[xf][index_low:index_high] = f[xf][sample_block, :]
                            else:
                                self[xf] = np.linspace(xmin, xmax, nx + 1)
                        except IndexError:
                            self[xf] = np.linspace(xmin, xmax, nx + 1)
                    else:
                        xrat = xrat_root ** (1.0 / 2 ** self.level)
                        self[xf] = (
                                xmin + (1.0 - xrat**np.arange(nx+1)) / (1.0 - xrat**nx)
                                * (xmax - xmin))
                self[xv] = np.empty(nx)
                for i in range(nx):
                    self[xv][i] = center_func(self[xf][i], self[xf][i+1])

            # Account for selection
            x1_select = False
            x2_select = False
            x3_select = False
            self.i_min = self.j_min = self.k_min = 0
            self.i_max = self.nx1
            self.j_max = self.nx2
            self.k_max = self.nx3
            error_string = '{0} must be {1} than {2} in order to intersect data range'
            if x1_min is not None and x1_min >= self['x1f'][1]:
                if x1_min >= self['x1f'][-1]:
                    raise AthenaError(error_string.format(
                            'x1_min', 'less', self['x1f'][-1]))
                x1_select = True
                self.i_min = np.where(self['x1f'] <= x1_min)[0][-1]
            if x1_max is not None and x1_max <= self['x1f'][-2]:
                if x1_max <= self['x1f'][0]:
                    raise AthenaError(
                            error_string.format('x1_max', 'greater', self['x1f'][0]))
                x1_select = True
                self.i_max = np.where(self['x1f'] >= x1_max)[0][0]
            if x2_min is not None and x2_min >= self['x2f'][1]:
                if x2_min >= self['x2f'][-1]:
                    raise AthenaError(error_string.format(
                            'x2_min', 'less', self['x2f'][-1]))
                x2_select = True
                self.j_min = np.where(self['x2f'] <= x2_min)[0][-1]
            if x2_max is not None and x2_max <= self['x2f'][-2]:
                if x2_max <= self['x2f'][0]:
                    raise AthenaError(
                            error_string.format('x2_max', 'greater', self['x2f'][0]))
                x2_select = True
                self.j_max = np.where(self['x2f'] >= x2_max)[0][0]
            if x3_min is not None and x3_min >= self['x3f'][1]:
                if x3_min >= self['x3f'][-1]:
                    raise AthenaError(error_string.format(
                            'x3_min', 'less', self['x3f'][-1]))
                x3_select = True
                self.k_min = np.where(self['x3f'] <= x3_min)[0][-1]
            if x3_max is not None and x3_max <= self['x3f'][-2]:
                if x3_max <= self['x3f'][0]:
                    raise AthenaError(
                            error_string.format('x3_max', 'greater', self['x3f'][0]))
                x3_select = True
                self.k_max = np.where(self['x3f'] >= x3_max)[0][0]

            # Adjust coordinates if selection made
            if x1_select:
                self['x1f'] = self['x1f'][self.i_min:self.i_max+1]
                self['x1v'] = self['x1v'][self.i_min:self.i_max]
            if x2_select:
                self['x2f'] = self['x2f'][self.j_min:self.j_max+1]
                self['x2v'] = self['x2v'][self.j_min:self.j_max]
            if x3_select:
                self['x3f'] = self['x3f'][self.k_min:self.k_max+1]
                self['x3v'] = self['x3v'][self.k_min:self.k_max]

            # Set to None any quantities not set by initialization
            for i in self.quantities:
                if i not in self:
                    self[i] = None

        if self.return_levels:
            self._get_levels()

    def _get_levels(self):
        self['Levels'] = np.ones((self._shape()), dtype=np.int32) * -1
        for block_num in range(self.num_blocks):
            block_level = self.levels[block_num]
            block_location = self.logical_locations[block_num, :]

            # Prolongate coarse data and copy same-level data
            if block_level <= self.level:

                # Calculate scale (number of copies per dimension)
                s = 2 ** (self.level - block_level)

                # Calculate destination indices, without selection
                il_d = block_location[0] * self.block_size[0] * s if self.nx1 > 1 else 0
                jl_d = block_location[1] * self.block_size[1] * s if self.nx2 > 1 else 0
                kl_d = block_location[2] * self.block_size[2] * s if self.nx3 > 1 else 0
                iu_d = il_d + self.block_size[0] * s if self.nx1 > 1 else 1
                ju_d = jl_d + self.block_size[1] * s if self.nx2 > 1 else 1
                ku_d = kl_d + self.block_size[2] * s if self.nx3 > 1 else 1

                # Calculate (prolongated) source indices, with selection
                il_s = max(il_d, self.i_min) - il_d
                jl_s = max(jl_d, self.j_min) - jl_d
                kl_s = max(kl_d, self.k_min) - kl_d
                iu_s = min(iu_d, self.i_max) - il_d
                ju_s = min(ju_d, self.j_max) - jl_d
                ku_s = min(ku_d, self.k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue

                # Account for selection in destination indices
                il_d = max(il_d, self.i_min) - self.i_min
                jl_d = max(jl_d, self.j_min) - self.j_min
                kl_d = max(kl_d, self.k_min) - self.k_min
                iu_d = min(iu_d, self.i_max) - self.i_min
                ju_d = min(ju_d, self.j_max) - self.j_min
                ku_d = min(ku_d, self.k_max) - self.k_min

            # Restrict fine data
            else:
                # Calculate scale
                s = 2 ** (block_level - self.level)

                # Calculate destination indices, without selection
                il_d = block_location[0] * self.block_size[0] / s if self.nx1 > 1 else 0
                jl_d = block_location[1] * self.block_size[1] / s if self.nx2 > 1 else 0
                kl_d = block_location[2] * self.block_size[2] / s if self.nx3 > 1 else 0
                iu_d = il_d + self.block_size[0] / s if self.nx1 > 1 else 1
                ju_d = jl_d + self.block_size[1] / s if self.nx2 > 1 else 1
                ku_d = kl_d + self.block_size[2] / s if self.nx3 > 1 else 1

                # Calculate (restricted) source indices, with selection
                il_s = max(il_d, self.i_min) - il_d
                jl_s = max(jl_d, self.j_min) - jl_d
                kl_s = max(kl_d, self.k_min) - kl_d
                iu_s = min(iu_d, self.i_max) - il_d
                ju_s = min(ju_d, self.j_max) - jl_d
                ku_s = min(ku_d, self.k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue

                # Account for selection in destination indices
                il_d = max(il_d, self.i_min) - self.i_min
                jl_d = max(jl_d, self.j_min) - self.j_min
                kl_d = max(kl_d, self.k_min) - self.k_min
                iu_d = min(iu_d, self.i_max) - self.i_min
                ju_d = min(ju_d, self.j_max) - self.j_min
                ku_d = min(ku_d, self.k_max) - self.k_min

            # Set level information for cells in this block
            self['Levels'][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_level

    # Function for contingently accessing data
    def __getitem__(self, item):
        if self._need_to_read(item):
            self._grab_quantities([item])
        try:
            return super(athdf, self).__getitem__(item)
        except KeyError:
            if item == 'Levels':
                self._get_levels()
                return super(athdf, self).__getitem__(item)

    # Function for contingently setting data
    def __setitem__(self, key, value):
        if not self.new_data:
            try:
                self._existing_keys.remove(key)
            except ValueError:
                pass
        return super(athdf, self).__setitem__(key, value)

    # Function for returning constant shape of all 3D arrays
    def _shape(self):
        return (self.k_max - self.k_min, self.j_max - self.j_min, self.i_max - self.i_min)

    # Function for determining if a quantity must be set
    def _need_to_read(self, quantity):
        if not self.new_data:
            if quantity in self._existing_keys:
                return True
        try:
            if super(athdf, self).__getitem__(quantity) is None:
                return True
        except KeyError:
            if quantity in self.quantities:
                return True
        return False

    def load_all(self):
        """Load and cache every available field quantity."""
        self._grab_quantities([i for i in self.keys() if self._need_to_read(i)])

    # Function for setting all needed quantities
    def _grab_quantities(self, quantities):

        # Create list of quantities to be set
        quantities = [q for q in quantities if self._need_to_read(q)]

        # Open file
        with h5py.File(self.filename, 'r') as f:

            # Prepare arrays for data and bookkeeping
            if self.new_data:
                for q in quantities:
                    self[q] = np.zeros((self._shape()), dtype=self.dtype)
            else:
                for q in quantities:
                    self[q].fill(0.0)
            if (not self.subsample and not self.fast_restrict
                    and self.max_level > self.level):
                restricted_data = np.zeros((self.lx3, self.lx2, self.lx1), dtype=bool)

            # Go through blocks in data file
            for block_num in range(self.num_blocks):

                # Extract location information
                block_level = self.levels[block_num]
                block_location = self.logical_locations[block_num, :]

                # Prolongate coarse data and copy same-level data
                if block_level <= self.level:

                    # Calculate scale (number of copies per dimension)
                    s = 2 ** (self.level - block_level)

                    # Calculate destination indices, without selection
                    il_d = (block_location[0] *
                            self.block_size[0] * s) if self.nx1 > 1 else 0
                    jl_d = (block_location[1] *
                            self.block_size[1] * s) if self.nx2 > 1 else 0
                    kl_d = (block_location[2] *
                            self.block_size[2] * s) if self.nx3 > 1 else 0
                    iu_d = il_d + self.block_size[0] * s if self.nx1 > 1 else 1
                    ju_d = jl_d + self.block_size[1] * s if self.nx2 > 1 else 1
                    ku_d = kl_d + self.block_size[2] * s if self.nx3 > 1 else 1

                    # Calculate (prolongated) source indices, with selection
                    il_s = max(il_d, self.i_min) - il_d
                    jl_s = max(jl_d, self.j_min) - jl_d
                    kl_s = max(kl_d, self.k_min) - kl_d
                    iu_s = min(iu_d, self.i_max) - il_d
                    ju_s = min(ju_d, self.j_max) - jl_d
                    ku_s = min(ku_d, self.k_max) - kl_d
                    if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                        continue

                    # Account for selection in destination indices
                    il_d = max(il_d, self.i_min) - self.i_min
                    jl_d = max(jl_d, self.j_min) - self.j_min
                    kl_d = max(kl_d, self.k_min) - self.k_min
                    iu_d = min(iu_d, self.i_max) - self.i_min
                    ju_d = min(ju_d, self.j_max) - self.j_min
                    ku_d = min(ku_d, self.k_max) - self.k_min

                    # Assign values
                    for q in quantities:
                        dataset = self.quantity_datasets[q]
                        index = self.quantity_indices[q]
                        block_data = f[dataset][index, block_num, :]
                        if s > 1:
                            if self.nx1 > 1:
                                block_data = np.repeat(block_data, s, axis=2)
                            if self.nx2 > 1:
                                block_data = np.repeat(block_data, s, axis=1)
                            if self.nx3 > 1:
                                block_data = np.repeat(block_data, s, axis=0)
                        self[q][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = (
                                block_data[kl_s:ku_s, jl_s:ju_s, il_s:iu_s])

                # Restrict fine data
                else:

                    # Calculate scale
                    s = 2 ** (block_level - self.level)

                    # Calculate destination indices, without selection
                    il_d = (block_location[0] *
                            self.block_size[0] / s) if self.nx1 > 1 else 0
                    jl_d = (block_location[1] *
                            self.block_size[1] / s) if self.nx2 > 1 else 0
                    kl_d = (block_location[2] *
                            self.block_size[2] / s) if self.nx3 > 1 else 0
                    iu_d = il_d + self.block_size[0] / s if self.nx1 > 1 else 1
                    ju_d = jl_d + self.block_size[1] / s if self.nx2 > 1 else 1
                    ku_d = kl_d + self.block_size[2] / s if self.nx3 > 1 else 1

                    # Calculate (restricted) source indices, with selection
                    il_s = max(il_d, self.i_min) - il_d
                    jl_s = max(jl_d, self.j_min) - jl_d
                    kl_s = max(kl_d, self.k_min) - kl_d
                    iu_s = min(iu_d, self.i_max) - il_d
                    ju_s = min(ju_d, self.j_max) - jl_d
                    ku_s = min(ku_d, self.k_max) - kl_d
                    if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                        continue

                    # Account for selection in destination indices
                    il_d = max(il_d, self.i_min) - self.i_min
                    jl_d = max(jl_d, self.j_min) - self.j_min
                    kl_d = max(kl_d, self.k_min) - self.k_min
                    iu_d = min(iu_d, self.i_max) - self.i_min
                    ju_d = min(ju_d, self.j_max) - self.j_min
                    ku_d = min(ku_d, self.k_max) - self.k_min

                    # Account for restriction in source indices
                    if self.nx1 > 1:
                        il_s *= s
                        iu_s *= s
                    if self.nx2 > 1:
                        jl_s *= s
                        ju_s *= s
                    if self.nx3 > 1:
                        kl_s *= s
                        ku_s *= s

                    # Apply subsampling
                    if self.subsample:

                        # Calculate fine-level offsets (nearest cell at or below center)
                        o1 = s/2 - 1 if self.nx1 > 1 else 0
                        o2 = s/2 - 1 if self.nx2 > 1 else 0
                        o3 = s/2 - 1 if self.nx3 > 1 else 0

                        # Assign values
                        for q in quantities:
                            dataset = self.quantity_datasets[q]
                            index = self.quantity_indices[q]
                            self[q][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = (f[dataset][
                                index, block_num, kl_s+o3:ku_s:s,
                                jl_s+o2:ju_s:s, il_s+o1:iu_s:s]
                            )

                    # Apply fast (uniform Cartesian) restriction
                    elif self.fast_restrict:

                        # Calculate fine-level offsets
                        io_vals = range(s) if self.nx1 > 1 else (0,)
                        jo_vals = range(s) if self.nx2 > 1 else (0,)
                        ko_vals = range(s) if self.nx3 > 1 else (0,)

                        # Assign values
                        for q in quantities:
                            dataset = self.quantity_datasets[q]
                            index = self.quantity_indices[q]
                            for ko in ko_vals:
                                for jo in jo_vals:
                                    for io in io_vals:
                                        self[q][kl_d:ku_d,
                                                jl_d:ju_d,
                                                il_d:iu_d] += f[dataset][
                                                    index, block_num,
                                                    kl_s+ko:ku_s:s,
                                                    jl_s+jo:ju_s:s,
                                                    il_s+io:iu_s:s]
                            self[q][kl_d:ku_d, jl_d:ju_d,
                                    il_d:iu_d] /= s ** self.num_extended_dims

                    # Apply exact (volume-weighted) restriction
                    else:

                        # Calculate sets of indices
                        i_s_vals = range(il_s, iu_s)
                        j_s_vals = range(jl_s, ju_s)
                        k_s_vals = range(kl_s, ku_s)
                        i_d_vals = range(il_d, iu_d)
                        j_d_vals = range(jl_d, ju_d)
                        k_d_vals = range(kl_d, ku_d)
                        if self.nx1 > 1:
                            i_d_vals = np.repeat(i_d_vals, s)
                        if self.nx2 > 1:
                            j_d_vals = np.repeat(j_d_vals, s)
                        if self.nx3 > 1:
                            k_d_vals = np.repeat(k_d_vals, s)

                        # Accumulate values
                        for k_s, k_d in zip(k_s_vals, k_d_vals):
                            if self.nx3 > 1:
                                self.x3m = f['x3f'][block_num, k_s]
                                self.x3p = f['x3f'][block_num, k_s+1]
                            for j_s, j_d in zip(j_s_vals, j_d_vals):
                                if self.nx2 > 1:
                                    self.x2m = f['x2f'][block_num, j_s]
                                    self.x2p = f['x2f'][block_num, j_s+1]
                                for i_s, i_d in zip(i_s_vals, i_d_vals):
                                    if self.nx1 > 1:
                                        self.x1m = f['x1f'][block_num, i_s]
                                        self.x1p = f['x1f'][block_num, i_s+1]
                                    vol = self.vol_func(
                                            self.x1m, self.x1p, self.x2m, self.x2p,
                                            self.x3m, self.x3p)
                                    for q in quantities:
                                        dataset = self.quantity_datasets[q]
                                        index = self.quantity_indices[q]
                                        self[q][k_d, j_d, i_d] += (
                                            vol
                                            * f[dataset][index, block_num, k_s, j_s, i_s])
                        loc1 = (self.nx1 > 1) * block_location[0] / s
                        loc2 = (self.nx2 > 1) * block_location[1] / s
                        loc3 = (self.nx3 > 1) * block_location[2] / s
                        restricted_data[loc3, loc2, loc1] = True

        # Remove volume factors from restricted data
        if self.level < self.max_level and not self.subsample and not self.fast_restrict:
            for loc3 in range(self.lx3):
                for loc2 in range(self.lx2):
                    for loc1 in range(self.lx1):
                        if restricted_data[loc3, loc2, loc1]:
                            il = loc1 * self.block_size[0]
                            jl = loc2 * self.block_size[1]
                            kl = loc3 * self.block_size[2]
                            iu = il + self.block_size[0]
                            ju = jl + self.block_size[1]
                            ku = kl + self.block_size[2]
                            il = max(il, self.i_min) - self.i_min
                            jl = max(jl, self.j_min) - self.j_min
                            kl = max(kl, self.k_min) - self.k_min
                            iu = min(iu, self.i_max) - self.i_min
                            ju = min(ju, self.j_max) - self.j_min
                            ku = min(ku, self.k_max) - self.k_min
                            for k in range(kl, ku):
                                if self.nx3 > 1:
                                    self.x3m = self['x3f'][k]
                                    self.x3p = self['x3f'][k+1]
                                for j in range(jl, ju):
                                    if self.nx2 > 1:
                                        self.x2m = self['x2f'][j]
                                        self.x2p = self['x2f'][j+1]
                                    for i in range(il, iu):
                                        if self.nx1 > 1:
                                            self.x1m = self['x1f'][i]
                                            self.x1p = self['x1f'][i+1]
                                        vol = self.vol_func(
                                                self.x1m, self.x1p, self.x2m, self.x2p,
                                                self.x3m, self.x3p)
                                        for q in quantities:
                                            self[q][k, j, i] /= vol


class AthenaError(RuntimeError):
    """General exception class for Athena++ read functions."""
    pass


class AthenaWarning(RuntimeWarning):
    """General warning class for Athena++ read functions."""
    pass
