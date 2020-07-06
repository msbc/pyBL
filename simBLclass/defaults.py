import os

_dirs = ['', '~/', '~/Dropbox/dev/pyBL', '/scratch/mcoleman/2d_bl_lc',
         '/scratch/gpfs/sashaph/BLayer', '/perseus/scratch/gpfs/sashaph/BLayer',
         '~/BLayer', '~/BLayer/fft_tests', '~/BLayer/2d_new/', '~/archive', '~/data/bl',
         '~/data/pleiades_data/bl']
_dirs = list(map(os.path.expanduser, _dirs))
_dirs += [os.path.join(d, 'Mach8stampede') for d in _dirs]
_dirs = [i for i in _dirs if os.path.isdir(i)]

_fig_base_dir = '/data/mcoleman/pleiades_data/bl/figs'
if not os.path.isdir(_fig_base_dir):
    _fig_base_dir = ''

_defaults = dict(
    ff = .5,
    rename_lc = False,
    skip = False,
    quiet = False,
    data_base = '/scratch/gpfs/sashaph/BLayer',
    ext = ['athdf', 'npy', 'hdf5'],
    pre = ['BL', 'disk', 'mock'],
    seed_type = {'r': 'block-random', 'random': 'globally random',
                 'mix': 'block-phased-mixed', 'prime': 'prime modes'},
    dirs = _dirs,
    int_fmt = '%5.5d',
    fig_base_dir = _fig_base_dir,
    rl = {5: .83, 6: .84, 7: .86, 8: .88, 9: .90, 10: .92, 11: .92, 12: .92, 13: .93,
          14: .93, 15: .84},
    ru = {15: 1.04, 14: 1.08, 13: 1.13, 12: 1.14, 11: 1.07, 10: 1.16, 9: 1.2, 8: 1.3,
           7: 1.4, 6: 1.4, 5: 1.4},
    coord_lbls = {'cartesian': ['$x$', '$y$', '$z$'][::-1],
                   'spherical_polar': ['$r$', r'$\theta$', r'$\phi$'][::-1],
                   'cylindrical': [r'$\varpi$', r'$\phi$', r'$z$'][::-1]}
)


class Parameters(dict):
    """Class providing parameters (dict subclass)"""
    def __init__(self, copy=None, config=None, token=None, roasts=None):
        super().__init__()
        if copy is not None:
            self.update(copy)

    def __call__(self, key, default=None):
        """Make this callable, allowing us to avoid/handle key errors"""
        if hasattr(self, key):
            attr = getattr(self, key)
            if callable(attr):
                return attr()
            return attr
        return self.dget(key, default)

    def dget(self, key, default=None):
        """Like get, but check the defaults first"""
        return self.get(key, _defaults.get(key, default))

    @property
    def file_fmts(self):
        return ['.'.join([a, 'out' + str(b), self('int_fmt'), c])
                for a in self('pre') for b in range(1, 5) for c in self('ext')]


rc = Parameters(_defaults)
