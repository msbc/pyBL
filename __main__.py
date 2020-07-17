import argparse
import os
from glob import glob
import numpy as np

if __name__ == '__main__':
    np.warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--maps',
                        default=False,
                        action='store_true',
                        help='Create maps of pertibations in the simulations.')
    parser.add_argument('--stripes',
                        default=False,
                        action='store_true',
                        help='Create strpes of pertibations in the simulations.')
    parser.add_argument('-p', '--path',
                        type=str,
                        default='~/data/pleiades_data/bl',
                        help='Path to search for simulations')
    parser.add_argument('-o', '--overwrite',
                        default=False,
                        action='store_true',
                        help='Overwrite existing plots')
    parser.add_argument('-q', '--quiet',
                        default=None,
                        action='store_true',
                        help='Minimize output to stdout')
    parser.add_argument('-a',
                        default=None,
                        action='store_true',
                        help='Only print args')
    parser.add_argument('-t', '--tbl',
                        default=None,
                        action='store_true',
                        help='Print simulation table (and nothing else)')
    parser.add_argument('--no_flux',
                        dest='flux',
                        default=True,
                        action='store_false',
                        help='Do not plot flux series')
    parser.add_argument('--no_vort',
                        dest='vort_prof',
                        default=True,
                        action='store_false',
                        help='Do not plot vorticity profiles')
    parser.add_argument('--no_prof',
                        dest='prof',
                        default=True,
                        action='store_false',
                        help='Do not plot profiles')
    parser.add_argument('-i',
                        type=int,
                        default=None,
                        help='Simulation index')
    parser.add_argument('-s', '--sim',
                        type=str,
                        default=None,
                        help='Simulation Name')
    parser.add_argument('--skip',
                        action='store_true',
                        default=False,
                        help='Skip data gen (if possible)')
    args = parser.parse_args()

    try:
        import git
        git.Repo(os.path.expanduser(os.path.split(__file__)[0])).remote().pull()
    except:
        print('git pull FAILED!')

    if args.a:
        print(vars(args))
    elif args.tbl:
        from .simBLclass.CompileSims import mk_sim_tbl
        print(mk_sim_tbl())
    else:
        _skip = args.skip
        path = os.path.expanduser(args.path)
        quiet = args.quiet
        if args.sim is None:
            tmp = os.path.join(path, 'M{0:}.*')
            d = '[0-9]'
            print(tmp.format(d))
            sims = sorted(glob(tmp.format(d)) + glob(tmp.format(d * 2)))
            if args.i is not None:
                sims = [sims[args.i]]
        else:
            tmp = os.path.join(path, args.sim)
            sims = sorted(glob(tmp))
        print(sims)
        if not sims:
            raise RuntimeError('No sims found.')
        if len(sims) > 1:
            if quiet is None:
                quiet = True
            from .simBLclass.CompileSims import mkplots
            mkplots(sims, maps=args.maps, overwrite=args.overwrite, fluxes=args.flux)
        else:
            if quiet is None:
                quiet = False
            opt = dict(quiet=quiet, working_dir=True, maps=args.maps,
                       overwrite=args.overwrite, fluxes=args.flux,
                       vort_prof=args.vort_prof, prof=args.prof, stripes=args.stripes)
            from .simBLclass.simBLclass import BLsim
            BLsim(sims[0], skip_data_gen=args.skip).main_plots(**opt)
