#! /usr/bin/env python

import sys
import os
import matplotlib as mpl
mpl.use('agg')
import pyBL.simBLclass as bl

try:
    pd = float(sys.argv[1])
except IndexError:
    pd = 0
sim = os.path.abspath('../')
sim = bl.BLsim(sim)
print sim, sim.path
title = r'$t=${t:.03f}, $\Omega_p=$' + '%.3g' % pd
cbl = r'$rv_r\sqrt{\Sigma}$'
sim.speed_shift(pd, base='cons', data='Rpseudo', title=title, cbl=cbl, vmax=.1)
