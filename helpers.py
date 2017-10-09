#! /usr/bin/env python2

#import __main__ as main
import numpy as np
import matplotlib
import sys
import os
#import re
#import glob
from scipy.stats import scoreatpercentile as percentile
#from scipy.stats import linregress
#from scipy.optimize import curve_fit
#import scipy.signal
#from itertools import groupby
#from operator import itemgetter
#import time
from NCcmap import NCcmap

def sanitize_lbl(name):
  '''sanitize_lbl(label):
  Makes labels latex friendly.'''
  out = name.split('$')
  for i in range(0, len(out), 2):
    out[i] = re.sub('_', '\_', out[i])
  return '$'.join(out)

def pert_str(name):
  name = name.strip('$')
  if name[0] in ['T']:
    return r'$\delta ' + name + r'/ \left<' + name + r'\right>$'
  else:
    return r'$\delta\! ' + name + r'/ \left<' + name + r'\right>$'

def c_correlate(a, b):
  '''Usage: c_correlate(a, b)
  Inputs 'a' and 'b' should be arrays of uniformly temporally (time) spaced data.
  Returns normalized cross correlation. 'a' variations leading (occurring before)
  'b' variations results in negative offset.

  Function based on built-in IDL function of the same name.'''

  amean = np.mean(a)
  bmean = np.mean(b)
  da = a - amean
  db = b - bmean
  norm = np.sqrt(np.sum(da*da)) * np.sqrt(np.sum(db*db))
  out = fftcorrelate(da, db)/norm
  return out

def fftcorrelate(a, b, mode='same'):
  '''Usage: fftcorrelate(a, b, mode='same')
  Uses fast Fourier transforms to correlate 'a' and 'b'.

  See scipy.signal.fftconvolve documentation for info on optional keyword 'mode'.'''
  a = np.conj(np.flipud(a))
  out = scipy.signal.fftconvolve(a, b, mode=mode)
  return np.flipud(out)

def update_progress(progress):
  '''Usage : update_progress(progress)
  An ASCII progression bar. Values of input should range from 0 to 1.'''
  barLength = 50 # Modify this to change the length of the progress bar
  status = ""
  if isinstance(progress, int):
      progress = float(progress)
  if not isinstance(progress, float):
      progress = 0
      status = "error: progress var must be float\r\n"
  if progress < 0:
      progress = 0
      status = "Halt...\r\n"
  if progress >= 1:
      progress = 1
      status = "Done...\r\n"
  block = int(round(barLength*progress))
  text = "\rPercent: [{0}] {1}% {2}".format("#"*block + "-"*(barLength-block),
      "%.3f"% (progress*100), status)
  sys.stdout.write(text)
  sys.stdout.flush()

def typecast(data):
  '''Usage : typecast(data)
  Attempt to convert stings into objects of the appropriate type.

  Examples:
  typecast('data')   => 'data'
  typecast('3')      => 3 (int)
  typecast('3.4')    => 3.4 (float)
  typecast('3.4e1')  => 3.4 (float)
  typecast('3.4d1')  => 3.4 (float)
  typecast('"data"') => 'data'
  typecast("'data'") => 'data'
  '''
  if data[0] == data[-1] and data[0] in ["'", '"'] :
    return data[1:-1]
  try :
    return int(data)
  except ValueError :
    pass
  try :
    return float(data)
  except ValueError :
    pass
  try :
    # FORTRAN uses 'd' in place of 'e' to denote double instead of floats
    # (e.g. 1e2 = 1d2). Python doesn't like this, so fix it.
    return float(re.sub('d', 'e', data))
  except ValueError :
    pass
  return data

def round_to(n, precission):
  '''Usage: round_to(n, precission)
  Round n to nearest multiple of precission.'''
  correction = 0.5 if n >= 0 else -0.5
  return int(n/precission+correction)*precission

def smartlim(dat, low=10, high=90):#, axis=None):
  '''Usage : smartlim(dat, low=10, high=90):
  Uses percentiles low, high to get better limits for data then min, max for plotting.
  Keywords low (10), high (90) specify the percentiles to get the low and high limits.
  '''
  data = dat.flatten()
  ymin = data.min()#axis=axis)
  ymax = data.max()#axis=axis)
  yh = percentile(data, high)#, axis=axis)
  yl = percentile(data, low)#, axis=axis)
  if ymax > 0 :
    if ymax > 2*yh : ymax = yh
  else:
    if ymax > .5*yh : ymax = yh
  if ymin < 0 :
    if ymin < 2*yl : ymin = yl
  else:
    if ymin < .5*yl : ymin = yl
  return np.array([ymin, ymax])

def derivative(var, axis=0, dx=1.):
  '''Usage : derivative(var, axis=0, dx=1.)
  Take dirivatives along axis that are regularly spaced with spacing dx.'''

  loc = [slice(None)]*axis
  left = loc[:] # copy loc
  left.append(slice(None, -2))
  right = loc[:]
  right.append(slice(2, None))
  mid = loc[:]
  mid.append(slice(1, -1))

  out = np.empty(np.atleast_1d(var).shape)
  out[mid] = .5 * (var[right] - var[left])
  out[loc + [0]] = (var[loc + [1]] - var[loc + [0]])
  out[loc + [-1]] = (var[loc + [-1]] - var[loc + [-2]])

  return out / dx

def ndmesh(*args):
  '''Usage : ndmesh(*args)
  A generalization of numpy's meshgrid() to arbitrary dimension.'''
  args = map(np.asarray,args)
  return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i] for i, x in enumerate(args)])

def ndmeshT(*args):
  '''Usage : ndmeshT(*args)
  A generalization of numpy's meshgrid() to arbitrary dimension, with transposition added.'''
#  args = map(np.asarray,args)
#  return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i].T for i, x in enumerate(args)])
  return [i.T for i in ndmesh(*args)]

def eformat(f, prec=2, math=True):
  '''Usage : eformat(f, prec=2, math=True)
  Format floats into scientific notation with latex syntax and return as a string.

  Keywords:
    prec = int : This is the number of significant figures - 1
    math = bool : This sets whether to include '$' in the output.
  '''
  d = '$'
  if not math : d = ''
  s = "%.*e"%(prec, f)
  m, e = s.split('e')
  e = int(e)
  if prec >= 0 : return d+r'{0:s} \! \times \! 10^{{{1:d}}}'.format(m, e)+d
  if e != 0 : return d+r'10^{{{0:d}}}'.format(e)+d
  return d+'1'+d

def pow_format(x, exp, prec=2):
  '''Usage : pow_format(x, exp, prec=2)
  Format x^exp for latex.'''
  if exp == 0 :
    return ''
  if exp == 1 :
    return x
  if abs(exp) < 100 :
    if exp == int(exp):
      exp = str(int(exp))
    else :
      exp = (('{0:.' + str(prec) + 'f}').format(exp)).rstrip('0')
  else :
    exp = eformat(exp, prec=prec, math=False)
  return x + '^{' + exp + '}'

def isZeroCent(data):
  '''Usage : isZeroCent(data)
  Determine whether data should be considered zero centered.
  This function is mainly intended for plotting reasons,
  especially if a colorbar is used.
  '''
  data = np.nan_to_num(data)
  dmin = data.min()
  dmax = data.max()
  dave = data.mean()
  dstd = data.std()
  if (dmin*dmax < 0) :
    if (1./1.2 < -dmin/dmax < 1.2) : return True
    if (abs(dave/dmin) < .2) and (abs(dave/dmax) < .2) : return True
    if abs(dave/dstd) < .2 : return True
  return False
