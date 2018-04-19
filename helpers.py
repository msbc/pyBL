#! /usr/bin/env python2

#import __main__ as main
import numpy as np
import matplotlib
import sys
import os
import re
from subprocess import call
#import glob
from scipy.stats import scoreatpercentile as percentile
#from scipy.stats import linregress
#from scipy.optimize import curve_fit
#import scipy.signal
#from itertools import groupby
#from operator import itemgetter
#import time
from .NCcmap import NCcmap
try:
    from astropy.convolution.convolve import convolve_fft
except ImportError:
    print('Warning, cannot load "convolve_fft" from astropy. Using scipy equivlent which uses zero padding.')
    from scipy.signal import fftconvolve

def labeler(name):
    sub = None
    for vec in ['mom', 'vel', 'mag', 'B']:
        if name[:len(vec)] == vec:
            try:
                i = int(name[len(vec):])
                if i == 1:
                    sub = 'r'
                elif i == 2:
                    sub = r'\phi'
                else:
                    raise TypeError
                name = vec
            except TypeError:
                pass
    try:
        out = {'vel': 'v', 'mom': r'\rho v', 'mag': 'B', 'dens': r'\rho',
               'pres': 'P', 'Rpseudo': r'rv_r\sqrt{\rho}', 'pseudo': r'v_r\sqrt{\rho}',
               'vorticity': r'\omega_z', 'vortensity': r'\omega_z/\rho',
               'vi': r'$r^2\nabla\times\left(v_r,\delta v_\phi\right)$',
               've': r'$r^2\nabla\times\left(v_r,\delta v_\phi\right)/\rho$',
               }[name]
        if sub:
            out += '_' + sub
        return '$' + out + '$'
    except KeyError:
        return sanitize_lbl(name)

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

def fftcorrelate(a, b):
  '''Usage: fftcorrelate(a, b)
  Uses fast Fourier transforms to correlate 'a' and 'b'.'''

  a = np.conj(np.flipud(a))
  assert(np.isfinite(a).all())
  assert(np.isfinite(b).all())
  try:
      out = convolve_fft(a, b, boundary='wrap', normalize_kernel=False, nan_treatment='fill')
  except NameError:
      out = scipy.signal.fftconvolve(a, b, mode='same')
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

def smooth(x,window_len=11,window='hanning'):
    #From http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[(window_len+1)//2-1:0:-1],x,x[-2:-(window_len+1)//2-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(), s, mode='valid')
    return y

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def mkmov(fnames, out=None, cmd=None, **kwargs):
    _lin = ['avconv', 'ffmpeg']
    _mac = ['avconvert']
    _cmds = _lin + _mac
    if out is None:
        i = 0
        tmp = ['movie', '', '.mp4']
        while os.path.isfile(sum(tmp,'')):
            tmp[1] = '{0:05d}'.format(i)
            i += 1
    while cmd is None and _cmds:
        tmp = _cmds.pop(0)
        if which(tmp):
            cmd = tmp
    if cmd is None:
        raise RuntimeError('Cannot find executable.')
    if cmd in _lin:
        _opt = dict(i=fnames, vcodec='mpeg4')
    elif cmd in _mac:
        raise NotImplementedError
    _opt.update(kwargs)
    query = map(str, filter(None, sum(map(list, _opt.items()), [cmd])))
    call(query)

def dir_mtime(dir):
    '''Usage : dir_mtime(dir)
    Gets the time of the most recently modified file in the given directory.'''
    out = [0, os.path.getmtime(dir)]
    out.extend([os.path.getmtime(os.path.join(dir, i)) for i in os.listdir(dir)])
    return max(out)