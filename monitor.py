#! /usr/bin/env python

_Send_To = 'mcoleman@ias.edu'  # your email address

import time
import os
import smtplib
import socket
import mimetypes
from email import encoders
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from optparse import OptionParser
import numpy as np
import getpass
from glob import glob
import warnings


warnings.filterwarnings("ignore")
_from = getpass.getuser() + '@' + socket.gethostname()

_queue_cmds = ['qstat', 'squeue']
_queue_opt = {'squeue': '-o "%i %P %j %u %t %M"', 'qstat': '-r'}
_queue_user = {'qstat': '-u {user:}', 'squeue': '-u {user:}'}
_remap_head = {'jobid': 'job-id', 'st': 'state', 's': 'state',
               'time-use': 'runtime', 'time': 'runtime', 'Full jobname': 'name'}
_two_word_head = [i.lower() for i in ['job id', 'time use', 'slots ja-task-id']]

np.seterr(divide='ignore')


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


def _setupMPL():
    #import pylab
    import matplotlib as mpl
    mpl.use('agg')
    mpl.rcParams['savefig.dpi'] = 300


class searchDirs(object):
    def __init__(self, solo_dirs=None, multi_dirs=None):
        if solo_dirs is None:
            solo_dirs = []
        if multi_dirs is None:
            multi_dirs = []
        self.solo_dirs = solo_dirs
        self.multi_dirs = multi_dirs
        self._search_dirs = None

    def abspath(self, *dirs):
        return [os.path.abspath(os.path.expanduser(i)) for i in dirs]

    def expand_multi(self, dirs=None):
        if dirs is None:
            dirs = self.abspath(*self.multi_dirs)
        out = dirs[:]
        for dir in dirs:
            out += [i for i in glob(os.path.join(dir, '*')) if os.path.isdir(i)]
        return out

    @property
    def paths(self):
        if self._search_dirs is not None:
            return self._search_dirs[:]
        out = self.abspath(*self.solo_dirs) + self.expand_multi()
        self._search_dirs = out
        return out[:]


BLdirs = searchDirs(['~/BLayer', '~/BLayer/fft_tests'], ['~/data/bl'])


def choose_job_type(job, **opt):
    jname = job['name'].rstrip('.sh')
    if 0:
        for d in ['~/zeus_rad/data/', '~/wd/', '~/nodust', '~/am_cvn/']:
            tmp = os.path.join(os.path.expanduser(d), jname)
            if os.path.isdir(tmp):
                #print 'Zeus:', tmp, d, os.path.join(os.path.expanduser(d), jname)
                return zeusJob(job, **opt)
    if not 'ext' in opt:
        opt['ext'] = 'png'
    for d in BLdirs.paths:
        tmp = os.path.abspath(os.path.join(d, jname))
        #print tmp, os.path.expanduser(d)
        if os.path.isdir(tmp):
            job['name'] = tmp
            return athenaBL(job, **opt)
    if 'BLayer' in jname:
        return athenaBL(job, **opt)
    #print job
    #return nonJob(job, **opt)
    return athenaBL(job, **opt)


class _job(dict):
    '''Base class for super computing jobs. Missing function gen_update.'''
    def __init__(self, data, run_test=False, ext='.pdf'):
        super(_job, self).__init__(data)
        self.data = dict(data)
        if run_test:
            self.run_test()
        self.name = self['name']
        self._subject = "Update for simulation " + self.name
        self.ext = ext

    def __repr__(self):
        return '<{0:}: {1:}>'.format(self.__class__.__name__, self.name)

    def gen_update(self):
        msg = 'This function must be implimented in a child class of ' + repr(_job)
        raise NotImplementedError(msg)

    def runtime(self):
        '''Returns runtime in seconds.'''
        if 'runtime' in self:
            rt =map(int, self['runtime'].split(':'))
            tmp = [86400, 3600, 60, 1]
            while len(tmp) > len(rt):
                tmp.pop(0)
            tmp = np.array(tmp)
            rt = np.array(rt)
            return int(np.sum(tmp * rt))
        if 'start' in self:
            start = self['start']
        else:
            start = self['submit/start'].split('/')
            start = [start[2], start[0], start[1]]
        if 'at' in self:
            start += self['at'].split(':')
        start = map(int, start)
        start += [-1] * (9 - len(start))
        start = time.mktime(start)
        return int(time.mktime(time.localtime()) - start)

    def updateQ(self):
        '''Detemine we should send an update for this job.'''
        if self['state'].lower()[0] != 'r':
            return False
        hours = self.runtime() // 3600
        if hours == 0:
            return False
        if hours % 12 == 11:
            return True
        if hours in [1, 6]:
            return True
        try:
            wt = map(int, self['walltime'].split(':'))
            if hours >= wt[-3] - 1:
                return True
        except (KeyError, IndexError):
            pass
        return False

    def send_file(self, fn=None, subject=None):
        print('Sending update for {0:}.'.format(self))
        if fn is None:
            fn = self.gen_update()
            print('Generated update file "{0:}".'.format(fn))
        if not subject:
            subject = self._subject
        opt = {}
        if subject:
            opt['subject'] = subject
        email_file(_Send_To, fn, **opt)
        return fn

    def run_test(self):
        if self.updateQ():
            self.send_file()


class zeusJob(_job):
    '''Class for Zeus shearing-box sims'''
    def gen_update(self):
        import read_sim as rs # only import if needed to reduce cpu time
        sim = rs.zeussim(self.name.rstrip('.sh'))
        fn = os.path.join(sim.simd, sim.name + '_diag' + self.ext)
        mk_plot = False
        if not os.path.isfile(fn):
            mk_plot = True
        elif rs.dir_mtime(os.path.join(sim.simd, 'hist')) > os.path.getmtime(fn):
            mk_plot = True
        if mk_plot:
            sim.diagnostic(fn=fn)
        return fn


class athenaBL(_job):
    '''Class for Athena++ BL sims'''
    def gen_update(self):
        import matplotlib as mpl
        mpl.use('agg')
        import pyBL.simBLclass as bl
        bl._quiet = True
        sim = bl.BLsim(self.name.rstrip('.sh'))
        return sim.diagnostic(save=True, ext=self.ext)


class nonJob(_job):
    def gen_update(self):
      return None


def _parse_head(head):
    head = head.lower().strip()
    head = head.split()
    for i in _two_word_head:
        a, b = i.split()
        if a in head:
            i = head.index(a)
            if i + 1 < len(head):
                if head[i + 1] == b:
                    head.pop(i + 1)
                    head[i] += '-' + b
    return [_remap_head.get(i, i) for i in head]


def parse_queue(cmds=None, user=getpass.getuser(), usr_trunc=None, job_opt=None):
    if job_opt is None:
        job_opt = {}
    if cmds is None:
        cmds = [i for i in _queue_cmds if which(i)]
    cmds = np.atleast_1d(cmds)
    if cmds.size == 0:
        raise RuntimeError('Cannot locate queue commands {0:}'.format(cmds))
    out = []
    for cmd in cmds:
        line_cmd = cmd
        if cmd in _queue_opt:
            line_cmd += ' ' + _queue_opt[cmd]
        if user and cmd in _queue_user:
            line_cmd += ' ' + _queue_user[cmd].format(user=user)
        with os.popen(line_cmd) as queue:
            lines = queue.readlines()
        try:
            head = lines.pop(0)
            while not lines[0].strip().strip('-'):
                lines.pop(0)
        except IndexError:
            return []
        head = _parse_head(head)
        lines = filter(None, [line.rstrip() for line in lines])
        for line in lines:
            if line[0] in [' ', '\t']:
                tmp = [i.strip() for i in line.split(':')]
                out[-1][_remap_head.get(tmp[0], tmp[0])] = tmp[1]
            else:
                out.append(dict(zip(head, line.split())))
    if user:
        usr = user[:usr_trunc]
        out = [i for i in out if i['user'] == usr]
    return [choose_job_type(i, **job_opt) for i in out]


def email_file(to, path, subject='Automated python scripted email', preamble=None, sender=_from):
    outer = MIMEMultipart()
    outer['Subject'] = subject
    outer['From'] = sender
    outer['To'] = to
    if preamble is None:
        preamble = subject
    outer.preamble = preamble

    if path:
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None:
            # No guess could be made, or the file is encoded (compressed), so
            # use a generic bag-of-bits type.
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        if maintype == 'text':
            fp = open(path)
            # Note: we should handle calculating the charset
            msg = MIMEText(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == 'image':
            fp = open(path, 'rb')
            msg = MIMEImage(fp.read(), _subtype=subtype)
            fp.close()
        elif maintype == 'audio':
            fp = open(path, 'rb')
            msg = MIMEAudio(fp.read(), _subtype=subtype)
            fp.close()
        else:
            fp = open(path, 'rb')
            msg = MIMEBase(maintype, subtype)
            msg.set_payload(fp.read())
            fp.close()
            # Encode the payload using Base64
            encoders.encode_base64(msg)
        # Set the filename parameter
        filename = path.split('/')[-1]
        msg.add_header('Content-Disposition', 'attachment', filename=filename)
        outer.attach(msg)

    try:
        s = smtplib.SMTP('localhost')
        s.sendmail(sender, to, outer.as_string())
        s.quit()
    except socket.error:
        print('Socket Error')
        s = smtplib.SMTP(host='smtp.gmail.com', port=587)
        s.sendmail(sender, to, outer.as_string())
        s.quit()

    return None


def monitor(ext=None, force=False, debug=False):
    opt = {}
    # make sure extension is formatted correctly
    if ext:
        if ext[0] != '.' :
            ext = '.' + ext
        opt['ext'] = ext
    for job in parse_queue(job_opt=opt):
        if force:
            _setupMPL()
            job.send_file()
        else:
            if job.updateQ:
                _setupMPL()
            job.run_test()
    return None


def email_diag(sims, ext=None):
    opt = {}
    # make sure extension is formatted correctly
    if ext:
        if ext[0] != '.':
            ext = '.' + ext
        opt['ext'] = ext
    for sim in sims:
        if '/' in sim:
            sim = os.path.abspath(os.path.expanduser(sim))
        job = choose_job_type({'name': sim}, **opt)
        print(job)
        job.send_file()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--force", action="store_true", dest="force", default=False,
                      help="Force update of all running simulations.")
    parser.add_option("-d", "--debug", action="store_true", dest="debug", default=False,
                      help="Enable debugging messages.")

    (opt, args) = parser.parse_args()
    if opt.debug:
        print('parsed like a boss')

    if args:
        email_diag(args)
    else:
        monitor(force=opt.force, debug=opt.debug)

