#!/usr/bin/env python
'''
Parallelize functions with simple arguments via torque
'''

import errno
import os
import subprocess
import tempfile


def submit(walltime, memory, cwd, tmpdir,
           script, name, nodes='1:ppn=1',
           shellfname=None, env=None, ssh_to='node028'):
    '''
    Submit a script to torque
    '''

    cmd_top = '''
    #!/bin/bash
    # walltime: defines maximum lifetime of a job
    # nodes/ppn: how many nodes (usually 1)? how many cores?

    #PBS -q batch
    #PBS -l walltime={walltime}:00:00
    #PBS -l nodes={nodes}
    #PBS -l mem={memory}gb
    #PBS -N {name}

    cd {cwd}
    mkdir -p cluster
    chmod a+rwx cluster

    #### set journal & error options
    #PBS -o {cwd}/$PBS_JOBID.o
    #PBS -e {cwd}/$PBS_JOBID.e

    '''.format(**{'walltime': walltime,
                  'nodes': nodes,
                  'memory': memory,
                  'cwd': cwd,
                  'script': script,
                  'name': name})

    if env is not None:
        cmd_top += 'source activate %s\n' % env

    cmd_bottom = '''
    # FILE TO EXECUTE
    {script} 1> {cwd}/$PBS_JOBID.out 2> {cwd}/$PBS_JOBID.err
    '''.format(**{'walltime': walltime,
                  'nodes': nodes,
                  'memory': memory,
                  'cwd': cwd,
                  'script': script,
                  'name': name})
    command = cmd_top + cmd_bottom
    with tempfile.NamedTemporaryFile(delete=False, dir=tmpdir,
                                     prefix='delete_me_tmp') as shellfname:
        shellfname.write(command)
        shellfname = shellfname.name
    if ssh_to is None:
        "qsub %s" % (shellfname)
    else:
        "ssh %s 'qsub %s'" % (ssh_to, shellfname)
    output = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT,
        shell=True)
    return output


def to_script(func, tmpdir, *args):
    '''
    Write a simple stub python function that calls this function.
    '''

    with tempfile.NamedTemporaryFile(delete=False, dir=tmpdir,
                                     prefix='delete_me_tmp') as script:
        code = """
print('Parameters:', '%s', '%s')
from %s import %s
%s(*%s)
        """ % (str(args).replace("'", ''), func.__name__,
               func.__module__, func.__name__,
               func.__name__, str(args))
        script.write(code)
        return script.name


def pmap(func, args, walltime=12, memory=10, logdir=None, tmpdir=None,
         name=None, nodes='1:ppn=1', verbose=True, env=None, ssh_to='node028'):
    if name is None:
        name = func.__name__
    if logdir is None:
        from os.path import expanduser, join
        home = expanduser("~")
        logdir = join(home, 'cluster_logs', func.__name__)
        mkdir_p(logdir)
    if tmpdir is None:
        from os.path import expanduser, join
        home = expanduser("~")
        tmpdir = join(home, 'cluster_logs', 'tmp')
        mkdir_p(tmpdir)
    out = []
    for arg in args:
        script = 'ipython ' + to_script(func, tmpdir, *arg)
        if verbose:
            print(arg, '->', script)
        pid = submit(walltime, memory, logdir, tmpdir, script, name, env=env)
        out.append(pid)
    return out


def status(pid):
    output = subprocess.check_output(
        "ssh node028 'qstat %s'" % pid.replace('\n', ''),
        stderr=subprocess.STDOUT,
        shell=True)
    if " C " in output.split('\n')[-2]:
        return True
    elif " E " in output.split('\n')[-2]:
        raise RuntimeError('Job %s failed')
    else:
        return False


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
