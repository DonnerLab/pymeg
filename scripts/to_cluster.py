'''
Run a simple script on one cluster node!
'''
import argparse
import os
import subprocess

def submit(walltime, cores, memory, cwd, script):
    command = '''
    #!/bin/bash
    # walltime: defines maximum lifetime of a job
    # nodes/ppn: how many nodes (usually 1)? how many cores?

    #PBS -q batch
    #PBS -l walltime=%i:00:00
    #PBS -l nodes=1:ppn=%s
    #PBS -l mem=%igb
    #PBS -l nodes=node041
    #PBS -N niklas
    #### set journal & error options
    #PBS -o /home/nwilming/scratch/$PBS_JOBID.o
    #PBS -e /home/nwilming/scratch/$PBS_JOBID.e

    # -- run in the current working (submission) directory --
    cd %s
    chmod g+wx cluster

    # FILE TO EXECUTE
    %s 1> %s/cluster/$PBS_JOBID.out 2> %s/cluster/$PBS_JOBID.err
    '''%(walltime, cores, memory, cwd, script, cwd, cwd)
    tmp = file('to_cluster.sh', 'w')
    tmp.write(command)
    tmp.close()

    subprocess.call("qsub to_cluster.sh", shell=True)


if __name__=='__main__':
    # 2. Create workers
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help='Which command to run?')
    parser.add_argument("--cores", help="How many cores do you need?",
                        type=int, default=1)
    parser.add_argument("--memory", help="How many GB ram?", default=12, type=int)
    parser.add_argument("--walltime", help="Walltime (h)?", default=15, type=int)
    parser.add_argument("--array", help="Map scripts function to workers?", default=False, action="store_true")
    parser.add_argument("--nosubmit", dest='submit', help="Do not submit to torque", default=True, action="store_false")

    parser.add_argument("--name", help="Give a name to this job", default='cluster')
    parser.add_argument("--redo-older", dest='older',
        help="Redo all tasks where results are older than this %Y%m%d (e.g. 20160825) date. Only effective for array jobs.",
        type=str, default='now')

    args = parser.parse_args()
    if not args.array:
        if args.submit:
            submit(args.walltime, args.cores, args.memory, os.getcwd(), args.script)
    else:
        '''
        Need to build script to execute.
        '''
        import tempfile
        pyfile = args.script.replace('.py', '')
        mod = __import__(pyfile)
        tmpdir = tempfile.mkdtemp(prefix=args.name+'_temp', dir=os.getcwd())
        for x in mod.list_tasks(args.older):
            if type(x) == str:
                x = '"'+x+'"'
            f =  tempfile.NamedTemporaryFile(dir=tmpdir, suffix='.py', delete=False)
            script = """
print 'Parameters:', '%s'
print '%s'
import %s
%s.execute(%s)
            """%(str(x), f.name, pyfile, pyfile, str(x))
            command = 'python ' + f.name
            f.write(script)
            f.close()
            subprocess.call(['chmod', '-R', 'a+rwx', tmpdir])
            if args.submit:
                submit(args.walltime, args.cores, args.memory, os.getcwd(), command)
