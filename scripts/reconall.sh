#!/bin/sh

echo '#!/bin/sh

#PBS -q batch
#PBS -l walltime=500:00:00
#PBS -l nodes=1:ppn=1
#PBS -l pmem=14gbs

# -- run in the current working (submission) directory --
cd $PBS_O_WORKDIR

chmod g=wx $PBS_JOBNAME

export SUBJECTS_DIR=${subjectdir}

recon-all -subject ${subid} -i ${nifti} -all 1> "$PBS_JOBID"1.out 2> "$PBS_JOBID"1.err' >> _reconall.sh

qsub -v subid=$1 -v nifti=$2 -v subjectdir=$3 _reconall.sh

