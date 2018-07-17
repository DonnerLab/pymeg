#!/bin/sh
qsub -v subid=$1 -v nifti=$2 -v subjectdir=$3 reconall.sh

