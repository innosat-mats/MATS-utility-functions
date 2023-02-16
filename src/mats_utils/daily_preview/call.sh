#!/bin/bash

# main path
MATS_dir='/home/waves/projects/MATS/'

# options 
level='1a'
version='0.4'
outdir=${MATS_dir}'animations/daily/'

# generate figures
{ python get_daily.py --version ${version} --outdir ${outdir} --level ${level} } &
pid=$!

# wait while process runs
wait $pid

# generate animation


# wait while process runs

# delete images in outdir/ALL