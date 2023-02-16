#!/bin/bash

# main path
MATS_dir='/home/waves/projects/MATS/'

# options 
level=1a
version=0.4

# output
dates=$(date +'%Y_%m_%d')
outdir=${MATS_dir}'animations/daily/'$dates'/'

echo -e 'Generating animation ...'
echo 'L'${level}' data (v. '${version}')..'
echo 'Output to be saved in: '${outdir}

# generate figures
{ python get_daily.py --version ${version} --outdir ${outdir} --level ${level}; } &
pid=$!

# wait while process runs
#wait $pid

# generate animation
# { ffmpeg -stream_loop 0 -r 25 -i ${outdir}"/ALL/%d.png" -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ${outdir}anim.mp4 }

# wait while process runs

# delete images in outdir/ALL