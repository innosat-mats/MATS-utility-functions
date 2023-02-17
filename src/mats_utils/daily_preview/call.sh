#!/bin/bash

# main path
MATS_dir='/home/waves/projects/MATS/'

# options 
level="1a"
version="0.4"
snippet="True"

# output
dates=$(date +'%Y_%m_%d')
outdir=${MATS_dir}'animations/daily/'$dates'/'

echo -e 'DAILY ANIMATION: Initiating ...'
echo 'MATS DATA: L'${level}' data (v. '${version}')..'
echo 'Generating figures in: '${outdir}

# generate figures
{ python get_daily.py --version ${version} --outdir ${outdir} --level ${level} --snippet; } &
pid=$!
wait $pid

echo 'Generating animation .....'

# generate animation
{ ffmpeg -stream_loop 0 -r 60 -i ${outdir}"/ALL/%d.png" -vcodec libx264 -crf 0 ${outdir}${dates}.mp4; } &
pid=$!
wait $pid

echo 'Removing figures .....'

# delete images in outdir/ALL/
rm -r ${outdir}"/ALL/"

echo 'Uploading video .....'

# upload video
{ python upload.py --file=${outdir}${dates}".mp4"
                   --title=${dates}
                   --description="MATS test upload"
                   --keywords="MATS"
                   --category="22"
                   --privacyStatus="private"; }

echo 'End of program .....'