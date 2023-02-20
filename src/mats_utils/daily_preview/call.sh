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
{ python get_daily.py --version ${version} --outdir ${outdir} --level ${level}; } &
pid=$!
wait $pid

# in dates, there are now subdirectories.
# $dates/part0;part1; etc; move them together 
count=$(find ${outdir} -maxdepth 1 -mindepth 1 -type d | wc -l)
mkdir -p ${outdir}"/temp"

for number in $(seq 0 $count);
do
    echo $number
    mv ${outdir}/part${number}/ALL/*.png ${outdir}/temp
    sleep 1
    rm -R ${outdir}/part${number}/

done

# generate animation
{ ffmpeg -stream_loop 0 -r 60 -i ${outdir}"/temp/%d.png" -vcodec libx264 -crf 0 ${outdir}${dates}.mp4; } &
pid=$!
wait $pid

# upload video
echo 'Uploading video .....'

 { python upload.py --file=${file} --title=${dates} --description="2 DAYS AGO" --keywords="MATS" --category="22" --privacyStatus="private"; }

echo 'End of program .....'