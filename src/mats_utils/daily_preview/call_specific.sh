#!/bin/bash

# activate conda environment
source /home/waves/anaconda3/etc/profile.d/conda.sh
conda activate MATS_img

# main path

MATS_dir='/home/waves/projects/MATS/'

# options 
level="1b"
version="0.4"
snippet="True"


# output for naming
dates=$(date -d '2023-02-08' +'%Y_%m_%d')
#folder_str='2023-02-08'

outdir='/media/waves/AVAGO/data/MATS/animations/specific/'

for day_str in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do
    # start end
    end_day_str=$((day_str+1))

    start_date="2023-02-${day_str}"
    end_date="2023-02-${end_day_str}"

    folder_str=start_date

    # initiate
    echo -e "DAILY ANIMATION (${start_date} to ${end_date}): Initiating ..."
    echo 'MATS DATA: L'${level}' data (v'${version}')..'
    echo 'Generating figures in: '${outdir}

    # generate figures
    { python ${MATS_dir}MATS-utility-functions/src/mats_utils/daily_preview/get_daily.py --version ${version} --outdir ${outdir} --level ${level} --not_daily --start_date ${start_date} --end_date ${end_date}; } &
    pid=$!
    wait $pid

    # in dates, there are now subdirectories.
    # $dates/part0;part1; generate video from each;
    count=$(find ${outdir} -maxdepth 1 -mindepth 1 -type d | wc -l)
    count=$(($count-1))

    for folder in $outdir*/;
    do
        # image offset 
        start=8

        # for output name
        folder_name=${folder#*$folder_str/}
        file_name=${folder_name%/*}

        # generate parts
        { ffmpeg -stream_loop 0 -start_number ${start} -r 60 -i ${folder}"ALL/%d.png" -vcodec libx264 -crf 0 ${outdir}${file_name}.mp4; } &
        pid=$!
        wait $pid

        # delete images
        rm -r ${folder}

    done

    # generate list of parts to merge (in order)
    txt_file="${outdir}files.txt"
    rm -f $txt_file
    dir=$(ls ${outdir}*.mp4 |sort -V)
    for f in ${dir}; do echo "file '$f'" >> $txt_file; done

    # merge video parts based on list
    file="${outdir}daily_${folder_str}.mp4"
    { ffmpeg -f concat -safe 0 -i ${txt_file} -c copy ${file}; }

    # upload video
    contact="bjorn.linder@misu.su.se / olemartin.christensen@misu.su.se"
    description="Autogenerated timelapse of MATS L${level} data (v${version}) from ${start_date} to ${end_date}. Contact: ${contact}"
    { python ${MATS_dir}MATS-utility-functions/src/mats_utils/daily_preview/upload.py --file="${file}" --title="MATS L${level}: ${start_date} to ${end_date}" --description="${description}" --keywords="MATS" --category="22" --privacyStatus="unlisted"; }

done

echo 'End of program .....'