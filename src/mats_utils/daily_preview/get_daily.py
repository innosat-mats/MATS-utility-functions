from mats_utils.rawdata.read_data import read_MATS_data
import datetime as DT
import argparse
from datetime import date, timedelta
from mats_utils.plotting.plotCCD import all_channels_plot
import numpy as np

def generate_day_interval(snippet=False):

    today = date.today()
    yesterday = today - timedelta(days=1)
    daybefore = today - timedelta(days=3)

    start_time = DT.datetime(daybefore.year,
                             daybefore.month,
                             daybefore.day,
                             0, 0, 0)

    if snippet:
        stop_time = start_time + timedelta(minutes=2)
    else:
        stop_time = DT.datetime(yesterday.year,
                                yesterday.month,
                                yesterday.day,
                                0, 0, 0)
    return start_time, stop_time


parser = argparse.ArgumentParser(description='Plots measurements from previous'
                                 ' day specify data level and version of data')
parser.add_argument('--outdir', type=str,
                    help='output directory')
parser.add_argument('--level', type=str, default='1a',
                    help='choose between 1a or 1b')
parser.add_argument('--version', type=str, default='0.4',
                    help='specifies version of data')
parser.add_argument('--snippet', action="store_true", default=False,
                    help='If supplied; short interval for debugging')

args = parser.parse_args()

level = args.level
version = args.version
outdir = args.outdir
snippet = args.snippet

start_time, stop_time = generate_day_interval(snippet=snippet)

# get measurements
CCDitems = read_MATS_data(start_time, stop_time, level=level, version=version)

# generate figures
# note: issue when plotting several thousands of figures,
# plotting slows down. for now: split up by calling different output folders:


sets = int(np.floor(len(CCDitems)/500))

if int(len(CCDitems)) > 500:
    for part in range(0, sets):
        
        if part == 0:
            start_point = 0
        else:
            start_point = part*500-1
        try:
            if (part+1)*500 < int(len(CCDitems)):
                all_channels_plot(CCDitems[start_point:(part+1)*500-1], outdir=outdir+'part'+str(part)+'/', optimal_range=True)
            else:
                all_channels_plot(CCDitems[start_point:int(len(CCDitems))-1], outdir=outdir+'part'+str(part)+'/', optimal_range=True)
        except:
            print(f'Error plotting data (part: {part})')
else:
    try:
        all_channels_plot(CCDitems, outdir=outdir+'part0/', optimal_range=True)
    except:
        print('Error plotting images..')


