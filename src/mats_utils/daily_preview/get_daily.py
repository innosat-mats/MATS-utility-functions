from mats_utils.rawdata.read_data import read_MATS_data
import datetime as DT
import argparse
from datetime import date, timedelta
from mats_utils.plotting.plotCCD import all_channels_plot
import cProfile
import time
import io
import pstats

def generate_day_interval(snippet=False):

    today = date.today()
    yesterday = today - timedelta(days=1)
    daybefore = today - timedelta(days=2)

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
all_channels_plot(CCDitems, outdir=outdir, optimal_range=True)
