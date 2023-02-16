from mats_utils.rawdata.read_data import read_MATS_data
import datetime as DT
import pandas as pd
import argparse
from datetime import date,timedelta
from mats_utils.plotting.plotCCD import all_channels_plot

parser = argparse.ArgumentParser(description='Plots measurements from previous day')
parser.add_argument('--outdir', type=str,
                    help='output directory')
parser.add_argument('--level', type=str,
                    help='choose between 1a or 1b')
parser.add_argument('--version', type=str,
                    help='specifies version of data')

args = parser.parse_args()

level = args.level
version = args.version
outdir = args.outdir

def generate_day_interval():

    today = date.today()
    yesterday = today - timedelta(days=4)

    start_time = DT.datetime(yesterday.year,
                             yesterday.month,
                             yesterday.day,
                             0, 0, 0)

    stop_time = DT.datetime(yesterday.year,
                            yesterday.month,
                            yesterday.day,
                            0, 2, 0)
    
    return start_time, stop_time


start_time, stop_time = generate_day_interval()

CCDitems = read_MATS_data(start_time, stop_time, level=level, version=version)

all_channels_plot(CCDitems, outdir=outdir, optimal_range=True)