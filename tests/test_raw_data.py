import rawdata.time_tools as time_tools
import rawdata.read_data as read_data
import datetime as DT
import plotting.plotCCD as pltCCD
import pandas as pd
from mats_l1_processing.read_parquet_functions import read_ccd_data_in_interval,read_ccd_items_in_interval,dataframe_to_ccd_items

def test_onboard_time():
    utctime = DT.datetime(2022,11,4,16,23,45,120000,tzinfo=DT.timezone.utc)
    onboard_time = time_tools.utc_to_onboardTime(utctime)
    
    assert onboard_time == 1351614243.12
    assert time_tools.onboardTime_to_utc(onboard_time) == utctime

def test_read_MATS_data():
    start_time = DT.datetime(2022,12,4,12,0,0,tzinfo=DT.timezone.utc)
    stop_time = DT.datetime(2022,12,4,12,10,0,tzinfo=DT.timezone.utc)
    df = read_data.read_MATS_data(start_time,stop_time)
    pltCCD.orbit_plot(df.iloc[0],'./') #test plotting single image
    pltCCD.orbit_plot(df,'./')#test plotting entire dataframe

if __name__ == "__main__":
    test_read_MATS_data()