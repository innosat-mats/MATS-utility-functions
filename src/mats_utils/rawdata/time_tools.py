import datetime as DT
import numpy as np
import pandas as pd

def utc_to_onboardTime(utc_date):
    """Function which converts a date in utc into onboard time (GPS) in seconds and rounds to nearest 10th of a second.

    Arguments:
        utc_date (:obj: datetime): the date as datetime object

    Returns:
        (float): Onboard GPS time in seconds.

    """

    dt_object = utc_date
    dateEpochGPS = DT.datetime(1980,1,6,0,0,0,0,tzinfo=DT.timezone.utc) + DT.timedelta(seconds=-18)
    onboardGPSTime = (dt_object-dateEpochGPS).total_seconds()

    return onboardGPSTime

def onboardTime_to_utc(onboardGPSTime):
    """Function which converts a date in utc into onboard time (GPS) in seconds and rounds to nearest 10th of a second.

    Arguments:
        (float): Onboard GPS time in seconds.

    Returns:
        utc_date (:obj:`datetime`): return utc time as datetimeobject.

    """

    dateEpochGPS = DT.datetime(1980,1,6,0,0,0,0,tzinfo=DT.timezone.utc) + DT.timedelta(seconds=-18)
    utc_date = dateEpochGPS+DT.timedelta(seconds = onboardGPSTime)
    
    if utc_date.tzinfo == None:
        utc_date = utc_date.replace(tzinfo=DT.timezone.utc)

    return utc_date



def add_datetime(df):
    """Converts all times to datetime objects for a MATS dataframe.

    Arguments:
        df (:obj:`datetime`): dataframe.

    Returns:
        df (:obj:`datetime`): dataframe with datetimeobjects.

    """

    df['EXP Date'] = pd.to_datetime(df['EXP Date'])
    df['TMHeaderTime'] = pd.to_datetime(df['TMHeaderTime'])
    df['RamsesTime'] = pd.to_datetime(df['RamsesTime'])

    return df
