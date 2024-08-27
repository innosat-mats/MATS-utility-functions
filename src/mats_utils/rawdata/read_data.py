from pyarrow import fs
import pyarrow.dataset as ds
import boto3
from datetime import timezone, timedelta
from mats_l1_processing.read_parquet_functions import read_ccd_data_in_interval,add_ccd_item_attributes,remove_faulty_rows,convert_image_data,read_instrument_data_in_interval
import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore
import datetime as DT
import logging
import pytz
#%matplotlib widget


def read_MATS_data(start_date,end_date,filter=None,version='0.4',level='1a',dev=False):
    session = boto3.session.Session(profile_name="mats")
    credentials = session.get_credentials()

    s3 = fs.S3FileSystem(
        secret_key=credentials.secret_key,
        access_key=credentials.access_key,
        region=session.region_name,
        connect_timeout=10,
        session_token=credentials.token)

    if start_date.tzinfo == None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo == None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    level_list = level.split("/",1)
    main_level = level_list[0]
    subdir = None
    if len(level_list)>1:
        subdir = level_list[1]
    if main_level == '1b' and version == "0.4":
        filesystem = f"ops-payload-level{main_level}-v{version}" + "/ops-payload-level1a-v0.5"
    elif main_level == '1b' and version == "0.3":
        filesystem = f"ops-payload-level{main_level}-v{version}" + "/ops-payload-level1a-v0.4"
    elif main_level == '0':
        if len(level_list) == 1:
            raise ValueError("For level 0 subdir must be given")
        
        filesystem = f"ops-payload-level{main_level}-v{version}" + "/" + subdir
    else:
        filesystem = f"ops-payload-level{main_level}-v{version}"
    if dev:
        filesystem = f"dev-payload-level{main_level}"
    
    print(filesystem)
    if (main_level == '1b') or (main_level == '1a') or (main_level == '0' and subdir == 'CCD'): 
        try:
            data = read_ccd_data_in_interval(start_date, end_date, filesystem, s3,filter=filter)
        except:
            raise ValueError("something wrong with dataset, probably it does not exists")
    else:
        try:
            data = read_instrument_data_in_interval(start_date, end_date, filesystem, s3,filter=filter)
        except:
            raise ValueError("something wrong with dataset, probably it does not exists")

    if (main_level == '1a') and (float(version) <= 0.5):
        add_ccd_item_attributes(data)
    
    if (main_level == '1a') or (main_level == '0' and subdir == 'CCD'):
        convert_image_data(data)
        remove_faulty_rows(data)    

    if len(data) == 0:
        raise Warning('Dataset is empty check version or time interval')

    if main_level == '1b':
        data["ImageCalibrated"] = data.apply(list_to_ndarray, axis=1)

    return (data)

def read_MATS_PM_data(start_date,end_date,filter=None,version='0.2',level='1a'):
    session = boto3.session.Session(profile_name="mats")
    credentials = session.get_credentials()

    s3 = fs.S3FileSystem(
        secret_key=credentials.secret_key,
        access_key=credentials.access_key,
        region=session.region_name,
        session_token=credentials.token)

    if start_date.tzinfo == None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo == None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    dataset = ds.dataset(f"ops-payload-level{level}-pm-v{version}", filesystem=s3)
    
    table = dataset.to_table(filter=(ds.field('PMTime') > start_date) 
                           & (ds.field('PMTime') < end_date) )

    df = table.to_pandas()
    if df.index.name == 'PMTime':
        df.reset_index(inplace=True)
        df.set_index('TMHeaderTime',inplace=True)
        df.sort_index(inplace=True)
        df.reset_index(inplace=True)
    else:
        df.reset_index(drop=True,inplace=True)
        df.set_index('TMHeaderTime',inplace=True)
        df.sort_index(inplace=True)
        df.reset_index(inplace=True)

    if len(df) == 0:
        raise Warning('Dataset is empty check version or time interval')

    return (df)

def list_to_ndarray(l1b_data_row):
    '''
        Converts a list of 1d arrays into a 2d numpy array
    '''
    return np.stack(l1b_data_row.ImageCalibrated) 


def read_MATS_payload_data(start_date,end_date,data_type='CCD',filter=None,version='0.3',columns=None):
    """Reads the payload data between the specified times. 

    Args:
        start (datetime):           Read payload data from this time (inclusive).
        stop (datetime):            Read payload data up to this time (inclusive).
        data_type (Optional str):            key describing the different types of data :
                                    CCD, CPRU, HTR, PWR, STAT, TCV, PM
                                    (Defaults: 'CCD')
        filter (Optional[dict]):    Extra filters of the form:
                                    `{fieldname1: [min, max], ...}`
                                    (Default: None)
        columns (Optional[str]):    List of columns to be imported in the dataframe
                                    (Default: None ie all the columns are imported)

    Returns:
        DataFrame:      The payload data.
    """

    session = boto3.session.Session(profile_name="mats")
    credentials = session.get_credentials()
    filesystem = f'ops-payload-level0-v{version}'
    file = f"{filesystem}/{data_type}"

    s3 = fs.S3FileSystem(
        secret_key=credentials.secret_key,
        access_key=credentials.access_key,
        region=session.region_name,
        session_token=credentials.token)
    
    if start_date.tzinfo == None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo == None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    partitioning = ds.partitioning(
        schema=pa.schema(
            [
                ("year", pa.int16()),
                ("month", pa.int8()),
                ("day", pa.int8()),
            ]
        ),
    )

    dataset = ds.dataset(
        file,
        filesystem=s3,
        partitioning=partitioning,
        )
    

    start_with_margin =  start_date - timedelta(days=1)
    stop_with_margin = end_date + timedelta(days=1)

    partition_filter = (
        ds.field("year") * 1000000
        + ds.field("month") * 10000
        + ds.field("day") * 100
        >= start_with_margin.year * 1000000
        + start_with_margin.month * 10000
        + start_with_margin.day * 100
    ) & (
        ds.field("year") * 1000000
        + ds.field("month") * 10000
        + ds.field("day") * 100
        <= stop_with_margin.year * 1000000
        + stop_with_margin.month * 10000
        + stop_with_margin.day * 100
    )
    
    filterlist = (
        (ds.field("TMHeaderTime") >= pd.Timestamp(start_date))
        & (ds.field("TMHeaderTime") <= pd.Timestamp(end_date))
    )
    if filter != None:
        for variable in filter.keys():
            filterlist &= (
                (ds.field(variable) >= filter[variable][0])
                & (ds.field(variable) <= filter[variable][1])
            )

    if columns != None and 'TMHeaderTime' not in columns: # the column TMHeaderTime has to be always selected as it is set as index
        columns.append('TMHeaderTime')
    table = dataset.to_table(filter=partition_filter & filterlist,columns=columns)
    dataframe = table.to_pandas()
    dataframe.reset_index(inplace=True)
    dataframe.set_index('TMHeaderTime',inplace=True)
    dataframe.sort_index(inplace=True)
    dataframe.reset_index(inplace=True)

    if len(dataframe) == 0:
        raise Warning('Dataset is empty check version or time interval')

    return dataframe


def store_as_parquet(df, fname):
    """ Stores pandas dataframe to parquet file.
        ImageCalibrated is flattened (do be restored after reading), as parquet can only handle 1-D ndarrays.

        Args:
           df - pandas dataframe to store;
           fname - filename for the parquet file.

        Returns:
           Nothing.

    """

    df["ImageCalibrated"] = [arr.flatten() for arr in df["ImageCalibrated"]]
    df.to_parquet(fname, compression='gzip', engine='pyarrow', allow_truncated_timestamps=True, coerce_timestamps='ms')


def load_parquet(fname, start=None, stop=None, filt={}, fail_if_empty=True, fail_if_no_file=True):
    """ Loads pandas dataframe from parquet file. Has similar args as read_MATS_data.

        Args:
            start - datetime, start of the time interval to retrieve;
            stop - datetime, end of the time interval to retrieve;
            filt - dict with entries of the form <pandas column name>: [<lower boundary>, <upper boundary>]
                   to filter the data. Design to work similarly to read_MATS_data filters.
            fail_if_empty - bool, if True, throws an error if filtered dataset ends up empty;
            fail_if_no_file - bool, if True, throws and error if file for the appropriate time not found in
                              the specified directory. False is useful for long data interfal retrieval in case
                              of data gaps;

        Returns:
            res - pandas data frame with MATS data.

    """

    try:
        df = pd.read_parquet(fname)
    except Exception:
        if fail_if_no_file:
            raise FileNotFoundError(f"File {fname} not found or unreadable!")
        else:
            return None

    valid = np.ones(len(df))
    for col, lim in filt.items():
        valid = np.logical_and(valid, df[col] >= lim[0])
        valid = np.logical_and(valid, df[col] <= lim[1])

    if start is not None:
        valid = np.logical_and(valid, df["EXPDate"] >= start.replace(tzinfo=timezone.utc))
    if stop is not None:
        valid = np.logical_and(valid, df["EXPDate"] <= stop.replace(tzinfo=timezone.utc))

    if not np.all(valid):
        df = df[valid]
    if len(df) == 0:
        if fail_if_empty:
            raise RuntimeError("There are no images in this data set! Adjust start and stop times or filters!")
        else:
            return df

    df["ImageCalibrated"] = [arr.reshape((df["NROW"].iloc[i], df["NCOL"].iloc[i] + 1))
                             for i, arr in enumerate(df["ImageCalibrated"])]
    df.reset_index(drop=True, inplace=True)
    return df


def load_multi_parquet(dirname, start, stop, filt={}, fail_if_empty=True):
    """ Loads pandas dataframe from parquet files from specified directory. If the specified time interval spans
        multiple files, the loaded data sets are merged. Has similar args as read_MATS_data.

        Args:
            start - datetime, start of the time interval to retrieve;
            stop - datetime, end of the time interval to retrieve;
            filt - dict with entries of the form <pandas column name>: [<lower boundary>, <upper boundary>]
                   to filter the data. Design to work similarly to read_MATS_data filters;
            fail_if_empty - bool, if True, throws an error if filtered dataset ends up empty.

        Returns:
            res - pandas data frame with MATS data.

    """

    hours = all_hours(start, stop)
    fnames = [f"{dirname}/{hours_filename(h)}" for h in hours]

    if len(hours) == 1:
        return load_parquet(fnames[0], start=start, stop=stop, filt=filt,
                            fail_if_empty=fail_if_empty)
    else:
        dfs = [load_parquet(fnames[0], start=start, stop=hours[1], filt=filt,
                            fail_if_empty=False, fail_if_no_file=False)]
        for i in range(1, len(hours) - 1):
            dfs.append(load_parquet(fnames[i], start=hours[1], stop=hours[i + 1], filt=filt,
                                    fail_if_empty=False, fail_if_no_file=False))
        dfs.append(load_parquet(fnames[-1], start=hours[-1], stop=stop, filt=filt,
                                fail_if_empty=False, fail_if_no_file=False))

    dfs = [x for x in dfs if x is not None]
    if len(dfs) != len(hours):
        if len(dfs) == 0:
            raise FileNotFoundError("None of the files for the specified time interval found! " +
                                    "Check time interval and directory!")
        else:
            logging.warn("Some files in the time interval were missing, likely because of missing data.")

    res = pd.concat(dfs, axis=0)
    res.reset_index(drop=True, inplace=True)
    if fail_if_empty and len(res) == 0:
        raise ValueError("No data matching the criteria found in the time interval!")
    else:
        return res


def all_hours(start, stop):
    """ Helper function that returns a list of datetime objects for every exact hour starting with the hour
        before start and ending with the hour before stop.

        Args:
            start - datetime, the start of time interval;
            stop - datetime, the end of time interval;

        Returns:
            list of datetime.
    """
    if start > stop:
        raise ValueError("Start of the given time interval is later than the end!")
    first = start.replace(second=0, microsecond=0, minute=0)
    numhours = int(np.ceil((stop - first) / DT.timedelta(hours=1)))
    return [first + i * DT.timedelta(hours=1) for i in range(numhours)]


def hours_filename(time):
    """ Returns the standard file name for parquet file starting at specified time.

        Args:
            time - datetime;

        Returns:
            str with the standard file name.
    """
    return f"{time.year}-{time.month:02d}-{time.day:02d}_{time.hour:02d}.parquet.gzip"
