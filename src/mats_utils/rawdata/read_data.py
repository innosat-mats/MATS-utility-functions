from pyarrow import fs
import pyarrow.dataset as ds
import boto3
from datetime import timezone, timedelta
from mats_l1_processing.read_parquet_functions import read_ccd_data_in_interval,add_ccd_item_attributes,remove_faulty_rows
import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore
#%matplotlib widget


def read_MATS_data(start_date,end_date,filter=None,version='0.4',level='1a'):
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

    if level == '1b' and version == "0.4":
        filesystem = f"ops-payload-level{level}-v{version}" + "/ops-payload-level1a-v0.5"
    elif level == '1b' and version == "0.3":
        filesystem = f"ops-payload-level{level}-v{version}" + "/ops-payload-level1a-v0.4"
    elif level == '0':
        filesystem = f"ops-payload-level{level}-v{version}" + "/CCD"
    else:
        filesystem = f"ops-payload-level{level}-v{version}"
    
    ccd_data = read_ccd_data_in_interval(start_date, end_date, filesystem, s3,filter=filter)

    if level == '1a':
        add_ccd_item_attributes(ccd_data)
        remove_faulty_rows(ccd_data)

    if len(ccd_data) == 0:
        raise Warning('Dataset is empty check version or time interval')

    if level == '1b':
        ccd_data["ImageCalibrated"] = ccd_data.apply(list_to_ndarray, axis=1)

    return (ccd_data)

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

    df = table.to_pandas().reset_index().set_index('TMHeaderTime')

    if len(df) == 0:
        raise Warning('Dataset is empty check version or time interval')

    return (df)

def list_to_ndarray(l1b_data_row):
    '''
        Converts a list of 1d arrays into a 2d numpy array
    '''
    return np.stack(l1b_data_row.ImageCalibrated) 


def read_MATS_payload_data(start_date,end_date,data_type='HTR',filter=None,version='0.3',drop_col=[]):
    """Reads the payload data between the specified times. 

    Args:
        start (datetime):           Read payload data from this time (inclusive).
        stop (datetime):            Read payload data up to this time (inclusive).
        data_type (str):            key describing the different types of data :
                                    CCD, CPRU, HTR, PWR, STAT, TCV, PM
        filter (Optional[dict]):    Extra filters of the form:
                                    `{fieldname1: [min, max], ...}`
                                    (Default: None)

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
    

    start_with_margin =  start_date - timedelta(hours=1)
    stop_with_margin = end_date + timedelta(hours=1)

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

    table = dataset.to_table(filter=partition_filter & filterlist)
    dataframe = table.to_pandas()
    dataframe.reset_index(inplace=True)
    dataframe.set_index('TMHeaderTime',inplace=True)
    dataframe.sort_index(inplace=True)
    dataframe.reset_index(inplace=True)

    if len(dataframe) == 0:
        raise Warning('Dataset is empty check version or time interval')

    return dataframe
