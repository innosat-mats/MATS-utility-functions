from pyarrow import fs
import pyarrow.dataset as ds
import boto3
from datetime import timezone
from mats_l1_processing.read_parquet_functions import read_ccd_data_in_interval,add_ccd_item_attributes,remove_faulty_rows
import numpy as np
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
    else:
        filesystem = "ops-payload-level{level}-v{version}"
    
    ccd_data = read_ccd_data_in_interval(start_date, end_date, filesystem, s3,filter=filter)

    if level == '1a':
        add_ccd_item_attributes(ccd_data)
        remove_faulty_rows(ccd_data)

    if level == '1b':
        ccd_data["ImageCalibrated"] = ccd_data.apply(list_to_ndarray, axis=1)

    if len(ccd_data) == 0:
        raise Warning('Dataset is empty check version or time interval')

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
