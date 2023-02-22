from pyarrow import fs
import boto3
from datetime import timezone
from mats_l1_processing.read_parquet_functions import read_ccd_data_in_interval,add_ccd_item_attributes,remove_faulty_rows
#%matplotlib widget

def read_MATS_data(start_date,end_date,filter=None,version='0.4',level='1a'):
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

    ccd_data = read_ccd_data_in_interval(start_date, end_date, f"ops-payload-level{level}-v{version}", s3,filter=filter)

    if level is '1a':
        add_ccd_item_attributes(ccd_data)
        remove_faulty_rows(ccd_data)

    if len(ccd_data) == 0:
        raise Warning('Dataset is empty check version or time interval')

    return (ccd_data)
