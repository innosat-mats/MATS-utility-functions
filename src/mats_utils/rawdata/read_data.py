from pyarrow import fs
import boto3
from datetime import timezone
from mats_l1_processing.read_parquet_functions import read_ccd_data_in_interval,add_ccd_item_attributes,remove_empty_images
#%matplotlib widget

def read_MATS_data(start_date,end_date):
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
    
    ccd_data = read_ccd_data_in_interval(start_date,end_date,"ops-payload-level1a-v0.2",s3)
    add_ccd_item_attributes(ccd_data)
    remove_empty_images(ccd_data)
    return (ccd_data)