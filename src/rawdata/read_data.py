from pyarrow import fs, schema, string
from pyarrow.dataset import FilenamePartitioning
import pyarrow.dataset as ds
import boto3
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pylab as plt
from scipy.spatial.transform import Rotation as R
import io
from mats_l1_processing.read_parquet_functions import read_ccd_data_in_interval,read_ccd_items_in_interval,dataframe_to_ccd_items
#%matplotlib widget

def read_MATS_data(start_date,end_date):
    session = boto3.session.Session(profile_name="mats")
    credentials = session.get_credentials()

    s3 = fs.S3FileSystem(
        secret_key=credentials.secret_key,
        access_key=credentials.access_key,
        region=session.region_name,
        session_token=credentials.token)

    dataset = ds.dataset(
        "ops-payload-level1a-v0.2",
        filesystem=s3   
    )

    if start_date.tzinfo == None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo == None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    
    df = read_ccd_data_in_interval(start_date,end_date,"ops-payload-level1a-v0.2",s3)
   
    return (df)

# df = df[df.CCDSEL==1] #only IR1
# print('Number of images = ',df.shape[0])
# clim=999
# plt.close('all')
# ccdnames=('IR1','IR4','IR3','IR2','UV1','UV2')
# for i in range(20):
#     fig,axis=plt.subplots(1,1,figsize=[8,2])
#     image=plt.imread(io.BytesIO(df.iloc[i].ImageData))
#     #image_lat,image_lon=Geoidlib.xyxECEF2geodetic( df.iloc[i] )
#     #sp.set_data(image)
#     q=df['afsAttitudeState'][i]
#     quat=R.from_quat(np.roll(q,-1))
#     FOVq=quat.apply(np.array([0, 0, -1]))
#     ra,dec=xyz2radec(np.expand_dims(FOVq,axis=1),deg=True)
#     print(ra,dec)
#     sp=plt.imshow(image, cmap="magma", origin="lower", interpolation="none")
#     axis.axis("auto")
#     if clim == 999:
#         [col, row]=image.shape
#         #Take the mean and std of the middle of the image, not boarders
#         mean = image[int(col/2-col*4/10):int(col/2+col*4/10), int(row/2-row*4/10):int(row/2+row*4/10)].mean()
#         std = image[int(col/2-col*4/10):int(col/2+col*4/10), int(row/2-row*4/10):int(row/2+row*4/10)].std()
#         sp.set_clim([mean - 2 * std, mean + 2 * std])
#     else:
#         sp.set_clim(clim)
#     plt.title("{:4s} RA = {:8.3f} Dec = {:8.3f} {:s}".format(ccdnames[df.iloc[i].CCDSEL - 1],ra[0],  dec[0],df.index[i].isoformat()))
#     plt.colorbar()
#     plt.show()