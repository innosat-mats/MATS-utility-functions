import pandas as pd
from mats_utils.rawdata.read_data import read_MATS_data
import datetime as DT
from mats_utils.rawdata.cropping import make_crop_filter
import numpy as np

def random_datetimes(starttime, endtime, number_of_images, seed):
    """ 
    Generate random datetimes within the given time range

    Args:
        starttime (datetime): The start time of the time range.
        endtime (datetime): The end time of the time range.
        number_of_images (int): The number of random images to select.
        seed (int): The seed for random number generation.

    Returns:
        random_datetimes (list): A list of random datetimes.

    """
    import random
    random.seed(seed)  # Set the seed for reproducibility
    random_numbers = [random.random() for _ in range(number_of_images)]

    random_datetimes = []
    for random_number in random_numbers:
        random_datetime = starttime + (endtime - starttime) * random_number
        random_datetimes.append(random_datetime)
    return random_datetimes

def select_random_images(starttime, endtime, filter, number_of_images, seed=42, idifference=0,level='1b'):
    """
    Selects random images within a given time range and returns a dataframe.

    Args:
        starttime (datetime): The start time of the time range.
        endtime (datetime): The end time of the time range.
        filter (str): The filter to apply for data download.
        number_of_images (int): The number of random images to select.
        seed (int): The seed for random number generation.
        idifference (int):  Default is 0, and no difference is calculated, i.e. this part of the code is ignored.
        The integer specifies how many images to skip between the images. For example, if idifference=1,
        the function will download the difference between one and the next image. 

    Returns:
        df (DataFrame): A dataframe containing the selected random images.
    """


    df = pd.DataFrame()  # Create an empty dataframe
    for i, intervalstarttime in enumerate(random_datetimes(starttime, endtime, number_of_images, seed)): #Use the randomly generated datetimes as starttime for the data download
        
        if (i + 1) % 50 == 0:
            print(f"Step number: {i + 1}")
        # Add 1 second
        endtime = intervalstarttime + DT.timedelta(seconds=7)

        
        try:
            if idifference==0: #If difference is 0, download the data as is, not a difference between two images
                df_shortinterval = read_MATS_data(intervalstarttime, endtime, filter, level=level)
                df = pd.concat([df, df_shortinterval[0:1]], ignore_index=True)  # Add the first element of df_shortinterval to df

                #print('length of df_shortinterval:', len(df_shortinterval))
            elif idifference>0: #If difference is positive, download the difference between two images
                endtime = intervalstarttime + DT.timedelta(seconds=7+7*idifference) #longer interval since more images are needed
                df_shortinterval = read_MATS_data(intervalstarttime, endtime, filter, level=level)
                df_shortinterval ['ImageCalibratedDiff'+str(idifference)] = df_shortinterval['ImageCalibrated'].diff(periods=idifference)
                df = pd.concat([df, df_shortinterval[idifference:idifference+1]], ignore_index=True)  # Add the idifference'th element of df_shortinterval, ie the element that holds the difference  
        except:
            print('found no data for intervalstarttime', intervalstarttime)
    
    return df


def select_random_images_all_channels(starttime, endtime, number_of_images, crop=None, seed=42, idifference=0, level='1b'): 
    """
    Selects random images within a given time range and returns 7 dataframes, one for each channel.

    Args:
        starttime (datetime): The start time of the time range.
        endtime (datetime): The end time of the time range.
        filter (str): The filter to apply for data download. 
        number_of_images (int): The number of random images to select.
        seed (int): The seed for random number generation.
        crop (str): The crop version to use for the data download.
        idifference (int):  Default is 0, and no difference is calculated, i.e. this part of the code is ignored.
        The integer specifies how many images to skip between the images. For example, if idifference=1,
        the function will download the difference between one and the next image. 

    Returns:
        dfchannelsdict (dict): A dictionary with dataframes containing the selected random images for each channel.
    """

   
    #df = pd.DataFrame()  # Create an empty dataframe
    channels=['IR1', 'IR2', 'IR3', 'IR4', 'UV1', 'UV2', 'NADIR']
    dfchannelsdict = {}
    for channel in channels:
          dfchannelsdict[channel] = pd.DataFrame()  # Create an empty dataframe for each channel


    for i, intervalstarttime in enumerate(random_datetimes(starttime, endtime, number_of_images, seed)): #Use the randomly generated datetimes as starttime for the data download
        
        if (i + 1) % 50 == 0:
            print(f"Step number: {i + 1}")
            for channel in channels:
                print('length of dfchannelsdict[channel]:', len(dfchannelsdict[channel]))


       
        endtime = intervalstarttime + DT.timedelta(seconds=7+7*idifference)

        df_shortinterval = pd.DataFrame()  # Create an empty dataframe
        founddata=False
        try:
            df_shortinterval= read_MATS_data(intervalstarttime, endtime, level=level, pritfilesys=False)
        except:
            print('found no data for intervalstarttime', intervalstarttime)
        else: #if the dataframe is not empty, ie MATS data was found
            for channel in channels:
                if crop:
                    filter_channelcrop=make_crop_filter(channel, crop)
                    df_channel_short = df_shortinterval[
                        (df_shortinterval['channel'] == filter_channelcrop['channel']) &
                        (df_shortinterval['NRSKIP'].isin(filter_channelcrop['NRSKIP'])) &
                        (df_shortinterval['NRBIN'].isin(filter_channelcrop['NRBIN'])) &
                        (df_shortinterval['NROW'].isin(filter_channelcrop['NROW'])) &
                        (df_shortinterval['NCSKIP'].isin(filter_channelcrop['NCSKIP'])) &
                        (df_shortinterval['NCBINCCDColumns'].isin(filter_channelcrop['NCBINCCDColumns'])) &
                        (df_shortinterval['NCOL'].isin(filter_channelcrop['NCOL'])) &
                        (df_shortinterval['NCBINFPGAColumns'].isin(filter_channelcrop['NCBINFPGAColumns']))
                    ].copy()  # Make a copy of the filtered dataframe
                else:
                    df_channel_short = df_shortinterval[df_shortinterval['channel'] == channel].copy()

                if len(df_channel_short)>=idifference+1: #if the dataframe is long enough to calculate the difference
                    if idifference > 0:  # If difference is positive, download the difference between two images as an additional column
                        df_channel_short.loc[:, 'ImageCalibratedDiff'+str(idifference)] = df_channel_short['ImageCalibrated'].diff(periods=idifference)
                    dfchannelsdict[channel] = pd.concat([dfchannelsdict[channel], df_channel_short[idifference:idifference+1]], ignore_index=True)  # Add the item, the first element of df_channel_short if idifference=0, the idifference'th element of df_channel_short if idifference>0
                else:
                    #print('found no data for '+channel+ ' in intervalstarttime', intervalstarttime, 'adding empty row of dataframe')
                    # Create empty DataFrame with empty rows
                    empty_data = {col: [np.nan for _ in range(1)] for col in df_shortinterval.columns} 
                    empty_df = pd.DataFrame(empty_data)
                    # Concatenate the original DataFrame with the empty DataFrame
                    dfchannelsdict[channel] = pd.concat([dfchannelsdict[channel], empty_df], ignore_index=True) # add first element of empty_df 
                        
            # Check so that all of the data frames are the same length
            if not all(len(dfchannelsdict['IR1']) == len(dfchannelsdict[channel]) for channel in channels):
                print('The dataframes are not of the same length', len(dfchannelsdict['IR1']), len(dfchannelsdict['IR2']), len(dfchannelsdict['IR3']), len(dfchannelsdict['IR4']), len(dfchannelsdict['UV1']), len(dfchannelsdict['UV2']), len(dfchannelsdict['NADIR']))
                print('Step number: ', i)
                break

    return dfchannelsdict