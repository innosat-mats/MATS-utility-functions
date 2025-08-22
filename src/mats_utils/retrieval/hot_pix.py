"""
These functions are not part of the main processing pipeline and are intended for 
the case when MATS pointing is off.
"""


from mats_utils.geolocation.coordinates import col_heights
import numpy as np



def compute_threshold_pixels(CCDitem, thresholdalt):
    """
    Calculates what pixel (ie what row)that is at the threshold altitude in each column

    Input:
    CCDitem: row of a dataframe with CCDitems
    thresholdalt: float with the altitude threshold

    Returns:
    pixel_indices: list of pixel indices (rows) at the threshold altitude in each column
    """
    thl = col_heights(CCDitem, 0, 2) # (left) the 2 makes the function return the lowest and highest pixel
    thr = col_heights(CCDitem, CCDitem.NCOL, 2) # (right) the 2 makes the function return the lowest and highest pixel
    altrowspan = thl[1] - thl[0]  # Altitude span of the image, could equally well have used thr
    altcoldiff = thl[0] - thr[0]  # Altitude difference between the sides of the image
    pixel_indices = []
    if thl[1] < thresholdalt and thr[1] < thresholdalt: #satellite pointing is too low for all pixels
        # no pixels should be selected
        pixel_indices = np.full(CCDitem.NCOL+1, CCDitem.NROW)
    elif thl[1] < thresholdalt or thr[1] < thresholdalt: #satellite pointing may be ok for some images but we are conservative
        # no pixels should be selecte
        pixel_indices = np.full(CCDitem.NCOL+1, CCDitem.NROW)
    elif thl[0] > thresholdalt and thr[0] > thresholdalt:# satellite pointing is high enough for all pixels to be dark, ie selected
        #set all pixels to 0
        pixel_indices = np.full(CCDitem.NCOL+1, 0)
    else:  # satellite pointing is ok for some pixels in each column
        rows=CCDitem.NROW
        dalt_dcol=altcoldiff / (CCDitem.NCOL - 1)  # Altitude difference per column
        dalt_drow=altrowspan / (rows - 1)  # Altitude difference per row
        drow_dcol=dalt_dcol / dalt_drow  # Number of rows per column
        # Calculate the pixel (ie row ) that corresponds to the threshold altitude in the first column
        firstpixel_index = int((thresholdalt - thl[0]) / altrowspan* (rows - 1))  # Pixel row index in the first column

        for col in range(CCDitem.NCOL+1):        
            pixel_index = int(firstpixel_index + col * drow_dcol)  # Calculate the pixel index for each column
            if pixel_index < 0:
                pixel_index = 0
            elif pixel_index >= rows:
                pixel_index = rows - 1
            pixel_indices.append(pixel_index)
    return pixel_indices



def create_hot_pix_map_one_channel(df, thresholdalt=120000, remove_background=True):
    """ This hot pixel compensation utelises the off pointing of MATS and uses 
    the portion of the image that is above the air glow layer to determine hot pixels. 
    Any background light is subtracted, so that the average of a row is set to the 
    value of the top of the image, if remove_background is set to true 

    Input:
    df: DataFrame containing the image data
    thresholdalt: Optional. float with the altitude threshold above which the image should be dark
    remove_background: Optional. boolean flag to remove background light

    Output:
    hot_pixel_map: 2D numpy array representing the hot pixel map for the channel
    """

    # Check that all rows have the same channel
    unique_channels = df['channel'].unique()
    if len(unique_channels) != 1:
        raise ValueError("DataFrame contains multiple channels. Please provide data from a single channel.")

    # List to store processed images
    processed_images = []

    
    for _, row in df.iterrows():
        image = np.array(row.ImageCalibrated, dtype=float)  # Ensure float for NaN support
        pixel_indices = compute_threshold_pixels(row, thresholdalt)

        # Set pixels above threshold to NaN
        for col, threshold_row in enumerate(pixel_indices):
            image[:threshold_row, col] = np.nan

        processed_images.append(image)

    # Stack images and compute nanmean
    stacked_images = np.stack(processed_images)
    hot_pixel_map = np.nanmean(stacked_images, axis=0)

    print("Hot pixel map created for channel", df['channel'].iloc[0], "with shape:", hot_pixel_map.shape)

    if remove_background: #Removes the extra light that is generally at the bottom of the image
        av_dark_signal= np.nanmean(hot_pixel_map[int(df.iloc[0].NROW*0.7):int(df.iloc[0].NROW*0.9):])
        hot_pixel_map = hot_pixel_map - np.nanmean(hot_pixel_map, axis=1, keepdims=True)+ av_dark_signal

    return hot_pixel_map

def create_all_hot_pix_maps(df,  thresholdalt=120000):
    hot_pix_maps_dict={}
    for channel in df['channel'].unique():
        #Make the hot pixel map for NADIR all zeros
        if channel == 'NADIR':
            print("Hot pixel map for Nadir set to all zeros.")
            hot_pix_maps_dict[channel] = np.zeros((df[df['channel']=='NADIR']['ImageCalibrated'].iloc[0].shape))
        else:
            hot_pix_maps_dict[channel] = create_hot_pix_map_one_channel(df[df.channel==channel],  thresholdalt=thresholdalt)
    return hot_pix_maps_dict




def hot_pix_removal_one_channel(df, hot_pix_map):
    # Check that all rows have the same channel
    unique_channels = df['channel'].unique()
    if len(unique_channels) != 1:
        raise ValueError("DataFrame contains multiple channels. Please provide data from a single channel.")

    df['ImageCalibrated_HPremoved'] = df['ImageCalibrated'].apply(lambda img: np.array(img) - hot_pix_map)
    return df

def hot_pix_removal_several_channels(df, hot_pix_maps_dict):
    df['ImageCalibrated_HPremoved'] = df.apply(
        lambda x: x.ImageCalibrated - hot_pix_maps_dict.get(x.channel, 0),
        axis=1
    )
    return df





def create_hot_pix_map_one_channel_using_midtangentpoint_only(df):
    """
    This is a faster version of creating hot pixels maps using mid tangent point only, 
    and then using a running mean filter to check when the signal is too high. It does not 
    work as well as the other method which only uses the part of the CCD that is in darkness
    """
    def apply_running_mean_filter(img, filter_rows, filter_cols, threshold):
        from scipy.ndimage import uniform_filter
        running_mean = uniform_filter(img, size=(filter_rows, filter_cols))
        img_filtered = img.copy().astype(float)
        img_filtered[running_mean > threshold] = np.nan
        return img_filtered

    def min_heights_of_sides(CCDitem):
        """
            #check the tangent points of the twwos sides, ie column=0 och column =NCOL, and take the minimum

        Input
        CCDitem: row of a dataframe with CCDitems

        Returns:
        th1: float with tangent height of lowest pixel on one side
        th2: flaat with tangent height of lowest pixel on the other side

        """
        from mats_utils.geolocation.coordinates import col_heights
        th1 = col_heights(CCDitem, 0, 1) # the one makes the function return the lowest pixel
        th2 = col_heights(CCDitem, CCDitem.NCOL, 1) # the one makes the function return the lowest pixel
        minheight=min(th1,th2)
        return minheight


    # Check that all rows have the same channel
    unique_channels = df['channel'].unique()
    if len(unique_channels) != 1:
        raise ValueError("DataFrame contains multiple channels. Please provide data from a single channel.")

    channel = unique_channels[0]

    # Set filter dimensions based on channel
    if channel in ['IR1', 'UV1', 'UV2', 'NADIR']:
        filter_rows, filter_cols, threshold = 5, 4, 30
    elif channel in ['IR2']:
        filter_rows, filter_cols, threshold = 5, 4, 30    
    elif channel in ['IR3', 'IR4']:
        filter_rows, filter_cols, threshold = 5, 2, 40 
    else:
        raise ValueError(f"Unknown channel: {channel}")


    df['minheight']=df.apply(lambda x: min_heights_of_sides(x),axis=1)
   
    filtered_images = df[df['minheight']> 60000]['ImageCalibrated'] #check that minimum tan height is above threshold
    #df.drop(columns=['minheight'], inplace=True)

    if filtered_images.empty:
        raise ValueError("No images meet the selection criteria.")


    image_arrays = [apply_running_mean_filter(np.array(img),filter_rows, filter_cols, threshold=threshold) for img in filtered_images]
    hot_pix_map = np.nanmean(image_arrays, axis=0)


    return hot_pix_map