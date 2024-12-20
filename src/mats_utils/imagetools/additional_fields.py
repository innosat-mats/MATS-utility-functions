
import numpy as np
def add_field_with_subtracted_rolling_mean(df, field, outfieldname,  window_before=20, window_after=20, skipbefore=0, skipafter=0):
    """
    Add a new field to the DataFrame that is the difference between the field and the rolling mean of the field.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the field to calculate the rolling mean for.
    field : str
        The name of the field to calculate the rolling mean for.
    outfieldname : str
        The name of the new field to add to the DataFrame.
    window_before : int
        The number of images before the current image to include in the rolling mean.
    window_after : int
        The number of images after the current image to include in the rolling mean.
    skipbefore : int
        The number of images to skip before the current image.
    skipafter : int 
        The number of images to skip after the current image.

        
    """

    # Create a new column to store the results
    df[outfieldname] = df[field].copy()
    
    # Iterate over each image in the DataFrame
    for i in range(len(df)):
        # Define the window range
        start = max(0, i - window_before)
        end = min(len(df), i + window_after + 1)

        combined_range = list(range(start, i - skipbefore)) + list(range(i + skipafter, end))
        
        # Calculate the rolling mean for the window
        rolling_mean = np.mean([df[field].iloc[j] for j in combined_range], axis=0)
        
        
        # Subtract the rolling mean from the current image
        df[outfieldname].iloc[i] = df[field].iloc[i] - rolling_mean

    return

def add_field_with_subtracted_rolling_mean2(df, field, outfieldname,  window_before=10, window_after=20, skipbefore=0, skipafter=0):
    #Alternative implementation of the function above
    def rolling_mean(images, window_before=10, window_after=20, skipbefore=0, skipafter=0):
        means = []
        for i in range(len(images)):
            # Define the window range
            start = max(0, i - window_before)
            end = min(len(images), i + window_after + 1)

            combined_range = list(range(start, i - skipbefore)) + list(range(i + skipafter, end))
            
            # Calculate the rolling mean for the window
            rolling_mean = np.mean([images[j] for j in combined_range], axis=0)
            means.append(rolling_mean)
        return means

    rolling_means = rolling_mean(df[field].tolist(), window_before=window_before, window_after=window_after, skipbefore=skipbefore, skipafter=skipafter)
    df[outfieldname] = [
        img - mean for img, mean in zip(df[field], rolling_means)
    ]
    