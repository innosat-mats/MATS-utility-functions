import numpy as np
import pandas as pd

def remove_flagged_images(df, bits_to_remove, return_mask=False):
    """
    Remove all images (rows in df) that have any pixel flagged with any of the specified error bits.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the calibration data. Must include a 'CalibrationErrors' column.
    bits_to_remove : list of int
        List of bit numbers to check for error flags (e.g., [8, 9, 12]).
    return_mask : bool, optional
        If True, also return the boolean mask indicating which images were removed.

    Returns
    -------
    df_clean : pandas.DataFrame
        DataFrame with flagged images removed.
    mask : pandas.Series of bool (only if return_mask=True)
        Boolean mask where True indicates the image was removed.
    """

    bit_info = {
        1: 'Bad column present in image',
        2: 'Single event corrected',
        3: 'Hot pixel corrected',
        4: 'No hot pixel correction done',
        5: 'Negative values after bias subtraction',
        6: 'Nonlinear correction >5% applied',
        7: 'Pixel saturated in summation well or readout',
        8: 'Desmear subtraction rendered negative result',
        9: 'Desmear failed due to unrealistic atmospheric parameters',
        10: 'Dark current subtraction rendered negative result',
        11: 'Extreme temperature (30°C)',
        12: 'No temperature reading; default -15°C used',
        13: 'Flatfield correction rendered negative value',
        14: 'Flatfield compensation factor abnormally large (>5%)'
    }

    for bit in bits_to_remove:
        if bit not in bit_info:
            raise ValueError(f"Unsupported bit: {bit}. Supported bits are 1–14.")

    combined_mask = 0
    for bit in bits_to_remove:
        combined_mask |= (1 << (bit - 1))

    def has_flagged_pixel(error_array):
        error_array = np.stack(error_array)  # Convert to proper 2D array
        return np.any((error_array & combined_mask) != 0)

    mask = df['CalibrationErrors'].apply(has_flagged_pixel)
    df_clean = df[~mask].reset_index(drop=True)

    if return_mask:
        return df_clean, mask
    else:
        return df_clean



def count_error_flags(df, xpix=None, ypix=None, bit=None, outputallindices=False, printinfo=False):
    """
    Count the number of error flags for a specific bit or all bits in the CalibrationErrors field of the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the calibration data.
    xpix : int
        Pixel x-coordinate.
    ypix : int
        Pixel y-coordinate.
    bit : int, optional
        Bit number to check for errors. If None, check all bits.

    Returns
    -------
    mask : pandas.Series
        Series containing the positions of raised error flags for the specified error bit.
    mask_summed : pandas.Series
        Series containing the sum of the error flags for
        each the selected pixel or the entire image.
    """
    bit_info = {
        1: 'Flag set to 1 across image if bad column is present in image',
        2: 'Flag to indicate that single event that has been corrected',
        3: 'Flag to indicate a hot pixel, which has been corrected',
        4: 'Flag to indicate that no hot pixel correction has been done to the image',
        5: 'Flag to indicate that negative values appeared after bias subtraction',
        6: 'Flag to indicate that nonlinear correction of more than 5% has been applied for pixel',
        7: 'Flag to indicate that pixel is saturated in summation well, read out register or single pixel',
        8: 'Flag to indicate that the desmear subtraction rendered negative result',
        9: 'Flag to indicate that the desmear subtraction could not estimate realistic atmospheric parameters needed for desmearing, and thus no desmearing has been done',
        10: 'Flag to indicate that the dark current subtraction rendered negative result',
        11: 'Flag for extreme temperature (30C)',
        12: 'Flag indicating no temperature reading and default temperature of -15 C has been used',
        13: 'Flag to indicate flatfield correction rendered negative value',
        14: 'Flag to indicate abnormally large (>5%) flatfield compensation factor'
    }

    if bit is not None:
        if bit not in bit_info:
            raise ValueError("Unsupported bit provided")
        bits_to_check = [bit]
    else:
        bits_to_check = list(bit_info.keys())



    for bit in bits_to_check:
        index = bit - 1
        if printinfo:
            print('Error flag bit:', bit, 'Info:', bit_info[bit])
        
        if xpix is not None and ypix is not None:
            # Check the specified bit for pixel xpix, ypix across all images
            mask = df['CalibrationErrors'].apply(lambda x: x[ypix, xpix] & (1 << index) != 0)
        elif xpix is None and ypix is None:
            # Check the specified bit for all pixels across all images
            #mask = df['CalibrationErrors'].apply(lambda x: np.any(x & (1 << index) != 0))
            mask = df['CalibrationErrors'].apply(lambda x: x & (1 << index) != 0)
        elif xpix is not None and ypix is None:
            # Check the specified bit for all pixels in column xpix across all images
            mask = df['CalibrationErrors'].apply(lambda x: x[:, xpix] & (1 << index) != 0)
        elif xpix is None and ypix is not None:
            # Check the specified bit for all pixels in row ypix across all images
            mask = df['CalibrationErrors'].apply(lambda x: x[ypix, :] & (1 << index) != 0)

        # Convert all true values in mask to ones and all false values to zero
        
        mask_summed = mask.apply(lambda x: np.where(x, 1, 0)).sum()




    return mask, mask_summed
