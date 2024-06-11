#%%
from mats_l1_processing import L1_calibrate
from mats_l1_processing.instrument import Instrument
from mats_l1_processing.read_parquet_functions import dataframe_to_ccd_items
from pandas import concat, DataFrame

def calibrate_dataframe(ccd_data_in: DataFrame, instrument: Instrument, debug_outputs: bool = False):
    """Calibrate l1a dataframe
    Takes in a l1a dataframe read via read_MATS_data (so with add_ccd_attributes) 
    and and instrument object. Then calibrates the data and returns l1b data as a Pandas Dataframe as though it was downloaded using read_MATS_data.

    Args:
        ccd_data (DataFrame):   l1a dataframe
        instrument (Instrument): MATS instrument object

    Returns:
        l1b_data (DataFrame):   Dataframe containing l1b data
    """
    ccd_data = ccd_data_in.drop(["IMAGE","flipped","temperature_HTR","temperature_ADC","id"],axis=1)    
    #ccd_data.reset_index(inplace=True)
    ccd_items = dataframe_to_ccd_items(
            ccd_data,
            remove_empty=False,
            remove_errors=False,
            remove_warnings=False,
        )

    for ccd in ccd_items:
        if ccd["IMAGE"] is None:
            image_calibrated = None
            errors = None
        else:
            (
                image_lsb, 
                image_se_corrected, 
                image_hot_pixel_corrected, 
                image_bias_sub, 
                image_linear, 
                image_desmeared, 
                image_dark_sub, 
                image_flatfielded, 
                image_flipped, 
                image_calibrated, 
                errors
            ) = L1_calibrate.L1_calibrate(ccd, instrument, force_table=False, return_steps=True)
            ccd["ImageCalibrated"] = image_calibrated
            ccd["CalibrationErrors"] = errors

            if debug_outputs:
                ccd["image_lsb"] = image_lsb
                ccd["image_se_corrected"] = image_se_corrected
                ccd["image_hot_pixel_corrected"] = image_hot_pixel_corrected
                ccd["image_bias_sub"] = image_bias_sub
                ccd["image_linear"] = image_linear
                ccd["image_desmeared"] = image_desmeared
                ccd["image_dark_sub"] = image_dark_sub
                ccd["image_flatfielded"] = image_flatfielded
                ccd["image_flipped"] = image_flipped
                





    if not debug_outputs:
        calibrated = DataFrame.from_records(
            ccd_items,
            columns=[
                "ImageCalibrated",
                "CalibrationErrors",
                "qprime",
                "channel",
                "flipped",
                "temperature",
                "temperature_HTR",
                "temperature_ADC",
            ],
        )
    else:
        calibrated = DataFrame.from_records(
            ccd_items,
            columns=[
                "ImageCalibrated",
                "CalibrationErrors",
                "qprime",
                "flipped",
                "temperature",
                "temperature_HTR",
                "temperature_ADC",
                "image_lsb",
                "image_se_corrected",
                "image_hot_pixel_corrected",
                "image_bias_sub",
                "image_linear",
                "image_desmeared",
                "image_dark_sub",
                "image_flatfielded",
                "image_flipped",
            ],
        )   

    calibrated.set_index(ccd_data.index,inplace=True)
    l1b_data = concat([
        ccd_data,
        calibrated,
    ], axis=1)
    l1b_data.set_index("TMHeaderTime").sort_index()
    l1b_data.reset_index()
    l1b_data.drop(["ImageData", "Errors", "Warnings"], axis=1, inplace=True)
    l1b_data = l1b_data[l1b_data.ImageCalibrated != None]  # noqa: E711
#    l1b_data["ImageCalibrated"] = [
#        ic.tolist() for ic in l1b_data["ImageCalibrated"]
#    ]
#    l1b_data["CalibrationErrors"] = [
#        ce.tolist() for ce in l1b_data["CalibrationErrors"]
#    ]

    return l1b_data
