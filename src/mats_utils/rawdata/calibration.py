#%%
from mats_l1_processing import L1_calibrate
from mats_l1_processing.instrument import Instrument
from mats_l1_processing.read_parquet_functions import dataframe_to_ccd_items
from pandas import concat, DataFrame

def calibrate_dataframe(ccd_data_in: DataFrame, instrument: Instrument):
    """Calibrate l1a dataframe
    Takes in a l1a dataframe read via read_MATS_data (so with add_ccd_attributes) 
    and and instrument object. Then calibrates the data and returns l1b data as a Pandas Dataframe as though it was downloaded using read_MATS_data.

    Args:
        ccd_data (DataFrame):   l1a dataframe
        instrument (Instrument): MATS instrument object

    Returns:
        l1b_data (DataFrame):   Dataframe containing l1b data
    """

    ccd_data = ccd_data_in.drop(["IMAGE","channel","flipped","temperature","temperature_HTR","temperature_ADC","id"],axis=1)
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
                _,
                _,
                _,
                _,
                _,
                image_calibrated,
                errors,
            ) = L1_calibrate.L1_calibrate(ccd, instrument, force_table=False)
            ccd["ImageCalibrated"] = image_calibrated
            ccd["CalibrationErrors"] = errors


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
    calibrated.set_index(ccd_data.index,inplace=True)
    l1b_data = concat([
        ccd_data,
        calibrated,
    ], axis=1)
    l1b_data.set_index("EXPDate").sort_index()
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
