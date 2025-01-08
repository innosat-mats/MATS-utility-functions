import numpy as np
from pandas import DataFrame
from xarray import Coordinates, Variable


def get_coords(df: DataFrame | None, config: dict[str, any]) -> Coordinates:
    empty = (df is None) or df.empty
    coords = Coordinates(
        {
            "im_col": np.arange(config["NCOL"] + 1, dtype="int32"),
            "im_row": np.arange(config["NROW"], dtype="int32"),
            "time": np.empty(0, dtype="datetime64[ns]") if empty else df.EXPDate,
            "quaternion": np.arange(4, dtype="int8"),
            "eci_pos": np.arange(3, dtype="int8"),
            "gnss_state": np.arange(6, dtype="int8"),
        }
    )
    if empty:
        coords["im_col"].attrs = {
            "units": "",
            # "standard_name": "im_col",
            "long_name": "Image Column Number",
        }
        coords["im_row"].attrs = {
            "units": "",
            # "standard_name": "im_row",
            "long_name": "Image Row Number",
        }
        coords["time"].attrs = {
            "standard_name": "time",
            "long_name": "Time of Observation",
        }
        coords["quaternion"].attrs = {
            "units": "",
            # "standard_name": "quaternion",
            "long_name": "Quaternion Components",
        }
        coords["eci_pos"].attrs = {
            "units": "",
            # "standard_name": "eci_pos",
            "long_name": "ECI Position Components",
        }
        coords["gnss_state"].attrs = {
            "units": "",
            # "standard_name": "gnss_state",
            "long_name": "GNSS State Components",
        }

    return coords


def get_data_vars(df: DataFrame | None, config: dict[str, any]) -> dict[str, Variable]:
    empty = (df is None) or df.empty
    return {
        "ImageCalibrated": Variable(
            dims=["time", "im_row", "im_col"],
            data=(
                np.empty((0, config["NROW"], config["NCOL"] + 1), dtype="float32")
                if empty
                else np.stack(df.ImageCalibrated.apply(lambda x: np.stack(x))).astype(
                    "float32"
                )
            ),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Calibrated image (spectral radiance)",
                    "standard_name": "spectral_radiance",
                    "units": "photon nanometer-1 meter-2 steradian-1 second-1",
                }
            ),
            encoding={"dtype": "float32"},
        ),
        "CalibrationErrors": Variable(
            dims=["time", "im_row", "im_col"],
            data=(
                np.empty((0, config["NROW"], config["NCOL"] + 1), dtype="int32")
                if empty
                else np.stack(df.CalibrationErrors.apply(lambda x: np.stack(x))).astype(
                    "int32"
                )
            ),
            attrs=(
                None
                if not empty
                else {
                    "description": "Error flags combined into single value (see documentation)",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "afsAttitudeState": Variable(
            dims=["time", "quaternion"],
            data=(
                np.empty((0, 4), dtype="float64")
                if empty
                else np.stack(df.afsAttitudeState)
            ),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Quaternion relating satelite body frame to Earth-centered inertial (ECI) coordinate",
                    # "standard_name": "afsAttitudeState",
                    "units": "",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "qprime": Variable(
            dims=["time", "quaternion"],
            data=np.empty((0, 4), dtype="float64") if empty else np.stack(df.qprime),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Quaternion relating channel to satellite body frame",
                    # "standard_name": "qprime",
                    "units": "",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "afsGnssStateJ2000": Variable(
            dims=["time", "gnss_state"],
            data=(
                np.empty((0, 6), dtype="float64")
                if empty
                else np.stack(df.afsGnssStateJ2000)
            ),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "GNSS state in J2000 frame",
                    # "standard_name": "afsGnssStateJ2000",
                    "units": "",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        # "afsTangentPointECI": Variable(
        #     dims=["time", "eci_pos"],
        #     data=(
        #         np.empty((0, 3), dtype="float64")
        #         if empty
        #         else np.stack(df.afsTangentPointECI)
        #     ),
        #     attrs=(
        #         None
        #         if not empty
        #         else {
        #             "long_name": "Tangent point in ECI frame",
        #             # "standard_name": "afsTangentPointECI",
        #             "units": "",
        #         }
        #    ),
        #    encoding={"dtype": "float64"},
        # ),
        # "afsTPLongGeod": Variable(
        #     dims=["time"],
        #     data=(
        #         np.empty((0,), dtype="float64")
        #         if empty
        #         else df.afsTPLongLatGeod.apply(lambda x: x[0])
        #     ),
        #    attrs=(
        #        None
        #         if not empty
        #         else {
        #             "long_name": "Tangent point longitude in geodetic coordinates",
        #            # "standard_name": "afsTPLongGeod",
        #            "units": "degree_east",
        #         }
        #    ),
        #     encoding={"dtype": "float64"},
        # ),
        # "afsTPLatGeod": Variable(
        #    dims=["time"],
        #    data=(
        #         np.empty((0,), dtype="float64")
        #         if empty
        #         else df.afsTPLongLatGeod.apply(lambda x: x[1])
        #     ),
        #     attrs=(
        #         None
        #         if not empty
        #         else {
        #            "long_name": "Tangent point latitude in geodetic coordinates",
        #            # "standard_name": "afsTPLatGeod",
        #             "units": "degree_north",
        #        }
        #    ),
        #     encoding={"dtype": "float64"},
        # ),
        "satheight": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.satheight.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Satellite altitude at time of measurement",
                    # "standard_name": "satheight",
                    "units": "meter",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "satlon": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.satlon.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Satellite longitude at time of measurement",
                    "standard_name": "deployment_longitude",
                    "units": "degree_east",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "satlat": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.satlat.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Satellite latitude at time of measurement",
                    "standard_name": "deployment_latitude",
                    "units": "degree_north",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "TPheight": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.TPheight.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Altitude of tangent point for nominal instrument axis at the time of measurement",
                    # "standard_name": "TPheight",
                    "units": "meter",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "TPlon": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.TPlon.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Longitude of tangent point for nominal instrument axis at the time of measurement",
                    # "standard_name": "TPlon",
                    "units": "degree_east",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "TPlat": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.TPlat.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Latitude of tangent point for nominal instrument axis at the time of measurement",
                    # "standard_name": "TPlat",
                    "units": "degree_north",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "JPEGQ": Variable(
            dims=[],
            data=config["JPEGQ"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "JPEG compression quality setting",
                    # "standard_name": "JPEGQ",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "TEXPMS": Variable(
            dims=[],
            data=config["TEXPMS"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Image exposure duration",
                    "units": "millisecond",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "schedule_id": Variable(
            dims=[],
            data=config["schedule_id"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Schedule ID",
                    # "standard_name": "schedule_id",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "schedule_version": Variable(
            dims=[],
            data=config["schedule_version"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Version of schedule",
                    # "standard_name": "schedule_version",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "schedule_yaw_correction": Variable(
            dims=[],
            data=config["schedule_yaw_correction"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Scheduled yaw correction (1 if True)",
                    # "standard_name": "schedule_yaw_correction",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "nadir_az": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.nadir_az.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Solar azimuth angle at satellite position",
                    # "standard_name": "nadir_az",
                    "units": "degree",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "nadir_sza": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.nadir_sza.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Solar zenith angle at satellite position",
                    # "standard_name": "nadir_sza",
                    "units": "degree",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "TPssa": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.TPssa.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Solar scattering angle at tangent point",
                    "standard_name": "scattering_angle",
                    "units": "degree",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "TPsza": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.TPsza.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Solar zenith angle at tangent point",
                    "standard_name": "solar_zenith_angle",
                    "units": "degree",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "temperature": Variable(
            dims=["time"],
            data=(np.empty((0,), dtype="float64") if empty else df.temperature.values),
            attrs=(
                None
                if not empty
                else {
                    "long_name": "CCD housing temperature",
                    # "standard_name": "temperature",
                    "units": "degree_celsius",
                }
            ),
            encoding={"dtype": "float64"},
        ),
        "NCBINCCDColumns": Variable(
            dims=[],
            data=config["NCBINCCDColumns"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Number of columns binned together on-chip",
                    # "standard_name": "NCBINCCDColumns",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "NCBINFPGAColumns": Variable(
            dims=[],
            data=config["NCBINFPGAColumns"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Number of columns binned together in FPGA",
                    # "standard_name": "NCBINFPGAColumns",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "NCOL": Variable(
            dims=[],
            data=config["NCOL"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Number of columns in the image",
                    # "standard_name": "NCOL",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "NCSKIP": Variable(
            dims=[],
            data=config["NCSKIP"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Number of columns skipped before binning",
                    # "standard_name": "NCSKIP",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "NRBIN": Variable(
            dims=[],
            data=config["NRBIN"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Number of rows that are binned together",
                    # "standard_name": "NRBIN",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "NROW": Variable(
            dims=[],
            data=config["NROW"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Number of rows in the image",
                    # "standard_name": "NROW",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
        "NRSKIP": Variable(
            dims=[],
            data=config["NRSKIP"],
            attrs=(
                None
                if not empty
                else {
                    "long_name": "Number of rows skipped before binning",
                    # "standard_name": "NRSKIP",
                    "units": "",
                }
            ),
            encoding={"dtype": "int32"},
        ),
    }
