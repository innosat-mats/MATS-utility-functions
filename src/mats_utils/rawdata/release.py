import netCDF4 as nc
import numpy as np
import datetime as DT
import pandas


def time2seconds(times):
    if type(times[0]) is np.datetime64:
        return (times - np.datetime64("2000-01-01 00:00:00.0")) / np.timedelta64(1, "s")
    elif type(times[0]) is pandas._libs.tslibs.timestamps.Timestamp:
        return (times - DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc)) / DT.timedelta(0, 1)
    else:
        raise ValueError(f"Unrecognised data type for timestamps: {type(times[0])}!")


def write_ncdf_L1b_release(pdata, outfile, channel, version, rversion):
    with nc.Dataset(outfile, 'w') as nf:
        # Create dimensions
        nf.createDimension("time", len(pdata))
        nf.createDimension("im_col", pdata["NCOL"][0] + 1)
        nf.createDimension("im_row", pdata["NROW"][0])
        nf.createDimension("quaternion", 4)
        nf.createDimension("eci_pos", 3)
        nf.createDimension("gnss_state", 6)
        nf.createDimension("str", 1)

        # Define variables (<variable name>, <ncdf data type>, <dimensions>, <long name>, <units>, [values])
        # If values can be handled trivially (i.e. pdata[<variable name>].to_numpy()), skip it.
        # Scalars (variables with no dimensions) are tested to ensure that all values in pandas rows are identical,
        # then that value is written to the file.
        var = [("time", "float64", ("time",),
                "Time of exposure", "seconds since 2000.01.01 00:00 UTC",
                time2seconds(pdata["EXPDate"].to_numpy())),
               ("im_col", "i4", ("im_col",),
                "Image column number", "dimensionless",
                np.arange(pdata["NCOL"][0] + 1, dtype=np.int32)),
               ("im_row", "i4", ("im_row",),
                "Image row number", "dimensionless",
                np.arange(pdata["NROW"][0], dtype=np.int32)),
               ("ImageCalibrated", "f4", ("time", "im_row", "im_col"),
                "Calibrated image (spectral radiance)", "photon nanometer-1 meter-2 steradian-1 second-1",
                np.stack(pdata["ImageCalibrated"].to_numpy())),
               ("CalibrationErrors", "i4", ("time", "im_row", "im_col"),
                "Error flags combined into single value (see documentation)", "dimensionless",
                [np.stack(pdata["CalibrationErrors"][i], axis=0)
                    for i in range(len(pdata["CalibrationErrors"]))]),
               ("afsAttitudeState", "f8", ("time", "quaternion"),
                "?", "?",
                np.stack(pdata["afsAttitudeState"].to_numpy())),
               ("qprime", "f8", ("time", "quaternion"),
                "?", "?",
                np.stack(pdata["qprime"].to_numpy())),
               ("afsGnssStateJ2000", "f8", ("time", "gnss_state"),
                "?", "?",
                np.stack(pdata["afsGnssStateJ2000"].to_numpy())),
               ("afsTangentPointECI", "f8", ("time", "eci_pos"),
                "?", "?",
                np.stack(pdata["afsTangentPointECI"].to_numpy())),
               ("afsTPLongGeod", "f8", ("time"),
                "?", "?",
                np.stack(pdata["afsTPLongLatGeod"].to_numpy())[:, 0]),
               ("afsTPLatGeod", "f8", ("time"),
                "?", "?",
                np.stack(pdata["afsTPLongLatGeod"].to_numpy())[:, 1]),
               ("afsTangentH_wgs84", "f8", ("time"),
                "?", "kilometer"),
               ("satheight", "f8", ("time"),
                "Satellite altitude at time of measurement", "meter"),
               ("satlon", "f8", ("time"),
                "Satellite longitude at time of measurement", "degree_east"),
               ("satlat", "f8", ("time"),
                "Satellite latitude at time of measurement", "degree_north"),
               ("TPheight", "f8", ("time"),
                "Altitude of tangent point for nominal instrument axis at the time of measurement",
                "meter"),
               ("TPlon", "f8", ("time"),
                "Longitude of tangent point for nominal instrument axis at the time of measurement",
                "degree_east"),
               ("TPlat", "f8", ("time"),
                "Latitude of tangent point for nominal instrument axis at the time of measurement",
                "degree_north"),
               ("JPEGQ", "i4", ("time"),
                "JPEG compression quality setting", "dimensionless"),
               ("TEXPMS", "i4", ("time"),
                "Exposure time used for measurement", "milisecond"),
               ("id", str, ("time", "str"),
                "Measurement ID", "dimensionless"),
               ("schedule_id", "i4", ("time"),
                "Schedule ID", "dimensionless"),
               ("schedule_name", str, ("time", "str"),
                "Name of schedule", "dimensionless"),
               ("schedule_version", 'i4', ("time"),
                "Version of schedule", "dimensionless"),
               ("schedule_yaw_correction", 'i1', ("time"),
                "Scheduled yaw correction (1 if True)", "dimensionless"),
               ("nadir_az", "f8", ("time"),
                "Solar azimuth angle at satellite position", "degree"),
               ("nadir_sza", "f8", ("time"),
                "Solar zenith angle at satellite position", "degree"),
               ("TPssa", "f8", ("time"),
                "Solar scattering angle at tangent point", "degree"),
               ("TPsza", "f8", ("time"),
                "Solar zenith angle at tangent point", "degree"),
               ("temperature", "f8", ("time"),
                "CCD housing temperature", "degree_celsius"),
               ("day", "i4", ("time"),
                "Day of the month (UTC) of measurement", "dimensionless"),
               ("month", "i4", ("time"),
                "Month (UTC) of measurement", "dimensionless"),
               ("year", "i4", ("time"),
                "Year (UTC) of measurement", "dimensionless"),
               ("hour", "i4", ("time"),
                "Hour (UTC) of measurement", "dimensionless"),
               ("TPlocaltime", str, ("time", "str"),
                "Local solar time at tangent point for nominal instument axis at the time of measurement",
                "dimensionless"),
               ("NCBINCCDColumns", "i4", (), "Number of columns binned together on-chip", "dimensionless"),
               ("NCBINFPGAColumns", "i4", (), "Number of columns binned together in FPGA", "dimensionless"),  # Change needed!
               ("NCOL", "i4", (), "Number of columns in the image", "dimensionless"),
               ("NCSKIP", "i4", (), "Number of columns skipped before binning", "dimensionless"),
               ("NRBIN", "i4", (), "Number of rows that are binned together", "dimensionless"),
               ("NROW", "i4", (), "Number of rows in the image", "dimensionless"),
               ("NRSKIP", "i4", (), "Number of rows skipped before binning", "dimensionless"),
              ]

        # Define global attributes that will be set from pandas variables
        # Pandas values will be tested to ensure they are all the same
        attrs = ["DataLevel", "L1BCode", "L1ACode"]
        # Write variables
        for v in var:
            name, dtype, dims, lname, units = v[:5]
            ncvar = nf.createVariable(name, dtype, dims)
            ncvar.long_name = lname
            ncvar.units = units
            if len(dims) == 0:
                assert all([pdata[name][0] == val for val in pdata[name]]), \
                    f"Variable {name} is defined as scalar, but its values are not the same for all images!"
                ncvar[:] = v[5] if len(v) == 6 else pdata[name][0]
            else:
                ncvar[:] = v[5] if len(v) == 6 else pdata[name].to_numpy()

        # Additional variable attributes
        nf["ImageCalibrated"].scale_factor = 1e12

        # Write attributes defined above
        for att in attrs:
            assert all([pdata[name][0] == val for val in pdata[name]]), \
                f"Variable {name} is defined as global attribute, but its values are not the same for all images!"
            nf.setncattr(att, pdata[name][0])

        # Write general attributes
        nf.date_created = DT.datetime.now().isoformat()
        nf.date_modified = DT.datetime.now().isoformat()
        nf.channel = channel
        nf.data_version = version
        nf.release_version = rversion
