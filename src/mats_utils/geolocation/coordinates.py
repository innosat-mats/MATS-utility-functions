from skyfield.framelib import itrs
from skyfield.positionlib import Geocentric, ICRF
from skyfield.units import Distance
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator
from skyfield.api import wgs84
from skyfield.api import load
from skyfield.units import Distance
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
from mats_l1_processing.pointing import pix_deg
import numpy as np
import datetime as DT
from datetime import datetime, timedelta, timezone
from pyarrow import fs
import boto3
import pyarrow as pa  # type: ignore
import pyarrow.dataset as ds  # type: ignore
from pandas import DataFrame, Timestamp  # type: ignore
from skyfield import api as sfapi
from scipy.interpolate import RegularGridInterpolator
import math


def get_deg_map(image):
    """
    Function for calculating pixel positions in terms
    of euler angles within the frame of the specific channel

    Arguments:
        image (pandas row or dict)

    Returns:
        ndarray, the two Euler angles as 2D grids, stacked in third dim.
    """

    dmap = np.zeros((image["NCOL"] + 1, image["NROW"], 2))
    for i in range(dmap.shape[0]):
        for j in range(dmap.shape[1]):
            dmap[i, j, :] = pix_deg(image, i, j)
    return dmap


def make_ref_grid(df, sample_factor=0.5, ext_factor=1.1):
    """
    This calculates an (example of) common grid for MATS images in terms
    of euler angles in satellite body frame. Can be stored/edited before
    actually interpolating to it.

    Arguments:
        df - Pandas data frame with at least one IR1 image;
        sample_factor - the grid constant for the common grid will be
              IR1 grid constant * sample_factor. Default: 0.5
        ext_factor - defines the size of margins around image, needed to
              accomodate the other channels. Ref. grid size = IR1 size * ext_factor
    Returns:
        deg_map - ndarray, the two Euler angles as 2D grids, stacked in third dim.
        grids - tuple, the two Euler angles as 1-D arrays
    """

    # Get IR1 grid as a prototype
    ir1_images = df[df['CCDSEL'] == 1]
    assert len(ir1_images) > 0, "There are no IR1 images in the selected dataset, choose a longer one."
    IR1_qprime = R.from_quat(ir1_images.iloc[0]["qprime"])
    IR1_deg_map = get_deg_map(ir1_images.iloc[0])
    conv_fix, _ = R.align_vectors([[1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, -1, 0]])
    xgrid_IR1, ygrid_IR1 = IR1_deg_map[:, 0, 0], IR1_deg_map[0, :, 1]

    # Find the positions of IR1 grid corners in body frame
    corner_ang = np.zeros((2, 2, 2))
    for i in [0, -1]:
        for j in [0, -1]:
            r = conv_fix * IR1_qprime * R.from_euler('xyz', [0, IR1_deg_map[i, j, 1],
                                                     IR1_deg_map[i, j, 0]], degrees=True)
            _, corner_ang[i, j, 1], corner_ang[i, j, 0] = r.as_euler('xyz', degrees=True)

    # Construct xgrid and ygrid
    grids = []
    for c, grid1 in enumerate([xgrid_IR1, ygrid_IR1]):
        step = sample_factor * np.diff(grid1).mean()
        extent = np.abs(corner_ang[:, :, c].min() - corner_ang[:, :, c].max())
        start = corner_ang[:, :, c].min() - (ext_factor - 1) * extent / 2
        extent *= ext_factor
        gstart, gstop = [np.floor(x / step) * step for x in [start, start + extent]]
        grids.append(np.arange(gstart, gstop + step / 2, step))
    yy, xx = np.meshgrid(grids[1], grids[0])
    deg_map = np.zeros((xx.shape[0], xx.shape[1], 2))
    deg_map[:, :, 0] = xx
    deg_map[:, :, 1] = yy
    return deg_map, grids


def to_ref(df, chn, ref_map, time_ref_chn=None):
    """
    Reinterpolates MATS images onto reference grid.

    Arguments:
        df - Pandas data frame;
        chn - string, name of the channel to reinterpolate, e.g. "IR1".
        ref_map - ndarray, 2-D Euler angle reference grid. Shape is (num. cols, num. rows, 2).
        time_ref_chn - bool, not implemented yet.
    Returns:
        ndarray, images on reference grid.
    """

    # Select images for the specified channel
    CCDSEL = {"IR1": 1, "IR2": 4, "IR3": 3, "IR4": 2, "UV1": 5, "UV2": 6}
    sel = df[df['CCDSEL'] == CCDSEL[chn]]
    numimg = len(sel)
    assert numimg > 0

    # Map ref. grid onto channel
    ref_angs = np.zeros_like(ref_map)
    chn_qprime = R.from_quat(sel.iloc[0]["qprime"])
    chn_deg_map = get_deg_map(sel.iloc[0])
    chn_grids = (chn_deg_map[:, 0, 0], chn_deg_map[0, :, 1])
    conv_fix, _ = R.align_vectors([[1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, -1, 0]])
    fix_rot = R.inv(conv_fix * chn_qprime)

    for i in range(ref_map.shape[0]):
        for j in range(ref_map.shape[1]):
            r = fix_rot * R.from_euler('xyz', [0, ref_map[i, j, 1], ref_map[i, j, 0]], degrees=True)
            _, ref_angs[i, j, 1], ref_angs[i, j, 0] = r.as_euler('xyz', degrees=True)
            # pix_vec = r.apply([1, 0, 0])
            # ref_angs[i, j, :] = [np.rad2deg(np.arcsin(pix_vec[i])) for i in [1, 2]]

    # Interpolate
    res = np.zeros((numimg, ref_map.shape[0], ref_map.shape[1]))
    for k in range(numimg):
        img = sel.iloc[k]["ImageCalibrated"].T
        interp = RegularGridInterpolator(chn_grids, img, method='cubic', bounds_error=False, fill_value=np.nan)
        res[k, :, :] = interp((ref_angs[:, :, 0], ref_angs[:, :, 1]))
    return res


def calc_nadir_orbplane_angles(df, chn, ref_map):
    """
    Calculates angle to nadir and angle to orbital plane (defined as perpenticular to r x v).

    Arguments:
        df - Pandas data frame with MATS data;
        chn - string, name of the channel (e.g. "IR1") that determines the times for which angles are calculated;
        ref_map - ndarray, 2-D Euler angle reference grid. Shape is (num. cols, num. rows, 2).
    Returns:
        nadir_angle, orbplane_angle - 2D darrays, respective angles for the reference grid.
    """

    # Select images for the specified channel
    CCDSEL = {"IR1": 1, "IR2": 4, "IR3": 3, "IR4": 2, "UV1": 5, "UV2": 6}
    sel = df[df['CCDSEL'] == CCDSEL[chn]]
    numimg = len(sel)

    # Calculate angles
    nadir_angle = np.zeros((numimg, ref_map.shape[0], ref_map.shape[1]))
    orbplane_angle = nadir_angle.copy()
    conv_fix, _ = R.align_vectors([[1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, -1, 0]])

    for k in range(numimg):
        align_rot = to_ref_att(sel.iloc[k]["afsGnssStateJ2000"])
        rot = align_rot * R.from_quat(np.roll(sel.iloc[k]["afsAttitudeState"], -1)) * conv_fix.inv()
        for i in range(ref_map.shape[0]):
            for j in range(ref_map.shape[1]):
                pix_vec = (rot * R.from_euler('xyz', [0, ref_map[i, j, 1],
                                              ref_map[i, j, 0]], degrees=True)).apply([1, 0, 0])
                nadir_angle[k, i, j] = np.rad2deg(np.arccos(-pix_vec[2]))
                orbplane_angle[k, i, j] = np.rad2deg(np.arccos(pix_vec[1])) - 90
    return nadir_angle, orbplane_angle


def to_ref_att(gnss):
    """
    Calculates the roatation needed to go from body frame to a frame where: z axis is towards zenith,\
    sat. velocity is aligned with x axis and angular momentum (r x v) is aligned with y axis.

    Arguments:
        gnss - an array of 6 components, GNSS-based statelite position ("afsGnssStateJ2000").
    Returns:
        scipy rotation
    """

    rad_vec = gnss[:3] / np.linalg.norm(gnss[:3])
    norm_vec = np.cross(rad_vec, gnss[3:])
    norm_vec /= np.linalg.norm(norm_vec)
    res, rssd = R.align_vectors([[0, 1, 0], [0, 0, 1]], [norm_vec, rad_vec])
    return res


def get_temporal_rotation(df, data_chn, ref_chn):
    """
    Calculates rotation corrections needed to compensate for temporal mismatch between channels.


    Arguments:
        df - Pandas data frame with MATS data;
        data_chn - corrections will be calculated for data of this channel;
        ref_chn - corrections will be calculated for interpolating data_chn data onto ref_chn times.
    Returns:
        trots - list of scipyrotations;
        times = list of datetime objects representimng times for which corrections are calculated.
    """

    assert data_chn != ref_chn

    # Get timestamps and sat. pointing for data data channel and ref. channel images within selection
    CCDSEL = {"IR1": 1, "IR2": 4, "IR3": 3, "IR4": 2, "UV1": 5, "UV2": 6}
    sat_att, data_time, ref_time = [df[df['CCDSEL'] == CCDSEL[chn]][var] for chn, var in
                                    [(data_chn, "afsAttitudeState"), (data_chn, "EXPDate"), (ref_chn, "EXPDate")]]
    assert len(ref_time) > 2
    assert len(data_time) > 2
    ref_time_og = ref_time
    ref_tp = ref_time[0]
    data_time, ref_time = [(arr - ref_tp).total_seconds() for arr in [data_time, ref_time]]

    # Check that both channels have the same temporal cadence
    data_step, ref_step = [np.diff(arr).mean() for arr in [data_time, ref_time]]
    assert np.abs(data_step / ref_step - 1) < 0.01

    # Initialize rotation interpolator
    rots = Slerp(data_time, [R.from_quat(q) for q in sat_att])

    # Assign each image to appropriate reference time and interpolate
    idxs = [np.argmin(np.abs(d - ref_time)) for d in data_time]
    trots, times = [], []
    for i, idx in enumerate(idxs):
        out = (data_time[i] < ref_time[0]) or (data_time[i] > ref_time[-1])
        misaligned = np.abs((data_time[i] - ref_time[idx]) / ref_step) > 0.5
        if not (out or misaligned):
            trots.append(R.from_quat(sat_att[i]).inv() * rots(ref_time[idx]))
            times.append(ref_time_og[idx])
    return trots, times


def funheight(s, t, pos, FOV):
    newp = pos + s * FOV
    newp = ICRF(Distance(m=newp).au, t=t, center=399)
    return wgs84.subpoint(newp).elevation.m


def meanquaternion(start_date: datetime, deltat: timedelta):
    """
    Function giving the mean quaternion during a
    time period


    Arguments:
        Start_date and delta t in date time format

    Returns:
        mean quaternion

    """
    session = boto3.session.Session(profile_name="mats")
    credentials = session.get_credentials()

    s3 = fs.S3FileSystem(
        secret_key=credentials.secret_key,
        access_key=credentials.access_key,
        region=session.region_name,
        session_token=credentials.token)

    dataset = ds.dataset(
        "ops-platform-level1a-v0.3/ReconstructedData",
        filesystem=s3,)
    # if start_date.tzinfo == None:
    #    start_date = start_date.replace(tzinfo=timezone.utc)
    stop_date = start_date + deltat
    table = dataset.to_table(
        filter=(
            ds.field('time') >= Timestamp(start_date)
        ) & (
            ds.field('time') <= Timestamp(stop_date)
        )
    )
    df = table.to_pandas()
    return np.vstack(df.afsAttitudeState).mean(axis=0)


def findtangent(t, pos, FOV, bracket=(1e5, 3e5)):
    res = minimize_scalar(funheight, args=(t, pos, FOV), bracket=bracket)
    return res

def targetheight(s,t,pos,FOV,height):
    return((funheight(s,t,pos,FOV)-height)**2)

def findheight(t, pos, FOV, height,  bracket=(1e5, 3e5)):
    res = minimize_scalar(targetheight, args=(t, pos, FOV, height), bounds= bracket)
    return res

def col_heights(ccditem, x, nheights=None, spline=False, splineTPpos=False):
    if nheights == None:
        nheights = ccditem['NROW']
    d = ccditem['EXPDate']
    ts = load.timescale()
    t = ts.from_datetime(d)
    ecipos = ccditem['afsGnssStateJ2000'][0:3]
    q = ccditem['afsAttitudeState']
    quat = R.from_quat(np.roll(q, -1))
    qprime = R.from_quat(ccditem['qprime'])
    ypixels = np.linspace(0, ccditem['NROW'], nheights)
    ths = np.zeros_like(ypixels)
    TPpos = np.zeros((len(ypixels), 3))
    xdeg, ydeg = pix_deg(ccditem, x, ypixels)
    for iy, y in enumerate(ydeg):
        los = R.from_euler('XYZ', [0, y, xdeg], degrees=True).apply([1, 0, 0])
        ecivec = quat.apply(qprime.apply(los))
        res = findtangent(t, ecipos, ecivec)
        TPpos[iy, :] = ecipos+res.x*ecivec
        ths[iy] = res.fun
    if spline:
        return CubicSpline(ypixels, ths)
    elif splineTPpos:
        return CubicSpline(ypixels, TPpos)
    else:
        return ths


def heights(ccditem):
    ths = np.zeros([ccditem['NROW'], ccditem['NCOL']+1])
    for col in range(ccditem['NCOL']+1):
        ths[:, col] = col_heights(ccditem, col)
    return ths

def fast_heights(ccditem, nx=5, ny=10):
    xpixels = np.linspace(0, ccditem['NCOL'], nx)
    ypixels = np.linspace(0, ccditem['NROW'], ny)
    ths_tmp = np.zeros([xpixels.shape[0], ypixels.shape[0]])
    for i,col in enumerate(xpixels): 
        ths_tmp[i,:]=col_heights(ccditem,col,ny*2,spline=True)(ypixels)
    interpolator=RegularGridInterpolator((xpixels,ypixels),ths_tmp, method = 'cubic')  
    fullxgrid=np.arange(ccditem['NCOL']+1)  
    fullygrid=np.arange(ccditem['NROW'])
    XX,YY=np.meshgrid(fullxgrid,fullygrid, sparse=True)
    return interpolator((XX,YY))



def satpos(ccditem):
    """Function giving the GPS position in lat lon alt..


    Arguments:
        ccditem or dataframe with the 'afsGnssStateJ2000'

    Returns:
        satlat: latitude of satellite (degrees)
        satlon: longitude of satellite (degrees)
        satheight: Altitude in metres

    """
    ecipos= ccditem['afsGnssStateJ2000'][0: 3]
    d = ccditem['EXPDate']
    ts= load.timescale()
    t = ts.from_datetime(d)
    satpo = Geocentric(position_au=Distance(
        m=ecipos).au, t=t)
    satlat, satlong, satheight = satpo.frame_latlon(itrs)
    return (satlat.degrees, satlong.degrees, satheight.m)

def TPpos(ccditem):
    """
    Function giving the GPS TP in lat lon alt..


    Arguments:
        ccditem or dataframe with the 'afsTangentPointECI'

    Returns:
        TPlat: latitude of satellite (degrees)
        TPlon: longitude of satellite (degrees)
        TPheight: Altitude in metres

    """
    eci= ccditem['afsTangentPointECI']
    d = ccditem['EXPDate']
    ts= load.timescale()
    t = ts.from_datetime(d)
    TPpos = Geocentric(position_au=Distance(
        m=eci).au, t=t)
    TPlat, TPlong, TPheight = TPpos.frame_latlon(itrs)
    return (TPlat.degrees, TPlong.degrees, TPheight.m)

def angles(ccditem):
    """
    Function giving various angles..


    Arguments:
        ccditem or dataframe with the 'EXPDate'

    Returns:
        nadir_sza: solar zenith angle at satelite position (degrees)
        TPsza: solar zenith angle at TP position (degrees)
        TPssa: solar scattering angle at TP position (degrees),
        tpLT: Local time at the TP (string)

    """
    planets= load('de421.bsp')
    earth, sun, moon = planets['earth'], planets['sun'], planets['moon']

    d = ccditem['EXPDate']
    ts= load.timescale()
    t = ts.from_datetime(d)
    satlat, satlon, satheight = satpos(ccditem)
    TPlat, TPlon, TPheight= TPpos(ccditem)
    sat_pos= earth + wgs84.latlon(satlat, satlon, elevation_m=satheight)
    sundir= sat_pos.at(t).observe(sun).apparent()
    obs= sundir.altaz()
    nadir_sza= (90-obs[0].degrees)  # nadir solar zenith angle
    TP_pos= earth + wgs84.latlon(TPlat, TPlon, elevation_m=TPheight)
    tpLT= ((d+DT.timedelta(seconds=TPlon/15*60*60)).strftime('%H:%M:%S'))  # 15*60*60 comes from degrees per hour

    FOV= (TP_pos-sat_pos).at(t).position.m
    FOV= FOV/norm(FOV)
    sundir= TP_pos.at(t).observe(sun).apparent()
    obs= sundir.altaz()
    TPsza = (90-obs[0].degrees)
    TPssa = (np.rad2deg(np.arccos(np.dot(FOV,sundir.position.m / norm(sundir.position.m)))))
    return nadir_sza, TPsza, TPssa, tpLT


def deg_map(ccditem):
    """
    Function to get the x and y angular deviation map for each pixel of the image. 
    The deviation is given in degrees relative to the center of the CCD
    
    
    Arguments
    ----------
    ccditem : CCDitem
        measurement
            
    Returns
    -------
    xmap : array[float]
        angular deviation map along the x axis in degrees (relative to the center of the CCD)
    ymap : array[float]
        angular deviation map along the y axis in degrees (relative to the center of the CCD) 
    """    
    im = ccditem['IMAGE']

    a,b = np.shape(im)
    X = range(b)
    Y = range(a)
    xpixel, ypixel = np.meshgrid(X,Y)
    xmap,ymap = pix_deg(ccditem, xpixel, ypixel)
    return xmap,ymap


def funheight_square(s, t, pos, FOV):
    """
    Function to get the distance between a point at position pos + s*FOV and the surface of the Geoid (wgs84 model),
     at time t.
    
    
    Arguments
    ----------
    s : float
        length along the straight line
    t : skyfield.timelib.Time
        time
    pos : array[float]
        position in space where the line starts (~position of MATS). Array of 3 position coordinates in m in the ICRF reference frame
    FOV : array[float]
        angle of the line (direction of the line), array of 3 elements in the IFRC reference frame
            
    Returns
    -------
    elevation**2 : float
        elevation squared of the point pos+s*FOV in m**2
    """
    newp = pos + s * FOV
    newp = ICRF(Distance(m=newp).au, t=t, center=399)
    return wgs84.subpoint(newp).elevation.m**2


def findsurface(t, pos, FOV):
    """
    Function to get the distance between a point at position pos and the surface of the Geoid (wgs84 model),
     at time t, along the line oriented along the FOV direction and starting at position pos
    
    
    Arguments
    ----------
    t : skyfield.timelib.Time
        time
    pos : array[float]
        position in space where the line starts (~position of MATS). Array of 3 position coordinates in m in the ICRF reference frame
    FOV : array[float]
        angle of the line (direction of the line), array of 3 elements in the IFRC reference frame
            
    Returns
    -------
    res : OptimizeResult object
        res.x is the distance found in m   
    """
    res = minimize_scalar(funheight_square, args=(t, pos, FOV), bracket=(3e5, 8e5))
    return res


def NADIR_geolocation(ccditem,x_sample=None,y_sample=None,interp_method='quintic'):
    """
    Function to get the latitude, longitude and solar zenith angle map for each pixel of the image.
    The values are calculated for some points and then interpolated for each pixel.
    WARNING : no images are flipped
    
    Arguments
    ----------
    ccditem : CCDitem
        measurement
    x_sample : int
        number of geolocated points along the x axis used for the interpolation. Default value is None, which means that there is no interpolation along the x-axis (each value is computed)
    y_step : int
        number of geolocated points along the y axis used for the interpolation. Default value is None, which means that there is no interpolation along the y-axis (each value is computed)
    interp_method :
        interpolation method : 'linear', 'nearest', 'slinear', 'cubic', 'quintic' and 'pchip'
        WARNING : choose the minimum x and y sampling according to the interpolation method
            
    Returns
    -------
    lat_map : array[float]
        map giving the latitude for each pixel in the image
    lon_map : array[float]
        map giving the longitude for each pixel in the image
    sza_map : array[float]
        map giving the solar zenith angle for each pixel in the image
    """
    im = ccditem['IMAGE']
    x_deg_map, y_deg_map = deg_map(ccditem) # creating angle deviation map for each pixel (degrees)
    a,b = np.shape(im)

    metoOHB  = R.from_matrix([[0,0,-1],[0,-1,0],[-1,0,0]])
    ts=sfapi.load.timescale()
    t=ts.from_datetime((ccditem['EXPDate']+timedelta(seconds=ccditem['TEXPMS']/(2*1000))).replace(tzinfo=sfapi.utc)) # exposure time (middle of the exposure timespan)  
    q=ccditem.afsAttitudeState
    quat=R.from_quat(np.roll(q,-1)) # quaternion of MATS attitude (for the satellite frame) 
    pos=ccditem.afsGnssStateJ2000[0:3] # position of MATS
    
    if x_sample == None or x_sample >= b: # no upsampling
        x_sample = b
    if y_sample == None or y_sample >= a: # no upsampling
        y_sample = a
    
    interpolation = True
    if x_sample == b and y_sample == a: # if both axis have enough sampling points, there is no interpolation
        interpolation = False

    xd = np.linspace(np.min(x_deg_map),np.max(x_deg_map),x_sample) # sampled angles on the x axis
    yd = np.linspace(np.min(y_deg_map),np.max(y_deg_map),y_sample) # sampled angles on the y axis
    x_deg_sample,y_deg_sample = np.meshgrid(xd,yd)

    if not interpolation:
        y_deg_sample,x_deg_sample = y_deg_map,x_deg_map # the sampled angles are the calculated angles for each pixel

    # sampled latitude, longitude and solar zenith angle values
    LAT = np.zeros((y_sample,x_sample))
    LON = np.zeros((y_sample,x_sample))
    SZA = np.zeros((y_sample,x_sample))

    # computing the latitude, longitude and solar zenith angles at the intersection of the line of sight and the earth surface
    # only the line of sights from some sampled pixels are computed
    for i in range(y_sample):
        for j in range(x_sample):
                # angular transformations
                # rotation from the line of sight of the LIMB imager to the line of sight of the NADIR pixel
                angle = R.from_euler('XYZ', [x_deg_sample[i,j],-(90-24)+y_deg_sample[i,j],0] , degrees=True).apply([1, 0, 0])
                FOV = quat.apply(metoOHB.apply(angle)) # attitude state for the line of sight of the NADIR pixel    
                # finding the distance between the point pos and the Geoid along the line of sight
                res = findsurface(t,pos,FOV)
                newp = pos + res.x * FOV 
                newp = ICRF(Distance(m=newp).au, t=t, center=399) # point at the intersection between the line of sight at the pixel and the Geoid surface
                LAT[i,j]=wgs84.subpoint(newp).latitude.degrees # latitude of the point
                LON[i,j]=wgs84.subpoint(newp).longitude.degrees # longitude of the point E [-180,+180] 

                # finding the solar zenith angle of the point
                planets = sfapi.load('de421.bsp')
                earth=planets['Earth']
                sun=planets['Sun']
                SZA[i,j]=90-((earth+wgs84.subpoint(newp)).at(t).observe(sun).apparent().altaz())[0].degrees
    
    # to get a continuous longitudinal field
    if np.max(LON)-np.min(LON) > 300: # this condition is met if points are on both sides of the -180/+180 deg line
        LON = np.where(LON<0,LON+360,LON)

    if interpolation: # interpolating the results along all the pixels
        # each interpolator object takes as argument an y and x angular deviation and gives a lat/lon/sza value
        interp_lat = RegularGridInterpolator((yd,xd),LAT,interp_method,bounds_error=False,fill_value=None) 
        interp_lon = RegularGridInterpolator((yd,xd),LON,interp_method,bounds_error=False,fill_value=None)
        interp_sza = RegularGridInterpolator((yd,xd),SZA,interp_method,bounds_error=False,fill_value=None)
        # interpolating on the real angular deviations for each pixel
        lat_map = interp_lat((y_deg_map,x_deg_map))
        lon_map = interp_lon((y_deg_map,x_deg_map))
        sza_map = interp_sza((y_deg_map,x_deg_map))
    else: # no interpolation       
        lat_map = LAT
        lon_map = LON
        sza_map = SZA

    return(lat_map,lon_map,sza_map)



def nadir_az(ccditem):
    """
    Function giving the solar azimuth angle for the nadir imager  
   
    Arguments:
        ccditem 
    Returns:
        nadir_az: float
            solar azimuth angle at nadir imager (degrees)       
        
    """
    planets=load('de421.bsp')
    earth,sun,moon= planets['earth'], planets['sun'],planets['moon']
   
     
    d = ccditem['EXPDate']
    ts =load.timescale()
    t = ts.from_datetime(d)
    satlat, satlon, satheight = satpos(ccditem)
    TPlat, TPlon, TPheight = TPpos(ccditem)
    
    sat_pos=earth + wgs84.latlon(satlat, satlon, elevation_m=satheight)
    TP_pos=earth + wgs84.latlon(TPlat, TPlon, elevation_m=TPheight)
    sundir=sat_pos.at(t).observe(sun).apparent()
    limbdir = TP_pos.at(t) - sat_pos.at(t)
    obs_limb = limbdir.altaz()
    obs_sun=sundir.altaz()
    nadir_az = (obs_sun[1].degrees - obs_limb[1].degrees) #nadir solar azimuth angle    
    return nadir_az
