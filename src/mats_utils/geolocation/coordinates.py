from skyfield.framelib import itrs
from skyfield.positionlib import Geocentric, ICRF
from skyfield.units import Distance
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from skyfield.api import wgs84 
from skyfield.api import load
from skyfield.units import Distance
from scipy.spatial.transform import Rotation as R
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
    #if start_date.tzinfo == None:
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

def findtangent(t, pos, FOV):
    res = minimize_scalar(funheight, args=(t, pos, FOV), bracket=(1e5, 3e5))
    return res


def col_heights(ccditem, x, nheights=None, spline=False):
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
    xdeg, ydeg = pix_deg(ccditem, x, ypixels)
    for iy, y in enumerate(ydeg):
        los = R.from_euler('XYZ', [0, y, xdeg], degrees=True).apply([1, 0, 0])
        ecivec = quat.apply(qprime.apply(los))
        ths[iy] = findtangent(t, ecipos, ecivec).fun
    if spline:
        return CubicSpline(ypixels, ths)
    else:
        return ths


def heights(ccditem):
    ths = np.zeros([ccditem['NROW'], ccditem['NCOL']+1])
    for col in range(ccditem['NCOL']+1):
        ths[ :,col] = col_heights(ccditem, col)
    return ths


def satpos(ccditem):
    """Function giving the GPS position in lat lon alt.. 

   
    Arguments:
        ccditem or dataframe with the 'afsGnssStateJ2000'

    Returns:
        satlat: latitude of satellite (degrees)
        satlon: longitude of satellite (degrees)
        satheight: Altitude in metres 
        
    """
    ecipos = ccditem['afsGnssStateJ2000'][0:3]
    d = ccditem['EXPDate']
    ts =load.timescale()
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
    eci=ccditem['afsTangentPointECI']
    d = ccditem['EXPDate']
    ts =load.timescale()
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
    planets=load('de421.bsp')
    earth,sun,moon= planets['earth'], planets['sun'],planets['moon']
   
    d = ccditem['EXPDate']
    ts =load.timescale()
    t = ts.from_datetime(d)
    satlat, satlon, satheight = satpos(ccditem)
    TPlat, TPlon, TPheight = TPpos(ccditem) 
    sat_pos=earth + wgs84.latlon(satlat, satlon, elevation_m=satheight)
    sundir=sat_pos.at(t).observe(sun).apparent()
    obs=sundir.altaz()
    nadir_sza = (90-obs[0].degrees) #nadir solar zenith angle
    TP_pos=earth + wgs84.latlon(TPlat, TPlon, elevation_m=TPheight)
    tpLT = ((d+DT.timedelta(seconds=TPlon/15*60*60)).strftime('%H:%M:%S')) #15*60*60 comes from degrees per hour

    FOV=(TP_pos-sat_pos).at(t).position.m
    FOV=FOV/norm(FOV)
    sundir=TP_pos.at(t).observe(sun).apparent()
    obs=sundir.altaz()
    TPsza = (90-obs[0].degrees)
    TPssa = (np.rad2deg(np.arccos(np.dot(FOV,sundir.position.m/norm(sundir.position.m)))))
    return nadir_sza, TPsza, TPssa, tpLT


def pix_deg2(ccditem, xpixel, ypixel):
    """
    Function to get the x and y angle from a pixel relative to the center of the CCD
    WARNING : no images are flipped in this function
    
    Parameters
    ----------
    ccditem : CCDitem
        measurement
    xpixel : int or array[int]
        x coordinate of the pixel(s) in the image
    ypixel : int or array[int]
        y coordinate of the pixel(s) in the image
        
    Returns
    -------
    xdeg : float or array[float]
        angular deviation along the x axis in degrees (relative to the center of the CCD)
    ydeg : float or array[float]
        angular deviation along the y axis in degrees (relative to the center of the CCD) 
    """
    h = 6.9 # height of the CCD in mm
    d = 27.6 # width of the CCD in mm
    # selecting effective focal length
    if (ccditem['CCDSEL']) == 7: # NADIR channel
        f = 50.6 # effective focal length in mm
    else: # LIMB channels
        f = 261    
    
    ncskip = ccditem['NCSKIP']
    try:
        ncbin = ccditem['NCBIN CCDColumns']
    except:
        ncbin = ccditem['NCBINCCDColumns']
    nrskip = ccditem['NRSKIP']
    nrbin = ccditem['NRBIN']
        
    yCCDpix = (nrskip + nrbin * (ypixel+0.5)) # y position of the pixel on the CCD (0.5 at the bottom, 510.5 on top)
    xCCDpix = (ncskip + ncbin * (xpixel+0.5)) # x position of the pixel on the CCD (0.5 on the left, 2047.5 on the right)
    
    xdeg = (180/pi)*np.arctan(d*(xCCDpix/2048-0.5)/f) # angular deviation along the x axis in degrees
    ydeg = (180/pi)*np.arctan(h*(yCCDpix/511-0.5)/f) # angular deviation along the y axis in degrees
    return xdeg, ydeg

def deg_map(ccditem):
    """
    Function to get the x and y angular deviation map for each pixel of the image. 
    The deviation is given in degrees relative to the center of the CCD
    WARNING : no images are flipped before calculating the angular deviation
    
    Parameters
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
    xmap,ymap = pix_deg2(ccditem, xpixel, ypixel)
    return xmap,ymap


def funheight_square(s, t, pos, FOV):
    """
    Function to get the distance between a point at position pos + s*FOV and the surface of the Geoid (wgs84 model),
     at time t.
    
    
    Parameters
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
    
    
    Parameters
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


def NADIR_geolocation(ccditem,x_step=2,y_step=2):
    """
    Function to get the latitude, longitude and solar zenith angle map for each pixel of the image.
    The values are calculated for some points and then interpolated for each pixel.
    WARNING : no images are flipped
    
    Parameters
    ----------
    ccditem : CCDitem
        measurement
    x_step : int
        step along the x-axis in the image between 2 sampled points used for interpolation. The default value is 2.
    y_step : int
        step along the y-axis in the image between 2 sampled points used for interpolation. The default value is 2.
            
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
    x_deg_map, y_deg_map = deg_map(ccditem) # creating angle deviation map for each pixel (degress)
    a,b = np.shape(im)

    metoOHB  = R.from_matrix([[0,0,-1],[0,-1,0],[-1,0,0]])
    ts=sfapi.load.timescale()
    t=ts.from_datetime(ccditem['EXPDate'].replace(tzinfo=sfapi.utc)) # exposure time  
    q=ccditem.afsAttitudeState
    quat=R.from_quat(np.roll(q,-1)) # quaternion of MATS attitude (for the LIMB imager) 
    pos=ccditem.afsGnssStateJ2000[0:3] # position of MATS
    
    xd = range(0,b,x_step) # sampled pixels on the x axis
    yd = range(0,a,y_step) # sampled pixels on the y axis
    LAT = np.zeros((len(yd),len(xd)))
    LON = np.zeros((len(yd),len(xd)))
    SZA = np.zeros((len(yd),len(xd)))

    # computing the latitude, longitude and solar zenith angles at the intersection of the line of sight and the earth surface
    # only the line of sights from some sampled pixels are computed
    for i in range(len(yd)):
        for j in range(len(xd)):
                x = xd[j]
                y = yd[i]
                # angular transformations
                # rotation from the line of sight of the LIMB imager to the line of sight of the NADIR pixel
                angle = R.from_euler('XYZ', [x_deg_map[y,x],-(90-23)+y_deg_map[y,x],0] , degrees=True).apply([1, 0, 0])
                FOV = quat.apply(metoOHB.apply(angle)) # attitude state for the line of sight of the NADIR pixel    
                # finding the distance between the point pos and the Geoid along the line of sight
                res = findsurface(t,pos,FOV)
                newp = pos + res.x * FOV 
                newp = ICRF(Distance(m=newp).au, t=t, center=399) # point at the intersection between the line of sight at the pixel and the Geoid surface
                LAT[i,j]=wgs84.subpoint(newp).latitude.degrees # latitude of the point
                LON[i,j]=wgs84.subpoint(newp).longitude.degrees # longitude of the point    

                # finding the solar zenith angle of the point
                planets = sfapi.load('de421.bsp')
                earth=planets['Earth']
                sun=planets['Sun']
                SZA[i,j]=90-((earth+wgs84.subpoint(newp)).at(t).observe(sun).apparent().altaz())[0].degrees
    
    # interpolating the results along all the pixels
    interp_lat = RegularGridInterpolator((yd,xd),LAT,method="quintic",bounds_error=False,fill_value=None) 
    interp_lon = RegularGridInterpolator((yd,xd),LON,method="quintic",bounds_error=False,fill_value=None)
    interp_sza = RegularGridInterpolator((yd,xd),SZA,method="quintic",bounds_error=False,fill_value=None)

    X_map,Y_map = np.meshgrid(range(b),range(a))
    lat_map = interp_lat((Y_map,X_map))
    lon_map = interp_lon((Y_map,X_map))
    sza_map = interp_sza((Y_map,X_map))

    return(lat_map,lon_map,sza_map)