import datetime as DT
import matplotlib.pylab as plt
from numpy.linalg import norm
import numpy as  np
import skyfield.api as sfapi
from skyfield.api import wgs84,utc
import skyfield.sgp4lib as sgp4lib
from scipy.optimize import minimize_scalar
from skyfield.positionlib import ICRF
from skyfield.units import Distance
from skyfield.framelib import itrs
import ephem
import os
import sqlite3 as sqlite

def get_position(date,TLE=None,database=False):
    """Function which extracts geolocation of satellite and tangent point
    based on TLE (only fixed TLE supported at the moment) and
    tangent point is calculated assuming 92km tanalt. 

    tle=['1 99988U 22123A   22327.21800926  .00000000  00000-0  22763-3 0   159',
    '2 99988  97.6525 329.4315 0012466 240.1585 268.5406 14.92722274002766']
 
    Arguments:
        date (obj:'datetime'): datetime object with date
        TLE: Not supported, ignore this. 

    Returns:
        satlat: latitude of satellite (degrees)
        satlon: longitude of satellite (degrees)
        satLT: local time of satellite (degrees)
        nadir_sza: solar zenith angle of nadir point (degrees)
        nadir_mza: solar zenith angle of moon (degrees)
        TPlat: tangent point latitude (degrees)
        TPlon: tangent point longitude (degrees)
        tpLT: local time at tangent point
        TPsza: solar zenith angle at tangent point (degrees)
        TPssa: solar scattering angle at tangent point (degrees)
    """
    #Only one TLE, only TP at 92 km

    planets=sfapi.load('de421.bsp')
    earth,sun= planets['earth'], planets['sun']
    moon = planets['moon']

    yaw = 0
    yawoffset = 0
    ts=sfapi.load.timescale()
    if TLE == None:
        sfodin=get_tle_MATS(date,database)
    else: 
        raise NotImplementedError('custom TLE not supported')

    t=ts.utc(date.year,date.month,date.day,date.hour,date.minute,date.second)
    g=sfodin.at(t)
    period= 2*np.pi/sfodin.model.nm
    ECI_pos=g.position.m
    ECI_vel=g.velocity.m_per_s
    vunit=np.array(ECI_vel)/norm(ECI_vel)
    mrunit=-np.array(ECI_pos)/norm(ECI_pos)
    yunit=np.cross(mrunit,vunit)
    rotmatrix=np.array([vunit,yunit,mrunit]).T 
    satlat=g.subpoint().latitude.degrees #nadir latitude
    satlon=g.subpoint().longitude.degrees #nadir longitude
    satLT = ((date+DT.timedelta(seconds=satlon/15*60*60)).strftime('%H:%M:%S')) #15*60*60 comes from degrees per hour
    pitch=findpitch(92000,t, ECI_pos, np.deg2rad(yaw)+yawoffset, rotmatrix)


    #yaw=-3.3*np.cos(np.deg2rad(tt*timestep.seconds/period/60*360-np.rad2deg(pitch)-0))
    #yaw =0
    #print(np.rad2deg(pitchdown))
    #pitch=findpitch(92000,t, ECI_pos, np.deg2rad(yaw)+yawoffset, rotmatrix)
    FOV=rotate(np.array([1,0,0]),np.deg2rad(yaw)+yawoffset,pitch,0,deg=False)
    FOV=np.matmul(rotmatrix,FOV)
    res = findtangent(t,ECI_pos,FOV)
    s=res.x
    newp = ECI_pos + s * FOV
#    pos_s=np.matmul(itrs.rotation_at(t),newp)
    newp=ICRF(Distance(m=newp).au,t=t,center=399)
    TPlat = (wgs84.subpoint(newp).latitude.degrees)
    TPlon = (wgs84.subpoint(newp).longitude.degrees)
    tpLT = ((date+DT.timedelta(seconds=TPlon/15*60*60)).strftime('%H:%M:%S')) #15*60*60 comes from degrees per hour
    sundir=(earth+wgs84.subpoint(newp)).at(t).observe(sun).apparent()
    obs=sundir.altaz()
    TPsza = (90-obs[0].degrees)
    TPssa = (np.rad2deg(np.arccos(np.dot(FOV,sundir.position.m/norm(sundir.position.m)))))

    sundir=(earth+wgs84.subpoint(g)).at(t).observe(sun).apparent()
    obs=sundir.altaz()
    nadir_sza = (90-obs[0].degrees) #nadir solar zenith angle
    moondir=(earth+wgs84.subpoint(g)).at(t).observe(moon).apparent()
    obs=moondir.altaz()
    nadir_mza = (90-obs[0].degrees) #nadir solar zenith angle

    return satlat,satlon,satLT,nadir_sza,nadir_mza,TPlat,TPlon,tpLT,TPsza,TPssa

def rotate (unitvec, yaw, pitch, roll, deg=False):
    def Rx (v,th):
        s=np.sin(th)
        c=np.cos(th)
        return np.matmul([[1,0,0],[0,c,-s],[0,s,c]],v)
    def Ry (v,th):
        s=np.sin(th)
        c=np.cos(th)
        return np.matmul([[c,0,s],[0,1,0],[-s,0,c]],v)
    def Rz (v,th):
        s=np.sin(th)
        c=np.cos(th)
        return np.matmul([[c,-s,0],[s,c,0],[0,0,1]],v)
    if deg :
        roll*=(np.pi/180)
        pitch*=(np.pi/180)
        yaw*=(np.pi/180)
    return Rz(Ry(Rx(unitvec,roll),pitch),yaw)

def xyz2radec(vector, deg=False, positivera=False):
    ra = np.arctan2(vector[1],vector[0])
    if positivera : 
        if ra <0 : ra+=2*np.pi
    dec = np.arcsin(vector[2,:]/norm(vector,axis=0))
    if deg :
        ra*=180./np.pi
        dec*=180./np.pi
    return [ra,dec]

def radec2xyz(ra,dec, deg=True):
    if deg:
        ra*=np.pi/180.
        dec*=np.pi/180.
    z=np.sin(dec)
    x=np.cos(ra)*np.cos(dec)
    y=np.sin(ra)*np.cos(dec)
    return [x,y,z]

        
def get_tle_MATS(d,database=False):
    ts=sfapi.load.timescale()
    if d.tzinfo == None:
        d = d.replace(tzinfo=DT.timezone.utc)

    if database:
        tle = get_tle_dateDB(d)
    else:
        print('Using Fixed TLEs will be dissalowed in near future, use database = True')
        if d<DT.datetime(2022,11,26,tzinfo=DT.timezone.utc):
            tle=['1 99988U 22123A   22327.21800926  .00000000  00000-0  22763-3 0   159',
                '2 99988  97.6525 329.4315 0012466 240.1585 268.5406 14.92722274002766']
        else:
            tle=['1 99988U 22123A   22332.47506944  .00000000  00000-0  24758-3 0   163',
                '2 99988  97.6531 334.5422 0012605 221.2886  80.4135 14.92738746003557']

    
    sfodin = sgp4lib.EarthSatellite(tle[0],tle[1])
    return sfodin

# def loadysb(d):
#     ysb=[]
#     with open('YBS.edb','r') as fb:
#         for line in fb:
#             if line[0] !='#' and len(line) >1 : 
#                 st=ephem.readdb(line)
#                 st.compute()
#                 ysb.append(st)
#     return ysb



def get_tle_dateDB(d,maxdays=3):

    d=d.replace(tzinfo=None)
    homecat=os.environ.get('HOME')
    tledb='data/matsTLE.db'
    db=sqlite.connect(tledb)
    #db=sqlite.connect('/tmp/matsTLE.db')
    cur=db.cursor()
    sperday=24.*60*60
    doy=d-DT.datetime(d.year,1,1)
    datekey =((d.year-int(d.year/100)*100)*1000 + doy.days+doy.seconds/sperday)*100
    #query="select tle1,tle2 from matsTletext where datekey between {} and {}"
    query="select datekey, Tle1,Tle2 FROM MatsTleText where abs(datekey-{}) < 100 * {}"

    #r=cur.execute(query.format(datekey-100*ndaysbefore,datekey+100*ndaysafter)) #four day margin
    r=cur.execute(query.format(datekey,maxdays)) #four day margin
    tle=r.fetchone()
    cur.close()
    db.close()

    return tle[1:]


def funpitch(pitch,t,th,pos,yaw,rotmatrix):
    FOV=rotate(np.array([1,0,0]),yaw,pitch,0,deg=False)
    FOV=np.matmul(rotmatrix,FOV)
    tp=findtangent(t,pos,FOV)
    return((tp.fun-th)**2)

def funheight (s,t,pos,FOV):
    newp = pos + s * FOV
    newp=ICRF(Distance(m=newp).au,t=t,center=399)
    return wgs84.subpoint(newp).elevation.m

def findtangent(t,pos,FOV):
    res=minimize_scalar(funheight,args=(t,pos,FOV),bracket=(1e5,3e5))
    return res

def findpitch (th,t,pos,yaw,rotmatrix):
    res=minimize_scalar(funpitch,args=(t,th,pos,yaw,rotmatrix),method="Bounded",bounds=(np.deg2rad(-30),np.deg2rad(-10)))
    return res.x

def angle2limbalt(theta,deg=False,satalt=591e3,z_limb=90e3):
    if deg:
        theta = np.deg2rad(theta)

    R_e = 6371e3
    z_limb = 90e3
    l_limb = np.sqrt((R_e+satalt)**2 - (R_e+z_limb)**2)
    z_alt = 2*np.tan(theta/2)*l_limb
    return z_alt

def limbalt2angle(z_alt,deg=False,satalt=591e3):
    R_e = 6371e3
    z_limb = 90e3
    l_limb = np.sqrt((R_e+satalt)**2 - (R_e+z_limb)**2)
    theta = 2*np.arctan(z_alt/2/l_limb)
    if deg:
        theta = np.rad2deg(theta)
    return theta
