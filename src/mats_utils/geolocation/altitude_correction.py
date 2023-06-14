from scipy.interpolate import CubicSpline
import numpy as np
from mats_utils.geolocation.coordinates import fast_heights


def rows_to_altitudes_on_image(image, ths, fixalt):
    """Restructures image so so that altitude of each row is constant (according to ths)

    Parameters
    ----------
    image : 2Darray
        CCD image
    ths : 2Darray
        tangent heights of the image prodused by for example heights in mats_utils.geolocation.coordinates
    
    Optional parameters:
    ----------
    fixalt: 1Darray
        Array specifying altitudes

    Returns
    -------
    image_fixalt : 2darray
        the original image interpolated to fixec altitude 
    fixalt : 1Darray
        Array specifying altitudes
        
    """


    image_fixalt=np.zeros([len(fixalt),image.shape[1]])
    for icol in range(image.shape[1]):
        splinefunc=CubicSpline(ths[:,icol],image[:,icol] )
        image_fixalt[:,icol]=splinefunc(fixalt)

    return image_fixalt, fixalt
    
    
def rows_to_altitudes(CCDitem, fixaltvec=[61000, 107000, 1000], imagefield='IMAGE'):
    """Funtion that flattend altitudes in a CCDitem

    Parameters
    ----------
    CCDitem: dataframe Series
        once CCDitem, ie a row in a CCDitems dataframe

    fixalt: 1Darray
        Array specifying altitudes    
        
    """
   
    #ths=heights(CCDitem) 
    ths=fast_heights(CCDitem,nx=10,ny=10)
    image_fixalt, _ =rows_to_altitudes_on_image(CCDitem[imagefield], ths, fixalt=fixaltvec)

    return image_fixalt
    
