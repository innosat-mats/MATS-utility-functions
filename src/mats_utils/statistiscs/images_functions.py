#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 2 9:38:36 2022

@author: lindamegner

File for modulating the images for statistical analyis.
"""


import numpy as np

def create_imagecube(dfCCDitems, image_specification='IMAGE'):
    """    
    Parameters
    ----------
    dfCCDitems : dataframe containing of CCDitems as rows
    image_specification : STR IMAGE or image_calibrated supported 

    Returns
    -------
    3d ndarray with all images and time as the last dimension

    """
    if image_specification!='IMAGE' and image_specification!='image_calibrated':
        Warning('image_specification must be "IMAGE" or "image-calibrated"')
    imagelist=[]
    for index, CCDitem in dfCCDitems.iterrows():
        image=CCDitem[image_specification]
        imagelist.append(image)

    imagecube=np.array(imagelist)

    return imagecube

   