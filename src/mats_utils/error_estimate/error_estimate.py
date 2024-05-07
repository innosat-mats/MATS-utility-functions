from mats_l1_processing import L1_calibration_functions as l1c
import numpy as np

def get_electrons(CCDitem):
    #takes L0 image and returns electrons (removes bias and corrects for non-linearity)
    image_lsb = CCDitem["IMAGE"]
    image_bias_sub,_ = l1c.get_true_image(CCDitem,image_lsb)
    image_linear,_ = l1c.get_linearized_image(CCDitem,image_bias_sub, force_table=True)
    image_electrons = image_linear*CCDitem["CCDunit"].alpha_avr(CCDitem["GAIN Mode"])/CCDitem["CCDunit"].ampcorrection

    return image_electrons

def get_shot_noise(CCDitem):
    #takes in CCDitem and calulates shot noise
    e_noise = np.sqrt(get_electrons(CCDitem)) #noise in get_electrons
    lsb_noise = e_noise/CCDitem["CCDunit"].alpha_avr(CCDitem["GAIN Mode"])*CCDitem["CCDunit"].ampcorrection
    return lsb_noise

def get_readout_noise(CCDitem):
    #takes in CCDitem and calulates shot noise
    e_noise = CCDitem["CCDunit"].ro_avr(CCDitem["GAIN Mode"])
    lsb_noise = e_noise/CCDitem["CCDunit"].alpha_avr(CCDitem["GAIN Mode"])*CCDitem["CCDunit"].ampcorrection

    return lsb_noise

def get_digitization_noise(CCDitem):
    #takes in CCDitem and calulates shot noise
    lsb_noise = 2**int(CCDitem['WDW InputDataWindow'][-1])
    return lsb_noise

def get_compression_noise(CCDitem):
    #takes in CCDitem and calulates shot noise
    rms_noise = np.array([6.2,6,5.3,4.5,3.3,2,0]) #from MATS CDR (can probably be remade)
    jpegq = np.array([70,75,80,85,90,95,100]) #from MATS CDR (can probably be remade)
    jpeg_noise = np.interp(CCDitem['JPEGQ'],jpegq,rms_noise)
    lsb_noise = 2**int(CCDitem['WDW InputDataWindow'][-1])*jpeg_noise
    return lsb_noise