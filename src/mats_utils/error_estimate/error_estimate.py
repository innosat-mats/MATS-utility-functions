from mats_l1_processing import L1_calibration_functions as l1cal

import numpy as np

def bin_abs_error(CCDitem, error_nonbinned):
    """
    This is a function to bin an error estimate. 
    Bins according to binning and NSKIP settings in CCDitem.

    Args:
        CCDitem:  dictonary containing CCD image and information
        erroe_nonbinned (optional): numpy array with error estimate for each pixel

    Returns:
        binned_error: binned error (currently assumes that the error is the same for all subpixels in a superpixel)

    """
    binned_error=np.sqrt(l1cal.meanbin_image_with_BC(CCDitem, error_nonbinned**2))
    
    return binned_error



def get_electrons(CCDitem,unit='lsb'):
    #takes L0 image and returns electrons (removes bias and corrects for non-linearity)
    image_lsb = CCDitem["IMAGE"]
    image_bias_sub,_ = l1cal.get_true_image(CCDitem,image_lsb)
    image_linear,_ = l1cal.get_linearized_image(CCDitem,image_bias_sub, force_table=True)
    image_electrons = image_linear*CCDitem["CCDunit"].alpha_avr(CCDitem["GAIN Mode"])/CCDitem["CCDunit"].ampcorrection

    return image_electrons

def get_shot_noise(CCDitem,unit='lsb'):
    #takes in CCDitem and calulates shot noise
    e_noise = np.sqrt(get_electrons(CCDitem)) #noise in get_electrons
    lsb_noise = e_noise/CCDitem["CCDunit"].alpha_avr(CCDitem["GAIN Mode"])*CCDitem["CCDunit"].ampcorrection
    return lsb_noise

def get_readout_noise(CCDitem,unit='lsb'):
    #takes in CCDitem and calulates readout noise
    e_noise = CCDitem["CCDunit"].ro_avr(CCDitem["GAIN Mode"])
    lsb_noise = e_noise/CCDitem["CCDunit"].alpha_avr(CCDitem["GAIN Mode"])*CCDitem["CCDunit"].ampcorrection

    return lsb_noise

def get_digitization_noise(CCDitem,unit='lsb'):
    #takes in CCDitem and calulates shot noise
    lsb_noise = 2**int(CCDitem['WDW InputDataWindow'][-1])

    return lsb_noise

def get_compression_noise(CCDitem,unit='lsb'):
    #takes in CCDitem and calulates shot noise
    rms_noise = np.array([6.2,6,5.3,4.5,3.3,2,0]) #from MATS CDR (can probably be remade)
    jpegq = np.array([70,75,80,85,90,95,100]) #from MATS CDR (can probably be remade)
    jpeg_noise = np.interp(CCDitem['JPEGQ'],jpegq,rms_noise)
    lsb_noise = 2**int(CCDitem['WDW InputDataWindow'][-1])*jpeg_noise

    return lsb_noise


def lin_error(image_linear,channel):

    counts_data = np.array([0,100,1000,2000,4000,6000,20000,40000,60000])
    relative_error_data = np.array([0.1,0.1,0.2,0.2,0.3,0.1,0.5,0.8,1])*0.01
    relative_error = np.interp(image_linear,counts_data,relative_error_data)
    lsb_error = relative_error*image_linear

    return lsb_error

def get_linearization_noise(CCDitem,unit='lsb'):
    #takes in CCDitem and calulates error from linearization
    image_lsb = CCDitem["IMAGE"]
    image_bias_sub,_ = l1cal.get_true_image(CCDitem,image_lsb)
    image_linear,_ = l1cal.get_linearized_image(CCDitem,image_bias_sub, force_table=True)
    lsb_noise = lin_error(image_linear,CCDitem["channel"])
    if unit == 'lsb':
        noise = lsb_noise
    elif unit == 'rad':
        image_linear_p = image_linear + lsb_noise
        image_linear_m = image_linear - lsb_noise

        image_desmeared, error_flags_desmear= l1cal.desmear_true_image(CCDitem, image_linear_p)
        image_dark_sub, error_flags_dark = l1cal.subtract_dark(CCDitem, image_desmeared)
        image_flatfielded, error_flags_flatfield= l1cal.flatfield_calibration(CCDitem, image_dark_sub)
        image_calibrated_p= l1cal.flip_image(CCDitem, image_flatfielded)

        image_desmeared, error_flags_desmear= l1cal.desmear_true_image(CCDitem, image_linear_m)
        image_dark_sub, error_flags_dark = l1cal.subtract_dark(CCDitem, image_desmeared)
        image_flatfielded, error_flags_flatfield= l1cal.flatfield_calibration(CCDitem, image_dark_sub)
        image_calibrated_m= l1cal.flip_image(CCDitem, image_flatfielded)

        rad_noise = (image_calibrated_p-image_calibrated_m)/2

    return 


def get_darkcurrent_error(CCDitem,unit = 'lsb'):
    """
    Takes a CCDitem and returns the error estimate for the darkvcurrent.

    CCDitem: CCDitem for which to add error
    """
    CCDunit=CCDitem['CCDunit']
    T=CCDitem["temperature"]

    if CCDitem["GAIN Mode"] == 'High':
        log_a_img_avr=CCDunit.log_a_img_avr_HSM
        log_b_img_avr=CCDunit.log_b_img_avr_HSM
        log_a_img_std=CCDunit.log_a_img_err_HSM
        log_b_img_std=CCDunit.log_b_img_err_HSM
    elif CCDitem["GAIN Mode"] == 'Low':
        log_a_img_avr=CCDunit.log_a_img_avr_LSM
        log_b_img_avr=CCDunit.log_b_img_avr_LSM 
        log_a_img_std=CCDunit.log_a_img_err_LSM
        log_b_img_std=CCDunit.log_b_img_err_LSM           
    else:
        raise Exception("Undefined mode")
    

    rawdark=CCDunit.getrawdark(log_a_img_avr, log_b_img_avr, T)

    #Add errors from log_a and log_b since they are correlated - this may be an overestimate - in fact they may be anticorrelated /LM 240513 
    errorab=CCDunit.getrawdark(log_a_img_avr+log_a_img_std, log_b_img_avr+log_b_img_std, T)-rawdark
    # Add error from temperature, assuming deltaT=3 degrees
    deltaT=3
    errorT=CCDunit.getrawdark(log_a_img_avr, log_b_img_avr, T+deltaT)-rawdark
    
    toterrorinelectronspers=np.sqrt(errorT**2+errorab**2) 

    dark_calc_err_image = (
        CCDunit.ampcorrection
        * toterrorinelectronspers
        / CCDunit.alpha_avr(CCDitem["GAIN Mode"])
    ) # in number of counts per second

    error_lsb = bin_abs_error(CCDitem, dark_calc_err_image)

    if unit == 'lsb':
        error = error_lsb
    elif unit == 'rad':    
        error_flatfielded, error_flags_flatfield= l1cal.flatfield_calibration(CCDitem, error_lsb)
        error= l1cal.flip_image(CCDitem, error_flatfielded) #10**12 photons/nm/m2/str/s

    return error 

def get_flatfield_error(CCDitem, calibration_data, unit='rad'):
    """
    Takes a CCDitem and returns the error estimate for the flatfielding step.

    CCDitem: CCDitem for which to add error
    """
    #calculate the error in 10**12 photons/nm/m2/str/s by multiplying the relative error with the calibrated signal
    image_lsb = CCDitem["IMAGE"]
    image_bias_sub,_ = l1cal.get_true_image(CCDitem,image_lsb)
    image_linear,_ = l1cal.get_linearized_image(CCDitem,image_bias_sub, force_table=True)
    image_desmeared, error_flags_desmear= l1cal.desmear_true_image(CCDitem, image_linear)
    image_dark_sub, error_flags_dark = l1cal.subtract_dark(CCDitem, image_desmeared)


    channel = CCDitem['channel']
    # The error measured as the standard deviation of three images divided by square root of 3
    flatfield_err = np.load(
                calibration_data["flatfield"]["flatfieldfolder"]
                + "flatfield_err_"
                + channel
                + "_HSM.npy")
    # The baffle scalefield, where 1 means no effect, ie the flatfield is the 
    # same as the one without baffle. 
    flatfield = np.load(
                calibration_data["flatfield"]["flatfieldfolder"]
                + "flatfield_"
                + channel
                + "_HSM.npy")
    flatfield_err_binned = l1cal.bin_image_with_BC(CCDitem, flatfield_err) #binning factor needs to be added for correct absolute calibration
    flatfield_binned,_=l1cal.calculate_flatfield(CCDitem)

       
    image_calib_nonflipped_p =  l1cal.absolute_calibration(CCDitem, image=image_dark_sub)/ (flatfield_binned+flatfield_err_binned)
    image_calib_nonflipped_n =  l1cal.absolute_calibration(CCDitem, image=image_dark_sub)/ (flatfield_binned-flatfield_err_binned)

    if unit == 'lsb':
        raise ValueError('LSB not supported for flatield')
    elif unit == 'rad':
        error_flatfielded = np.abs((image_calib_nonflipped_p-image_calib_nonflipped_n))/2
        error= l1cal.flip_image(CCDitem, error_flatfielded)

    return error 


def get_absolute_error(CCDitem, unit='rad'):
    """
    Takes a CCDitem and returns the error estimate for the absolute calibration step.

    CCDitem: CCDitem for which to add error
    """
    #calculate the error in 10**12 photons/nm/m2/str/s by multiplying the relative error with the calibrated signal
    rel_error = 0.1
    
    image_lsb = CCDitem["IMAGE"]
    image_bias_sub,_ = l1cal.get_true_image(CCDitem,image_lsb)
    image_linear,_ = l1cal.get_linearized_image(CCDitem,image_bias_sub, force_table=True)
    image_desmeared, _= l1cal.desmear_true_image(CCDitem, image_linear)
    image_dark_sub, _ = l1cal.subtract_dark(CCDitem, image_desmeared)
    image_flatfielded, _ = l1cal.flatfield_calibration(CCDitem, image_dark_sub)
    image_calibrated_m = l1cal.flip_image(CCDitem, image_flatfielded)

    if unit == 'lsb':
        raise ValueError('LSB not supported for flatield')
    elif unit == 'rad':   
        error = image_calibrated_m*rel_error

    return error 

def get_cross_channel_error(CCDitem, unit='rad'):
    """
    Takes a CCDitem and returns the error estimate for the cross channel error.

    CCDitem: CCDitem for which to add error
    """
    #calculate the error in 10**12 photons/nm/m2/str/s by multiplying the relative error with the calibrated signal
    rel_error = 0.02
    
    image_lsb = CCDitem["IMAGE"]
    image_bias_sub,_ = l1cal.get_true_image(CCDitem,image_lsb)
    image_linear,_ = l1cal.get_linearized_image(CCDitem,image_bias_sub, force_table=True)
    image_desmeared, _= l1cal.desmear_true_image(CCDitem, image_linear)
    image_dark_sub, _ = l1cal.subtract_dark(CCDitem, image_desmeared)
    image_flatfielded, _ = l1cal.flatfield_calibration(CCDitem, image_dark_sub)
    image_calibrated_m = l1cal.flip_image(CCDitem, image_flatfielded)

    if unit == 'lsb':
        raise ValueError('LSB not supported for flatield')
    elif unit == 'rad':   
        error = image_calibrated_m*rel_error

    return error 