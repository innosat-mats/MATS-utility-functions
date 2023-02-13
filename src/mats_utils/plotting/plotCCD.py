import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cartopy.crs as ccrs
import pandas as pd
from mats_utils.geolocation import satellite as satellite
from cartopy.feature.nightshade import Nightshade
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mats_utils.geolocation import coordinates

flipped_CCDs = ['IR1', 'IR3', 'UV1', 'UV2']
image_var = {'l1a': 'IMAGE', 'l1b': 'ImageCalibrated'}
channel_var = {'1': 'IR1', '2': 'IR4', '3': 'IR3',
               '4': 'IR2', '5': 'UV1', '6': 'UV2',
               '7': 'NADIR'}


def check_type(CCD_dataframe):
    """Check format of CCD_dataframe
    Exit program if type is not DataFrame or Series (single row of dataframe)

    Parameters
    ----------
    CCD_dataframe : any
        CCD_dataframe

    Returns:
        (type): Datatype of dataframe
    """

    if isinstance(CCD_dataframe, (pd.core.frame.DataFrame,
                                  pd.core.series.Series)) is False:
        sys.exit("CCD_dataframe need to be converted to DataFrame!")

    return type(CCD_dataframe)


def check_level(CCD_dataframe):
    """Checks level of data to name variables accordingly

    Parameters
    ----------
    CCD_dataframe : any
        CCD_dataframe

    Returns:
    ----------
    image_str: str
        image variable name

    """

    if 'IMAGE' in CCD_dataframe.keys():
        lvl_str = 'l1a'
    if 'ImageCalibrated' in CCD_dataframe.keys():
        lvl_str = 'l1b'

    return lvl_str


def calculate_geo(CCD):
    """calculates orbital parameters

    Parameters
    ----------
    CCD : CCDitem
        measurement

    Returns
    -------
    _type_
        positions
    """

    return satellite.get_position(CCD['EXPDate'])


def save_figure(outpath, CCD, format, date_name=False):
    """Saves figure to outpath

    Parameters
    ----------
    outpath : str
        save path
    CCD : CCDitem
        measurement
    format : str
        format of saved figure
    """

    # filename
    if date_name:
        outname = CCD['EXPDate'].strftime("%Y%m%dT%H%M%S%f")

    else:
        outname = f"{CCD['ImageName'][:-4]}"

    plt.tight_layout()
    plt.savefig(f'{outpath}/{outname}.{format}', format=format)


def calculate_range(image, ranges, nstd, custom_cbar):
    """Calculates ranges, means and std

    Parameters
    ----------
    image : array
        image to compute from
    ranges : array
        if requested min max range
    nstd : int
        number of std dev
    custom_cbar : bool
        if false use nstd for ranges

    Returns
    -------
    vmin : float
        min point in range
    vmax : float
        max point in range
    mean : float
        mean of image
    std : float
        std dev of image
    """

    # calc std and mean
    std = image.std()
    mean = image.mean()

    if custom_cbar:
        vmin = ranges[0]
        vmax = ranges[1]
    else:
        vmax = mean+nstd*std
        vmin = mean-nstd*std

    return vmin, vmax, mean, std

def make_ths(CCD):
    xpixels = np.linspace(0, CCD['NCOL'], 5)
    ypixels = np.linspace(0, CCD['NROW'], 10)

    ths = np.zeros([xpixels.shape[0], ypixels.shape[0]])
    #print (ths.shape)
    for i,col in enumerate(xpixels): 
        ths[i,:]=coordinates.col_heights(CCD,col,40,spline=True)(ypixels)
    return xpixels,ypixels,ths.T


def generate_map(CCD, fig, ax, satlat, satlon, TPlat, TPlon):
    """Generates a map

    Parameters
    ----------
    CCD : CCDitem
        measurement
    fig : fig
        figure to plot in
    ax : ax
        axis to plot on
    satlat : float
        sat lat pos
    satlon : float
        sat lon pos
    TPlat : float
        tangent point lat
    TPlon : float
        tangent point lon

    Returns
    -------
    _type_
        _description_
    """

    # map settings
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.9, color='black',
                      alpha=0.5, linestyle='-')
    gl.xlabels_top = True
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    ax.set_xlabel('longitude [deg]')
    ax.set_ylabel('latitude [deg]')
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(Nightshade(CCD['EXPDate'], alpha=0.2))
    ax.coastlines()

    # plot sat position and tangent point
    ax.scatter(satlon, satlat, s=10,
               color='red', label='satellite pos.')
    ax.scatter(TPlon, TPlat, s=10,
               color='green', label='TP pos.')
    ax.legend(ncol=2, fontsize=7, loc='lower right')

    return fig, ax


def generate_histogram(ax, image, ranges, nstd, custom_cbar):
    """Generates histogram based on image

    Parameters
    ----------
    ax : axis
        axis to plot on
    image : _type_
        image from which histogram will be generated
    ranges : _type_
        ranges for plot
    nstd : _type_
        number of standard deviations
    custom_cbar : bool
        if custom ranges

    Returns
    -------
    ax : axes
        axis with histogram
    """
    # calculate means
    vmin, vmax, mean, std = calculate_range(image, ranges, nstd, custom_cbar)

    nbins = int(1 + np.ceil(np.log2(len(image.flatten()))))
    ax.hist(image.flatten(), bins=nbins, alpha=0.6,
            density=True, range=[mean-nstd*std, mean+nstd*std])
    ax.set_xlabel('counts')
    ax.axvline(x=mean, label='mean',
               linestyle='--', linewidth=1.5)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.legend(loc='upper right')
    ax.grid()

    return ax


def plot_image(CCD, ax=None, fig=None, outpath=None,
               nstd=2, cmap='inferno', custom_cbar=False,
               ranges=[0, 1000], format='png', save=True,
               fontsize=10):
    """
    Function to plot single MATS image

    Parameters
    ----------
    CCD : pandas.Series
        row of pandas dataframe of the image to be plotted
    fig : fig, optional
        figure handle to plot on
    outdir : str, optional
        Out directory, default = None
    nstd : int, optional
        number of standard deviations, by default 2
    cmap : str, optional
       colormap for plot, by default 'inferno'
    custom_cbar : bool, optional
        if custom cbar set True, by default False
    ranges : list, optional
        limits for custom cbar, by default [0,1000]
    format : str, optional
        format for files, by default 'png'

    """

    if (fig is None) & (ax is None):
        fig = plt.figure(figsize=(12, 3))
        ax = plt.axes()

    # check level of data
    lvl = check_level(CCD)

    # save parameters for plot
    image = CCD[image_var[lvl]]
    if lvl == 'l1b':
        image = np.stack(image)

    # geolocation stuff
    (satlat, satlon, satLT,
     nadir_sza, nadir_mza,
     TPlat, TPlon,
     TPLT, TPsza, TPssa) = calculate_geo(CCD)

    texpms = CCD['TEXPMS']
    exp_date = CCD['EXPDate'].strftime("%Y-%m-%dT%H:%M:%S:%f")
    channel = channel_var[str(CCD['CCDSEL'])]

    # calculate ranges
    vmin, vmax, mean, std = calculate_range(image, ranges, nstd, custom_cbar)

    # plot CCD image
    if (channel in flipped_CCDs) and (lvl == 'l1a'):
        nrows = np.arange(0, CCD['NROW'])
        ncols = np.arange(0, CCD['NCOL']+1)
        img = ax.pcolormesh(np.flip(ncols), nrows,
                            image, cmap=cmap,
                            vmax=vmax, vmin=vmin)

    else:
        img = ax.pcolormesh(image, cmap=cmap,
                            vmax=vmax, vmin=vmin)

    # add heights
    if lvl == 'l1b' and (CCD['CCDSEL'] != 7):
        CS = ax.contour(*make_ths(CCD), [50000,
                        60000, 70000, 80000, 90000,
                        100000, 110000,200000,250000,300000],
                        colors='w', alpha=0.2)
        ax.clabel(CS, inline=True)

    # add title
    ax.set_title(f'ch: {channel}; time: '
                 + f'{exp_date}; TEXPMS: {texpms}', fontsize=fontsize)

    if save:
        # print out additional information
        plt.figtext(0.1, 0.8, f'tpSZA: {TPsza:.6}',
                    fontsize=10, color='white')
        plt.figtext(0.5, 0.8, (f'satlat, satlon: ({satlat:.6}' +
                               f', {satlon:.6})'),
                    fontsize=10, color='white')
        plt.figtext(0.25, 0.8, f'TPlat, TPlon: ({TPlat:.6}, {TPlon:.6})',
                    fontsize=10, color='white')
        # save figure
        plt.tight_layout()
        save_figure(outpath, CCD, format)

        return
    else:
        return fig, ax, img


def simple_plot(CCD_dataframe, outdir, nstd=2, cmap='magma', custom_cbar=False,
                ranges=[0, 1000], format='png'):
    """Generates plots from CCD_dataframe with basic orbit parameters included.
    Images will be sorted in folders based on CCDSEL in directory specified.

    Parameters
    ----------
    CCD_dataframe : DataFrame
        CCD_dataframe for plotting
    outdir : str
        Out directory
    nstd : int, optional
        number of standard deviations, by default 2
    cmap : str, optional
       colormap for plot, by default 'inferno'
    custom_cbar : bool, optional
        if custom cbar set True, by default False
    ranges : list, optional
        limits for custom cbar, by default [0,1000]
    format : str, optional
        format for files, by default 'png'
    """

    dftype = check_type(CCD_dataframe)

    fig = plt.figure(figsize=(12, 3))
    ax = plt.axes()

    for CCDno in range(0, 8):
        if dftype == pd.core.frame.DataFrame:
            CCDs = CCD_dataframe[CCD_dataframe['CCDSEL'] == CCDno]
        elif dftype == pd.core.series.Series:
            if CCD_dataframe['CCDSEL'] == CCDno:
                CCDs = CCD_dataframe
            else:
                CCDs = []
        else:
            raise TypeError('Invalid dataframe')

        if len(CCDs) > 0:
            outpath = f"{outdir}CCDSEL{str(CCDno)}"
            if not os.path.exists(outpath):
                os.makedirs(outpath)
        else:
            continue

        if dftype == pd.core.series.Series:
            plot_image(CCDs, fig=fig, ax=ax, outpath=outpath,
                       nstd=nstd, cmap=cmap, custom_cbar=custom_cbar,
                       ranges=ranges, format=format)
        else:
            for index, CCD in CCDs.iterrows():
                plot_image(CCD, ax, fig=fig, outpath=outpath,
                           nstd=nstd, cmap=cmap, custom_cbar=custom_cbar,
                           ranges=ranges, format=format)


def orbit_plot(CCD_dataframe, outdir, nstd=2, cmap='magma', custom_cbar=False,
               ranges=[0, 1000], format='png'):
    """
       Generates plots from (several) CCD items: image, histogram and map.
       Figures will be saved in subfolders of outdir by CCDSEL.

    Parameters
    ----------
    CCD_dataframe : DataFrame
        CCD_dataframe to be plotted.
    outdir : str, optional
        path where images will be saved
    nstd : int, optional
        Number of standard deviations for cbar and histogram, by default 2
    cmap : str, optional
        Colourmap for image, by default 'inferno'
    custom_cbar : bool, optional
        Custom cbar, by default False
    ranges : tuple, optional
        If custom_cbar == True, specify cbar limits, by default (0,1000)
    format : str
        file format for output img
    """

    check_type(CCD_dataframe)

    for CCDno in range(0, 8):

        CCDs = CCD_dataframe[CCD_dataframe['CCDSEL'] == CCDno]

        if len(CCDs) > 0:
            if outdir is not None:
                outpath = f"{outdir}CCDSEL{str(CCDno)}"

                if not os.path.exists(outpath):
                    os.makedirs(outpath)

            for index, CCD in CCDs.iterrows():

                # check level of data
                lvl = check_level(CCD)

                # save parameters for plot
                image = CCD[image_var[lvl]]
                if lvl == 'l1b':
                    image = np.stack(image)

                # geolocation stuff
                (satlat, satlon, satLT,
                 nadir_sza, nadir_mza,
                 TPlat, TPlon,
                 TPLT, TPsza, TPssa) = calculate_geo(CCD)

                # generate figure and grid
                fig = plt.figure(figsize=(10, 7))

                ax = plt.subplot2grid((2, 2), (1, 0),
                                      colspan=1, rowspan=1,
                                      projection=ccrs.PlateCarree(),
                                      fig=fig)
                ax1 = plt.subplot2grid((2, 2), (0, 0),
                                       rowspan=1, colspan=2, fig=fig)
                ax2 = plt.subplot2grid((2, 2), (1, 1), rowspan=1,
                                       colspan=1, fig=fig)

                # plot map in figure
                ax = generate_map(CCD, fig, ax, satlat,
                                  satlon, TPlat, TPlon)

                # plot CCD image
                fig, ax1, img = plot_image(CCD, ax1, fig, outpath=outdir,
                                           nstd=nstd, cmap=cmap,
                                           custom_cbar=custom_cbar,
                                           ranges=ranges, format=format,
                                           save=False)

                fig.colorbar(img, ax=ax1)

                # print out additional information
                plt.figtext(0.15, 0.03, f'nadirSZA: {nadir_sza:.6}',
                            fontsize=10)
                plt.figtext(0.15, 0.06, f'nadirMZA: {nadir_mza:.6}',
                            fontsize=10)
                plt.figtext(0.35, 0.03, f'tpSZA: {TPsza:.6}', fontsize=10)
                plt.figtext(0.35, 0.06, f'tpSSA: {TPssa:.6}', fontsize=10)

                # plot histogram
                generate_histogram(ax2, image, ranges,
                                   nstd, custom_cbar)

                save_figure(outpath, CCD, format)
                fig.clear()
                plt.close()

    return


def all_channels_plot(CCD_dataframe, outdir, nstd=2, cmap='magma',
                      custom_cbar=False,
                      ranges=[0, 1000], format='png'):

    check_type(CCD_dataframe)

    fig, ax = plt.subplots(3, 3, figsize=(16, 9))
    fig.patch.set_facecolor('lightgrey')
    ax=ax.ravel()

    for i in range(0,len(ax)):
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])

    ax[8].remove()
    ax[7].remove()
    ax_cart = fig.add_subplot(3, 3, 8, projection=ccrs.PlateCarree())
    #ax_cart.xlabel('test')

    if outdir is not None:
        outpath = f"{outdir}ALL"

        if not os.path.exists(outpath):
            os.makedirs(outpath)

    for index, CCD in CCD_dataframe.iterrows():

        (satlat, satlon, satLT,
        nadir_sza, nadir_mza,
        TPlat, TPlon,
        TPLT, TPsza, TPssa) = calculate_geo(CCD)

        ax[CCD['CCDSEL'] - 1].clear()
        fig, ax[CCD['CCDSEL'] - 1], img= plot_image(CCD, ax[CCD['CCDSEL'] - 1], fig, outdir,
                nstd, cmap, custom_cbar,
                ranges, format, save=False, fontsize=10)

        if CCD['CCDSEL'] == 1:
            ax_cart.remove()
            ax_cart = fig.add_subplot(3, 3, 9, projection=ccrs.PlateCarree())
            generate_map(CCD, fig, ax_cart, satlat, satlon, TPlat, TPlon)

        save_figure(outpath, CCD, format,date_name=True)

    return