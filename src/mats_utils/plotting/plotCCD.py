import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cartopy.crs as ccrs
import pandas as pd
from mats_utils.geolocation import coordinates as coordinates
import cartopy
from cartopy.feature.nightshade import Nightshade
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mats_utils.geolocation import coordinates
from matplotlib.artist import Artist

flipped_CCDs = ['IR1', 'IR3', 'UV1', 'UV2', 'NADIR']
image_var = {'L1a': 'IMAGE', 'L1b': 'ImageCalibrated'}
channel_var = {'1': 'IR1', '2': 'IR4', '3': 'IR3',
               '4': 'IR2', '5': 'UV1', '6': 'UV2',
               '7': 'NADIR'}

# optimal ranges only for L1b so far
range_UV1 = [0, 30]
range_UV2 = [0, 30]
range_NADIR = [0, 75]
ranges_dayglow = {'IR1': [0, 30], 'IR2': [0, 30],
                  'IR3': [0, 30], 'IR4': [0, 30],
                  'UV1': range_UV1, 'UV2': range_UV2,
                  'NADIR': range_NADIR}
ranges_nightglow = {'IR1': [0, 5], 'IR2': [0, 5],
                    'IR3': [0, 5], 'IR4': [0, 5],
                    'UV1': range_UV1, 'UV2': range_UV2,
                    'NADIR': range_NADIR}
rswitch_sza = 97.5 # TPsza dayglow/nightglow change

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
        lvl_str = 'L1a'
    if 'ImageCalibrated' in CCD_dataframe.keys():
        lvl_str = 'L1b'

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

    satlat, satlon, satheight = coordinates.satpos(CCD)
    TPlat,TPlon,TPheight = coordinates.TPpos(CCD)
    nadir_sza, TPsza, TPssa, TPlt = coordinates.angles(CCD)

    return (satlat, satlon,
            nadir_sza,
            TPlat, TPlon,
            TPsza, TPssa,TPlt)

def save_figure(outpath, CCD, format, filename=None):
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
    if filename != None:
        outname = filename

    else:
        outname = f"{CCD['ImageName'][:-4]}"

    plt.tight_layout()
    plt.savefig(f'{outpath}/{outname}.{format}', format=format)


def calculate_range(image, ranges, nstd):
    """Calculates ranges, means and std

    Parameters
    ----------
    image : array
        image to compute from
    ranges : array
        if requested min max range
    nstd : int
        number of std dev

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

    if ranges is not None:
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

def update_plot_cbar(CCD, ax, fig, cbar,
                     outdir, nstd, cmap,
                     ranges, optimal_range, format,
                     save=False, fontsize=10):
    ax.clear()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig, ax, img = plot_image(CCD, ax, fig, outdir,
                              nstd, cmap,
                              ranges, optimal_range, format,
                              save, fontsize)
    cbar.update_normal(img)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(color='w')
    cbar.ax.tick_params(labelcolor='w')

    return

def generate_map(CCD, fig, ax, satlat, satlon, TPlat, TPlon,
                 mark_size=10, legend_fsize=7, labels=True,):
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
    gl.xlabels_top = labels
    gl.xlabels_bottom = labels
    gl.ylabels_left = labels
    gl.ylabels_right = False
    gl.xlines = True
    ax.set_xlabel('longitude [deg]')
    ax.set_ylabel('latitude [deg]')
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(Nightshade(CCD['EXPDate'], alpha=0.2))
    ax.add_feature(cartopy.feature.OCEAN)
    ax.coastlines()

    # plot sat position and tangent point
    ax.scatter(satlon, satlat, s=mark_size,
               color='red', label='satellite pos.')
    ax.scatter(TPlon, TPlat, s=mark_size,
               color='green', label='TP pos.')
    ax.legend(ncol=2, fontsize=legend_fsize, loc='lower right')

    return fig, ax


def generate_histogram(ax, image, ranges, nstd):
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

    Returns
    -------
    ax : axes
        axis with histogram
    """
    # calculate means
    vmin, vmax, mean, std = calculate_range(image, ranges, nstd)

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
               nstd=2, cmap='inferno', ranges=None,
               optimal_range=False, format='png', save=True,
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
    ranges : list, optional
        limits for custom cbar, overrides nstd
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
    if lvl == 'L1b':
        image = np.stack(image)

    # geolocation stuff
    (satlat, satlon,
     nadir_sza,
     TPlat, TPlon,
     TPsza, TPssa, TPlt) = calculate_geo(CCD)

    texpms = CCD['TEXPMS']
    exp_date = CCD['EXPDate'].strftime("%Y-%m-%dT%H:%M:%S:%f")
    channel = channel_var[str(CCD['CCDSEL'])]

    # calculate ranges
    if optimal_range and (lvl == 'L1b'):
        if TPsza < rswitch_sza:
            vmin = ranges_dayglow[channel][0]
            vmax = ranges_dayglow[channel][1]
        else:
            vmin = ranges_nightglow[channel][0]
            vmax = ranges_nightglow[channel][1]
    else:
        vmin, vmax, mean, std = calculate_range(image, ranges, nstd)

    # plot CCD image
    if (channel in flipped_CCDs) and (lvl == 'L1a'):
        nrows = np.arange(0, CCD['NROW'])
        ncols = np.arange(0, CCD['NCOL']+1)
        img = ax.pcolormesh(np.flip(ncols), nrows,
                            image, cmap=cmap,
                            vmax=vmax, vmin=vmin)

    else:
        img = ax.pcolormesh(image, cmap=cmap,
                            vmax=vmax, vmin=vmin)

    # add heights
    if lvl == 'L1b' and (CCD['CCDSEL'] != 7):
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


def simple_plot(CCD_dataframe, outdir, nstd=2, cmap='magma',
                ranges=None, optimal_range=False, format='png'):
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
    ranges : list, optional
        limits for custom cbar
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
                       nstd=nstd, cmap=cmap,
                       ranges=ranges, optimal_range=optimal_range,
                       format=format)
        else:
            for index, CCD in CCDs.iterrows():
                plot_image(CCD, ax, fig=fig, outpath=outpath,
                           nstd=nstd, cmap=cmap,
                           ranges=ranges, optimal_range=optimal_range,
                           format=format)


def orbit_plot(CCD_dataframe, outdir, nstd=2, cmap='magma',
               ranges=None, optimal_range=False, format='png'):
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
    ranges : tuple, optional
        Specify cbar limits, by default from nstd
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
                if lvl == 'L1b':
                    image = np.stack(image)

                # geolocation stuff
                (satlat, satlon,
                 nadir_sza,
                 TPlat, TPlon,
                 TPsza, TPssa, TPlt) = calculate_geo(CCD)

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
                                           ranges=ranges,
                                           optimal_range=optimal_range,
                                           format=format, save=False)

                fig.colorbar(img, ax=ax1)

                # print out additional information
                plt.figtext(0.15, 0.03, f'nadirSZA: {nadir_sza:.6}',
                            fontsize=10)
                plt.figtext(0.15, 0.06, f'tpLT: {TPlt}',
                            fontsize=10)
                plt.figtext(0.35, 0.03, f'tpSZA: {TPsza:.6}', fontsize=10)
                plt.figtext(0.35, 0.06, f'tpSSA: {TPssa:.6}', fontsize=10)

                # plot histogram
                generate_histogram(ax2, image, ranges,
                                   nstd)

                save_figure(outpath, CCD, format)
                fig.clear()
                plt.close()

    return


def all_channels_plot(CCD_dataframe, outdir, nstd=2, cmap='viridis',
                      ranges=None, optimal_range=False, format='png', version=None):

    check_type(CCD_dataframe)
    lvl = check_level(CCD_dataframe)

    fig, ax = plt.subplots(3, 3, figsize=(16, 9))
    fig.patch.set_facecolor('lightgrey')
    ax=ax.ravel()

    #dummy data for generation of cbar
    Z = np.random.rand(999, 999)
    x = np.arange(1, 1000, 1) 
    y = np.arange(1, 1000, 1) 

    # generate cbars
    cbaxes, cbars = [], []
    for i in range(0,len(ax)-2):
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        img = ax[i].pcolormesh(x,y,Z,cmap=cmap)
        cbaxes.append(inset_axes(ax[i], width="40%", height="6%", loc=8))
        cbars.append(plt.colorbar(img, cax = cbaxes[i], orientation='horizontal'))
        cbars[i].set_ticks([])

    # idle titles
    ax[0].set_title('IR1 (idle..)')
    ax[1].set_title('IR3 (idle..)')
    ax[2].set_title('UV1 (idle..)')
    ax[3].set_title('IR2 (idle..)')
    ax[4].set_title('IR4 (idle..)')
    ax[5].set_title('UV2 (idle..)')
    ax[6].set_title('NADIR (idle..)')

    # remove and replace some ax
    ax[8].remove()
    ax[7].remove()
    ax_cart = fig.add_subplot(3, 3, 8, projection=ccrs.PlateCarree())
    ax_cart.set_yticklabels([])
    ax_cart.set_xticklabels([])

    if outdir is not None:
        outpath = f"{outdir}ALL"

        if not os.path.exists(outpath):
            os.makedirs(outpath)

    for index, CCD in CCD_dataframe.iterrows():

        (satlat, satlon,
         nadir_sza,
         TPlat, TPlon,
         TPsza, TPssa, TPlt) = calculate_geo(CCD)

        # animation stuff (update plot and cbar) 
        if CCD['CCDSEL'] == 3:
            update_plot_cbar(CCD, ax[1], fig, cbars[1],
                             outdir, nstd, cmap,
                             ranges, optimal_range, format,
                             save=False, fontsize=10)

        elif CCD['CCDSEL'] == 2:
            update_plot_cbar(CCD, ax[4], fig, cbars[4],
                             outdir, nstd, cmap,
                             ranges, optimal_range, format,
                             save=False, fontsize=10)
        elif CCD['CCDSEL'] == 5:
            update_plot_cbar(CCD, ax[2], fig, cbars[2],
                             outdir, nstd, cmap,
                             ranges, optimal_range, format,
                             save=False, fontsize=10)

        else:
            update_plot_cbar(CCD, ax[CCD['CCDSEL'] - 1],
                             fig, cbars[CCD['CCDSEL'] - 1],
                             outdir, nstd, cmap,
                             ranges, optimal_range, format,
                             save=False, fontsize=10)

        if CCD['CCDSEL'] == 1:
            ax_cart.remove()
            ax_cart = fig.add_subplot(3, 3, 8, projection=ccrs.PlateCarree())
            ax_cart.set_yticklabels([])
            ax_cart.set_xticklabels([])

            generate_map(CCD, fig, ax_cart, satlat,
                         satlon, TPlat, TPlon, mark_size=12, legend_fsize=9,
                         labels=False)

        # additional information
        frames = []
        frames.append(plt.figtext(0.70, 0.04, f'nadirSZA: {nadir_sza:.4}',
                      fontsize=11))
        frames.append(plt.figtext(0.70, 0.07, f'tpLT: {TPlt}',
                      fontsize=11))
        frames.append(plt.figtext(0.85, 0.04, f'tpSZA: {TPsza:.4}',fontsize=11))
        frames.append(plt.figtext(0.85, 0.07, f'tpSSA: {TPssa:.4}',fontsize=11))

        frames.append(plt.figtext(0.70, 0.15, f'satlat: {satlat:.4}',fontsize=11))
        frames.append(plt.figtext(0.85, 0.15, f'satlon: {satlon:.4}',fontsize=11))
        frames.append(plt.figtext(0.70, 0.12, f'TPlat: {TPlat:.4}',fontsize=11))
        frames.append(plt.figtext(0.85, 0.12, f'TPlon: {TPlon:.4}',fontsize=11))
        frames.append(plt.figtext(0.70, 0.11, ('_____________________' +
                                    '_________________' +
                                    '______________')))
        frames.append(plt.figtext(0.70, 0.175, ('_____________________' +
                                    '_________________' +
                                    '______________')))
        frames.append(plt.figtext(0.70, 0.28, 'MATS', fontsize=22,
                    weight="bold"))
        frames.append(plt.figtext(0.76, 0.28, f'{lvl}', fontsize=11,
                    weight="bold"))
        if version != None:
            frames.append(plt.figtext(0.701, 0.255, f'v. {str(version)}', fontsize=11))
        if lvl == 'L1a':
            plt.figtext(0.70, 0.19, 'units: counts', weight='bold',fontsize= 11)
        if lvl == 'L1b':
            plt.figtext(0.70, 0.19, 'units: photons/nm', weight='bold',fontsize= 11)
        save_figure(outpath, CCD, format, filename=str(index))

        for i in range(0,len(frames)):
            Artist.remove(frames[i])

    return
