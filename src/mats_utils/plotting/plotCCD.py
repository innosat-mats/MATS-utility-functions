import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cartopy.crs as ccrs
import pandas as pd
from mats_utils.geolocation import satellite as satellite
from cartopy.feature.nightshade import Nightshade


flipped_CCDs = ['IR1', 'IR3', 'UV1', 'UV2']


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

    if isinstance(CCD_dataframe, (pd.core.frame.DataFrame, pd.core.series.Series)) is False:
        sys.exit("CCD_dataframe need to be converted to DataFrame!")
    
    return type(CCD_dataframe)

def plot_image(CCD,fig=None, outpath=None, nstd=2, cmap='inferno', custom_cbar=False,
                ranges=[0, 1000], format='png'):
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

    if fig == None:
        fig = plt.figure(figsize=(12, 3))

    # save parameters for plot
    channel = CCD['channel']
    image = CCD['IMAGE']
    texpms = CCD['TEXPMS']
    exp_date = CCD['EXPDate'].strftime("%Y-%m-%dT%H:%M:%S:%f")

    # filename
    outname = f"{CCD['ImageName'][:-4]}"

    # calc std and mean
    std = image.std()
    mean = image.mean()

    if custom_cbar:
        vmin = ranges[0]
        vmax = ranges[1]
    else:
        vmax = mean+nstd*std
        vmin = mean-nstd*std

    # orbital parameters
    (satlat, satlon, satLT,
        nadir_sza, nadir_mza,
        TPlat, TPlon,
        TPLT, TPsza, TPssa) = satellite.get_position(CCD['EXPDate'])

    # plot CCD image
    if channel in flipped_CCDs:
        nrows = np.arange(0, CCD['NROW'])
        ncols = np.arange(0, CCD['NCOL']+1)
        plt.pcolormesh(np.flip(ncols), nrows,
                        image, cmap=cmap,
                        vmax=vmax, vmin=vmin)

    else:
        plt.pcolormesh(image, cmap=cmap,
                        vmax=vmax, vmin=vmin)

    # print out additional information
    plt.figtext(0.1, 0.8, f'tpSZA: {TPsza:.6}',
                fontsize=10, color='white')
    plt.figtext(0.5, 0.8, (f'satlat, satlon: ({satlat:.6}' +
                            f', {satlon:.6})'),
                fontsize=10, color='white')
    plt.figtext(0.25, 0.8, f'TPlat, TPlon: ({TPlat:.6}, {TPlon:.6})',
                fontsize=10, color='white')

    plt.title(f'ch: {channel}; time: {exp_date}; TEXPMS: {texpms}')
    plt.tight_layout()

    # save figure
    if outpath != None:
        plt.savefig(f'{outpath}/{outname}.{format}', format=format)
        fig.clear()


def simple_plot(CCD_dataframe, outdir, nstd=2, cmap='inferno', custom_cbar=False,
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
            plot_image(CCDs, fig=fig, outpath=outpath, nstd=nstd, cmap=cmap, custom_cbar=custom_cbar,
                ranges=ranges, format=format)
        else:
            for index,CCD in CCDs.iterrows():
                plot_image(CCD, fig=fig, outpath=outpath, nstd=nstd, cmap=cmap, custom_cbar=custom_cbar,
                    ranges=ranges, format=format)



def orbit_plot(CCD_dataframe, outdir, nstd=2, cmap='inferno', custom_cbar=False,
               ranges=[0, 1000], format='png'):
    """
       Generates plots from CCD items: image, histogram and map.
       If simple_plot is True, only CCD image is plotted.
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
            if outdir != None:
                outpath = f"{outdir}CCDSEL{str(CCDno)}"

                if not os.path.exists(outpath):
                    os.makedirs(outpath)

            for index, CCD in CCDs.iterrows():

                # save parameters for plot
                channel = CCD['channel']
                image = CCD['IMAGE']
                texpms = CCD['TEXPMS']
                exp_date = CCD['EXPDate'].strftime("%Y-%m-%dT%H:%M:%S:%f")

                # filename
                outname = f"{CCD['ImageName'][:-4]}_{index}"

                # calc std and mean
                std = image.std()
                mean = image.mean()

                if custom_cbar:
                    vmin = ranges[0]
                    vmax = ranges[1]
                else:
                    vmax = mean+nstd*std
                    vmin = mean-nstd*std

                # orbital parameters
                (satlat, satlon,
                 satLT, nadir_sza,
                 nadir_mza, TPlat,
                 TPlon, TPLT, TPsza,
                 TPssa) = satellite.get_position(CCD['EXPDate'])

                fig = plt.figure(figsize=(10, 7))

                # generate subplot grid
                ax = plt.subplot2grid((2, 2), (1, 0),
                                      colspan=1, rowspan=1,
                                      projection=ccrs.PlateCarree(),
                                      fig=fig)
                ax1 = plt.subplot2grid((2, 2), (0, 0),
                                       rowspan=1, colspan=2, fig=fig)
                ax2 = plt.subplot2grid((2, 2), (1, 1), rowspan=1,
                                       colspan=1, fig=fig)

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

                # plot CCD image
                if channel in flipped_CCDs:
                    nrows = np.arange(0, CCD['NROW'])
                    ncols = np.arange(0, CCD['NCOL']+1)
                    img = ax1.pcolormesh(np.flip(ncols), nrows,
                                         image, cmap=cmap,
                                         vmax=vmax, vmin=vmin)

                else:
                    img = ax1.pcolormesh(image, cmap=cmap,
                                         vmax=vmax, vmin=vmin)

                ax1.set_title(f'ch: {channel}; time: '
                              + f'{exp_date}; TEXPMS: {texpms}')
                fig.colorbar(img, ax=ax1)

                # plot sat position and tangent point
                ax.scatter(satlon, satlat, s=10,
                           color='red', label='satellite pos.')
                ax.scatter(TPlon, TPlat, s=10,
                           color='green', label='TP pos.')
                ax.legend(ncol=2, fontsize=7, loc='lower right')

                # print out additional information
                plt.figtext(0.15, 0.03, f'nadirSZA: {nadir_sza:.6}',
                            fontsize=10)
                plt.figtext(0.15, 0.06, f'nadirMZA: {nadir_mza:.6}',
                            fontsize=10)
                plt.figtext(0.35, 0.03, f'tpSZA: {TPsza:.6}', fontsize=10)
                plt.figtext(0.35, 0.06, f'tpSSA: {TPssa:.6}', fontsize=10)

                # plot histogram
                nbins = int(1 + np.ceil(np.log2(len(image.flatten()))))
                ax2.hist(image.flatten(), bins=nbins, alpha=0.6,
                         density=True, range=[mean-nstd*std, mean+nstd*std])
                ax2.set_xlabel('counts')
                ax2.axvline(x=mean, label='mean',
                            linestyle='--', linewidth=1.5)
                ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax2.legend(loc='upper right')
                ax2.grid()
                plt.savefig(f'{outpath}/{outname}.{format}', format=format)
                fig.clear()

    return
