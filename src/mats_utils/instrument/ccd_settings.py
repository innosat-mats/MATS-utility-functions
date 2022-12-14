import numpy as np

def calc_rowcol(nrowtot,bin,rounding='up'):
    if rounding == 'up':
        return np.ceil(nrowtot/bin)
    elif rounding == 'down':
        return np.floor(nrowtot/bin)
    else:
        raise ValueError('Rounding must be up or down')

def calc_row(nrowtot,bin,nrowskip=0,rounding='up'):
    nrow = calc_rowcol(nrowtot,bin,rounding)
    if nrow*bin + nrowskip > 515:
        raise ValueError('Reading outside of CCD')
    return nrow

def calc_col(ncoltot,bin,ncolskip=0,rounding='up'):
    ncol = calc_rowcol(ncoltot,bin,rounding)
    if ncol*bin + ncolskip > 2047:
        raise ValueError('Reading outside of CCD')
    return ncol

def calc_row_and_col(nrowtot,rowbin,nrowskip,ncoltot,colbin,ncolskip,rounding='up'):

    nrow = calc_row(nrowtot,rowbin,nrowskip,rounding)
    ncol = calc_col(ncoltot,colbin,ncolskip,rounding) - 1 #-1 from software ICD

    return nrow, ncol
