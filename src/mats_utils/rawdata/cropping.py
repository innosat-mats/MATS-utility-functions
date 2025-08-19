def set_crop_settings(channel, cropversion):

    if channel=="NADIR": 
        NRSKIP=0
        NRBIN=36
        NROW=14
        NCSKIP=0
        NCBIN=36
        NCOL=55
        NCBINFPGA=1
    else:
        if cropversion=='CROPF':
            if channel=="UV1": 
                NRSKIP=5
                NRBIN=2
                NROW=162
                NCSKIP=201
                NCBIN=40
                NCOL=43-1
                NCBINFPGA=0

            if channel=="UV2":
                NRSKIP=188
                NRBIN=2
                NROW=162
                NCSKIP=271
                NCBIN=40
                NCOL=43-1
                NCBINFPGA=0

            if channel=="IR1": 
                NRSKIP=49
                NRBIN=2
                NROW=217
                NCSKIP=27
                NCBIN=40
                NCOL=43-1
                NCBINFPGA=0

            if channel=="IR2": 
                NRSKIP=76
                NRBIN=2
                NROW=217
                NCSKIP=75
                NCBIN=40
                NCOL=43-1
                NCBINFPGA=0

            if channel=="IR3": 
                NRSKIP=67
                NRBIN=6
                NROW=73
                NCSKIP=156
                NCBIN=215
                NCOL=8-1
                NCBINFPGA=0

            if channel=="IR4": 
                NRSKIP=0
                NRBIN=6
                NROW=73
                NCSKIP=216
                NCBIN=215
                NCOL=8-1
                NCBINFPGA=0



        elif cropversion=='CROPD': #det vi kört i januari till april 20223
            if channel=='UV1':
                NRSKIP=65
                NRBIN=2
                NROW=132
                NCSKIP=201
                NCBIN=40
                NCOL=43
                NCBINFPGA=0

            elif channel=='UV2':
                NRSKIP=247
                NRBIN=2
                NROW=132
                NCSKIP=271
                NCBIN=40
                NCOL=43
                NCBINFPGA=0

            elif channel=='IR1':
                NRSKIP=109
                NRBIN=2
                NROW=187
                NCSKIP=27
                NCBIN=40
                NCOL=43
                NCBINFPGA=0

            elif channel=='IR2': 
                NRSKIP=136
                NRBIN=2
                NROW=187
                NCSKIP=75
                NCBIN=40
                NCOL=43
                NCBINFPGA=0

            elif channel=='IR3':
                NRSKIP=127
                NRBIN=6
                NROW=63
                NCSKIP=156
                NCBIN=200
                NCOL=8
                NCBINFPGA=0

            elif channel=='IR4':
                NRSKIP=60
                NRBIN=6
                NROW=63 
                NCSKIP=216
                NCBIN=200
                NCOL=8
                NCBINFPGA=0


        elif cropversion=='CROP_TO_BOTTOM':
            if channel=='UV1':
                NRSKIP=0 #65
                NRBIN=2
                NROW=164 #132
                NCSKIP=201
                NCBIN=40
                NCOL=43
                NCBINFPGA=0

            elif channel=='UV2':
                NRSKIP=0 #247
                NRBIN=2
                NROW=256 #132
                NCSKIP=271
                NCBIN=40
                NCOL=43
                NCBINFPGA=0

            elif channel=='IR1':
                NRSKIP=0 #109
                NRBIN=2
                NROW=242 #187
                NCSKIP=27
                NCBIN=40
                NCOL=43
                NCBINFPGA=0

            elif channel=='IR2': 
                NRSKIP=0 #136
                NRBIN=2
                NROW=255 # 87
                NCSKIP=75
                NCBIN=40
                NCOL=43
                NCBINFPGA=0

            elif channel=='IR3':
                NRSKIP=0 #127
                NRBIN=6
                NROW=84 #63
                NCSKIP=156
                NCBIN=215
                NCOL=8
                NCBINFPGA=0

            elif channel=='IR4':
                NRSKIP=0 #60
                NRBIN=6
                NROW=73 #63 
                NCSKIP=216
                NCBIN=215
                NCOL=8
                NCBINFPGA=0

    if NCBINFPGA==0:
        NCBINFPGA=1
    return NRSKIP, NRBIN, NROW, NCSKIP, NCBIN, NCOL, NCBINFPGA

def make_crop_filter(channel, cropversion):
    NRSKIP, NRBIN, NROW, NCSKIP, NCBIN, NCOL, NCBINFPGA=set_crop_settings(channel, cropversion)
    if NCBINFPGA==0:
        NCBINFPGA=1
    if NCBIN==0:
        NCBIN=1
    cropfilter={'channel':channel, 'NRSKIP':[NRSKIP,NRSKIP], 'NRBIN':[NRBIN, NRBIN], 'NROW': [NROW, NROW], 
            'NCSKIP':[NCSKIP,NCSKIP], 'NCBINCCDColumns':[NCBIN, NCBIN], 'NCOL': [NCOL, NCOL],'NCBINFPGAColumns':[NCBINFPGA,NCBINFPGA]}
    return cropfilter
