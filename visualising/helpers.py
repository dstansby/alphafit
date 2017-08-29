import sys
import os
from datetime import timedelta
import pandas as pd

sys.path.append('./fitting')
from config import get_dirs

output_dir = get_dirs()


def load_corefit(probe, starttime, endtime, verbose=False):
    starttime_orig = starttime
    paramlist = []
    starttime_orig = starttime
    while starttime < endtime + timedelta(days=1):
        year = str(starttime.year)
        doy = starttime.strftime('%j')
        fname = 'h' + probe + '_' + year + '_' + doy + '_' + '3D_fits.h5'
        saveloc = os.path.join(output_dir,
                               'helios' + probe,
                               'fits',
                               year,
                               fname)
        try:
            params = pd.read_hdf(saveloc, 'fits')
        except FileNotFoundError:
            starttime += timedelta(days=1)
            if verbose:
                print('{}/{} corefit data not available'.format(year, doy))
            continue
        paramlist.append(params)
        starttime += timedelta(days=1)
        if verbose:
            print('{}/{} corefit data loaded'.format(year, doy))
    paramlist = pd.concat(paramlist)
    time = paramlist.index.get_level_values('Time')
    paramlist = paramlist[(time > starttime_orig) &
                          (time < endtime)]
    return paramlist
