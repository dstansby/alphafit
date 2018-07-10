# Script to save fit parameters from alpha particle fitting processes
#
# David Stansby 2018
import sys
import numpy as np
import pandas as pd
from heliopy.data import helios

from config import get_dirs

sys.path.append('visualising/')
import helpers

output_dir, corefit_code_dir = get_dirs()
probes = ['1', ]
years = range(1975, 1976)
doys = range(74, 75)


def fit_single_day(year, doy, probe):
    starttime, endtime = helpers.doy2stime_etime(year, doy)
    if starttime.year != year:
        return
    corefit = helios.corefit(probe, starttime, endtime)
    distparams = helios.distparams(probe,
                                   starttime, endtime, verbose=True)

    # Loop through each timestamp
    for time, row in corefit.iterrows():
        # Only do alpha fitting if fitting proton core velocity was successful
        if not np.isfinite(row['vp_x']):
            continue
        from plot_fitted_dist_alphas import plot_dist_time
        import matplotlib.pyplot as plt
        plot_dist_time(probe, time)
        plt.show()
        return


def do_fitting():
    '''
    Main method for doing all the fitting.
    '''
    # Loop through each probe
    for probe in probes:
        # Loop through years
        for year in years:
            # Loop through days of the year
            for doy in doys:
                fit_single_day(year, doy, probe)


if __name__ == '__main__':
    do_fitting()
