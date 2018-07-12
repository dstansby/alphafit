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
import helpers_data

output_dir, corefit_code_dir = get_dirs()
probes = ['2', ]
years = range(1976, 1977)
doys = range(108, 109)


def fit_single_dist(probe, time, dist3D):

    kwargs = {}
    from plot_fitted_dist_alphas import plot_dist_time
    import matplotlib.pyplot as plt
    plot_dist_time(probe, time, **kwargs)
    plt.show()
    exit()


def fit_single_day(year, doy, probe):
    starttime, endtime = helpers.doy2stime_etime(year, doy)
    if starttime.year != year:
        return
    old_starttime = starttime
    # Uncomment next line to start from a specific datetime
    # starttime = starttime + timedelta(hours=23)
    if starttime != old_starttime:
        input('Manually setting starttime, press enter to continue')

    corefit = helios.corefit(probe, starttime, endtime)
    distparams = helios.distparams(probe,
                                   starttime, endtime, verbose=True)

    # Load a days worth of ion distribution functions
    try:
        dists_3D, I1as, I1bs, distparams = helpers_data.load_dists(
            probe, starttime, endtime)
    except RuntimeError as err:
        print(str(err))
        if 'No data available for times' in str(err):
            return
        raise

    # Loop through each timestamp
    for time, row in corefit.iterrows():
        # Only do alpha fitting if high data mode
        if not (distparams.loc[time]['data_rate'] == 1):
            continue
        # Only do alpha fitting if fitting proton core velocity was successful
        if not np.isfinite(row['vp_x']):
            continue
        print(dists_3D)
        dist3D = dists_3D.loc[time]
        fit_single_dist(probe, time, dist3D)


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
