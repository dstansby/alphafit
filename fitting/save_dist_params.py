# Script to save fit parameters from fitting processes
#
# David Stansby 2017
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import heliopy.data.helios as helios
from heliopy import config as helioconfig

import astropy.units as u
import astropy.constants as const

import helpers
import helpers_data
import proton_core_1D as ions_1D
import proton_core_3D as ions_3D
from config import get_dirs

probes = ['1', '2']
years = range(1974, 1986)
doys = range(1, 367)
output_dir, corefit_code_dir = get_dirs()


# Method to remove bad data
def remove_bad_data(data, probe):
    # Import keys that need to be set to nan when we manually get rid of
    # bad data
    keys = ions_3D.keys.copy()
    keys.remove('Time')
    keys.remove('Ion instrument')
    for comp in ['x', 'y', 'z']:
        keys.remove('vp_' + comp)
    manual_bad_data = pd.read_csv((corefit_code_dir /
                                   'fitting' /
                                   'manual_remove_intervals.csv'),
                                  parse_dates=[1, 2])
    for _, bad_data in manual_bad_data.iterrows():
        if str(bad_data['Probe']) != str(probe):
            continue
        bad_locations = ((data.index > bad_data['Start']) &
                         (data.index < bad_data['End']))
        if bad_locations.any():
            for key in keys:
                data.loc[bad_locations, key] = np.nan
            data.loc[bad_locations, 'Status'] = 5
            data.loc[bad_locations, 'B instrument'] = -1
    return data


def save_fits(fits, probe, year, doy, dim, fdir):
    # Save fits
    fname = ('h' + probe + '_' + str(year) + '_' + str(doy).zfill(3) +
             '_' + dim + 'D_fits.h5')
    saveloc = os.path.join(fdir, fname)
    fits.to_hdf(saveloc, 'fits', mode='w', format='f')


def fit_single_day(year, doy, probe, pltfigs):
    starttime, endtime = helpers.doy2stime_etime(year, doy)
    if starttime.year != year:
        return
    old_starttime = starttime
    # Uncomment next line to start from a specific datetime
    # starttime = starttime + timedelta(hours=23)
    if starttime != old_starttime:
        input('Manually setting starttime, press enter to continue')

    # Load magnetic field data
    mag4hz, mag6s = helpers_data.get_mag(probe, starttime, endtime)
    # If no magnetic field data available
    if mag4hz is None and mag6s is None:
        print('No mag data available for '
              'probe {} year {} doy {}'.format(probe, year, doy))
        return

    # Load a days worth of ion distribution functions
    try:
        dists_3D, I1as, I1bs, distparams = helpers_data.load_dists(
            probe, starttime, endtime)
    except RuntimeError as err:
        print(str(err))
        if 'No data available for times' in str(err):
            return
        raise

    distparams['vth_i1a'] = helpers.temp2vth(distparams['Tp_i1a'].values)

    # Add a velocity level to 1D dataframe
    I1as['v'] = I1as.index.get_level_values('v')
    I1bs['v'] = I1bs.index.get_level_values('v')

    # Throw away zero distribution function values
    I1as, I1bs = helpers_data.clean1D(I1as, I1bs)
    dists_3D = helpers_data.clean3D(dists_3D)

    # fitlist_1D = []
    fitlist_3D = []
    # Loop through individual times
    for time, dist_3D in dists_3D.groupby(level='Time'):
        params = distparams.loc[time].copy()

        if len(params.shape) > 1:
            params = params.iloc[0, :]
        I1a = I1as.loc[time]
        I1b = I1bs.loc[time]

        # Do 1D fit
        fit_1D = ions_1D.oned_fitting(I1a, I1b, params, time,
                                      plotfigs=pltfigs)
        # fitlist_1D.append(fit_1D)

        # Do 3D fit
        fit_3D = ions_3D.iondistfitting(
            dist_3D, params, fit_1D,
            mag4hz, mag6s, time, I1a, I1b, pltfigs)

        if pltfigs:
            import matplotlib.pyplot as plt
            plt.show()

        # Add orbital information
        for var in ['r_sun', 'clong', 'clat',
                    'carrot', 'earth_he_angle']:
            fit_3D[var] = params[var]
        fitlist_3D.append(fit_3D)
    # End of a single day, put each day into its own DataFrame
    # fits_1D = pd.DataFrame(fitlist_1D)
    fits_3D = pd.DataFrame(fitlist_3D)
    if fits_3D.empty:
        return

    fits_3D = fits_3D.set_index('Time', drop=True)
    print('Removing manually identified known bad data')
    fits_3D = remove_bad_data(fits_3D, probe)
    # Make directory to save fits
    fdir = os.path.join(output_dir,
                        'helios' + probe,
                        'fits',
                        str(year))
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    save_fits(fits_3D, probe, year, doy, '3', fdir)

    # Save 1D fits
    # save_fits(fits_1D, probe, year, doy, '1', fdir)


def do_fitting(pltfigs=False):
    '''
    Main method for doing all the fitting.
    '''
    # Loop through each probe
    for probe in probes:
        # Loop through years
        for year in years:
            # Loop through days of the year
            for doy in doys:
                fit_single_day(year, doy, probe, pltfigs)


if __name__ == '__main__':
    do_fitting()
