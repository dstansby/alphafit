# Script to save fit parameters from fitting processes
#
# David Stansby 2017
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import heliopy.data.helios as helios
import heliopy.constants as const
from heliopy import config as helioconfig

from helpers import doy2dtime
import proton_core_1D as ions_1D
import proton_core_3D as ions_3D
from config import get_dirs

output_dir = get_dirs()


def save_fits(fits, probe, year, doy, dim, fdir):
    # Save fits
    fname = ('h' + probe + '_' + str(year) + '_' + str(doy).zfill(3) +
             '_' + dim + 'D_fits.h5')
    saveloc = os.path.join(fdir, fname)
    fits.to_hdf(saveloc, 'fits', mode='w', format='f')


def do_fitting(pltfigs=False):
    # Loop through each probe
    for probe in ['1', '2']:
        # Loop through years
        for year in range(1974, 1986):
            # Make directory to save fits
            fdir = os.path.join(output_dir,
                                'helios' + probe,
                                'fits',
                                str(year))
            if not os.path.exists(fdir):
                os.makedirs(fdir)

            # Loop through days of the year
            for doy in range(1, 367):
                starttime = doy2dtime(year, doy)
                if starttime.year != year:
                    continue
                endtime = starttime + timedelta(hours=24) -\
                    timedelta(microseconds=1)
                # Uncomment next line to start from a specific datetime
                # starttime = datetime()

                # Load 4Hz magnetic field data
                try:
                    mag4hz = helios.mag_4hz(probe, starttime, endtime,
                                            verbose=False, try_download=False)

                    # Deal with the flipped 4Hz data on Helios 2
                    if probe == '2':
                        mag4hz['By'] *= -1
                        mag4hz['Bz'] *= -1

                except Exception as err:
                    if str(err) != 'No raw 4Hz mag data available':
                        print(str(err))
                    mag4hz = None

                # Also load 6s data as backup
                try:
                    mag6s = helios.mag_ness(probe, starttime, endtime,
                                            verbose=False, try_download=False)
                except Exception as err:
                    if 'No 6s mag data avaialble' not in str(err):
                        print(str(err))
                    mag6s = None

                # If no magnetic field data available
                if mag4hz is None and mag6s is None:
                    # TODO: append some code to output
                    print('No mag data available for '
                          'probe {} year {} doy {}'.format(probe, year, doy))
                    continue

                # Load a days worth of ion distribution functions
                try:
                    dists_3D = helios.ion_dists(probe,
                                                starttime, endtime,
                                                verbose=True)
                    print('Loaded 3D dists')
                    dists_1D = helios.integrated_dists(probe,
                                                       starttime, endtime,
                                                       verbose=True)
                    print('Loaded 1D dists')
                    distparams = helios.distparams(probe,
                                                   starttime, endtime,
                                                   verbose=True)
                    print('Loaded distribution parameters')
                except RuntimeError as err:
                    print(str(err))
                    if 'No data available for times' in str(err):
                        continue
                    raise

                distparams['vth_i1a'] =\
                    np.sqrt(const.k_B * distparams['Tp_i1a'] / const.m_p) / 1e3

                # Add a velocity level to 1D dataframe
                I1as = dists_1D['a']
                I1bs = dists_1D['b']
                I1as['v'] = I1as.index.get_level_values('v')
                I1bs['v'] = I1bs.index.get_level_values('v')

                # Throw away zero distribution function values
                I1as = I1as[I1as['df'] != 0]
                I1bs = I1bs[I1bs['df'] != 0]

                # Re-order 3D index levels
                dists_3D = dists_3D.reorder_levels(
                    ['Time', 'E_bin', 'El', 'Az'], axis=0)
                dists_3D = dists_3D[dists_3D['counts'] != 1]

                # A handful of files seem to have some garbage counts in them
                dists_3D = dists_3D[dists_3D['counts'] < 32768]

                # Throw away high and low energy bins
                # (which contain just noise)
                dists_3D = dists_3D[
                    dists_3D.index.get_level_values('E_bin') > 3]
                dists_3D = dists_3D[
                    dists_3D.index.get_level_values('E_bin') < 32]

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
                    continue

                fits_3D = fits_3D.set_index('Time', drop=True)
                save_fits(fits_3D, probe, year, doy, '3', fdir)

                # Save 1D fits
                # save_fits(fits_1D, probe, year, doy, '1', fdir)


if __name__ == '__main__':
    do_fitting()
