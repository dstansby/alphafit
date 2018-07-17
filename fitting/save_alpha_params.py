# Script to save fit parameters from alpha particle fitting processes
#
# David Stansby 2018
from datetime import datetime, timedelta
import os
import sys
import warnings

import scipy.optimize as opt
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


def save_fits(fits, probe, year, doy, fdir):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    # Save fits
    fname = 'h{}_{}_{}_alpha_fits.hdf'.format(probe, year, str(doy).zfill(3))
    fits.to_hdf(fdir / fname, 'fits', mode='w', format='f')


def find_speed_cut(I1a, I1b):
    # Take the peak velocity as the proton core, and only look at bins after
    # that
    peak_1D_v = I1a['df'].idxmax()
    # Calculate ratio between I1b and I1a
    I1a['I1b'] = np.interp(I1a.index.values, I1b.index.values,
                           I1b['df'].values)
    I1a['Ratio'] = I1a['df'] / I1a['I1b']
    I1a['Ratio'] /= I1a.loc[peak_1D_v, 'Ratio']
    last_high_ratio = ((I1a['Ratio'] < 0.8) & (I1a.index > peak_1D_v)).idxmax()
    return I1a, last_high_ratio


def bimaxwellian_fit(vs, df, guesses):

    # Residuals to minimize
    def resid(maxwell_params, vs, df):
        fit = helpers.bi_maxwellian_3D(vs[:, 0], vs[:, 1],
                                       vs[:, 2], *maxwell_params)
        return np.log10(df - fit)

    with np.errstate(invalid='ignore'):
        return opt.leastsq(resid, guesses, args=(vs, df),
                           full_output=True)


def fit_single_dist(probe, time, dist3D, I1a, I1b, corefit, params):
    I1a, speed_cut = find_speed_cut(I1a, I1b)
    # Cut out what we think is the alpha distribution
    alpha_dist = helpers.dist_cut(dist3D, speed_cut + 1)
    # Convert to km/s
    alpha_dist[['vx', 'vy', 'vz', '|v|']] /= 1e3
    # sqrt(2) charge to mass ratio correction
    alpha_dist[['vx', 'vy', 'vz', '|v|']] /= np.sqrt(2)

    df = alpha_dist['pdf'].values
    vs = alpha_dist[['vx', 'vy', 'vz']].values

    # Rotate velocities into field aligned co-ordinates
    B = corefit[['Bx', 'By', 'Bz']].values
    magempty = not np.isfinite(B[0])
    if not magempty:
        R = helpers.rotationmatrix(B)
        vprime = np.dot(R, vs.T).T
    else:
        R = None
        vprime = vs

    # Initial proton parameter guesses
    # Take maximum of distribution function for amplitude
    Aa_guess = np.max(df)
    # Take numerical ion velocity moment for v_a
    va_guess = [np.sum(df * vprime[:, 0]) / np.sum(df),
                np.sum(df * vprime[:, 1]) / np.sum(df),
                np.sum(df * vprime[:, 2]) / np.sum(df)]
    # Take proton thermal speeds as guesses for alpha thermal speeds
    vtha_perp_guess = corefit['vth_p_perp']
    vtha_par_guess = corefit['vth_p_par']
    guesses = (Aa_guess, vtha_perp_guess, vtha_par_guess,
               va_guess[0], va_guess[1], va_guess[2])
    fitout = bimaxwellian_fit(vprime, df, guesses)
    fitmsg = fitout[3]
    fitstatus = fitout[4]
    fitparams = fitout[0]

    fit_dict = helpers.process_fitparams(fitparams, 'a', vs, magempty, params, R)
    print(fit_dict)
    plotfigs = True
    if plotfigs and not isinstance(fit_dict, int) and not magempty:
        kwargs = {'last_high_ratio': speed_cut,
                  'alpha_dist': alpha_dist,
                  'fit_dict': fit_dict}
        from plot_fitted_dist_alphas import plot_dist
        import matplotlib.pyplot as plt
        plot_dist(time, probe, dist3D, params, corefit, I1a, I1b,
                  **kwargs)
        plt.show()
    return fit_dict


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
    fitlist = []
    for time, row in corefit.iterrows():
        # Only do alpha fitting if high data mode
        if time not in distparams.index:
            warnings.warn('Could not find time {} in distparams'.format(time))
            continue
        if not (distparams.loc[time]['data_rate'] == 1):
            continue
        # Only do alpha fitting if fitting proton core velocity was successful
        if not np.isfinite(row['vp_x']):
            continue
        dist3D = dists_3D.loc[time]
        I1a = I1as.loc[time]
        I1b = I1bs.loc[time]
        params = distparams.loc[time]
        fit_dict = fit_single_dist(probe, time, dist3D, I1a, I1b, row, params)

        if isinstance(fit_dict, int):
            continue
        fit_dict.update({'Time': time})
        fitlist.append(fit_dict)
        print(time)

    # End of a single day, put each day into its own DataFrame
    fits = pd.DataFrame(fitlist)
    if fits.empty:
        return

    fits = fits.set_index('Time', drop=True)
    # Make directory to save fits
    fdir = output_dir / 'helios{}'.format(probe) / 'fits' / str(year)
    save_fits(fits, probe, year, doy, fdir)


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
