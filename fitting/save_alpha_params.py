# Script to save fit parameters from alpha particle fitting processes
#
# David Stansby 2018
import argparse
from datetime import datetime, timedelta
import logging
import os
import sys
import warnings

import scipy.optimize as opt
import numpy as np
import pandas as pd
from heliopy.data import helios
import matplotlib.pyplot as plt

from config import get_dirs

sys.path.append('visualising/')
import helpers
import helpers_data
import vis_helpers
output_dir, corefit_code_dir = get_dirs()

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(
    description='A test script for http://stackoverflow.com/q/14097061/78845'
)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
if args.verbose:
    logging.basicConfig(level=logging.INFO)

# Status dictionary to map status integers to descriptions
status_dict = {-1: "Couldn't find time in distparams",
               1: 'Fitting successful',
               2: 'No magnetic field available',
               3: 'Curve fitting failed',
               4: 'Low data rate distribution',
               5: 'No proton corefit data available',
               }

expected_params = set(['Ta_perp', 'Ta_par', 'va_x',
                       'va_y', 'va_z', 'n_a', 'Status'])


def check_output(fit_dict, status):
    """
    Check the output of the fitting process.

    Parameters
    ----------
    fit_dict : dict or int
        Must be either empty, or contain all expected fields.
    status : int
        See status_dict for information.
    """
    if status not in status_dict:
        raise RuntimeError('Status must be in the status dictionary')

    if status == 1:
        pass
    elif status == 2:
        for param in ['Ta_perp', 'Ta_par', 'n_a']:
            fit_dict[param] = np.nan
    else:
        assert fit_dict == {}, 'fit_dict must be empty for this error code'
        for param in expected_params:
            fit_dict[param] = np.nan
    fit_dict['Status'] = status
    assert set(fit_dict.keys()) == expected_params, 'Keys not as expected: {}'.format(fit_dict.keys())
    return fit_dict


def save_fits(fits, probe, year, doy, fdir):
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    # Save fits
    fname = 'h{}_{}_{}_alpha_fits.hdf'.format(probe, year, str(doy).zfill(3))
    fits.to_hdf(fdir / fname, 'fits', mode='w', format='f')


def find_speed_cut(I1a, I1b):
    # Take the peak velocity as the proton core, and only look at bins after
    # that
    peak_1D_v = np.nanargmax(I1a['df'].values)
    # Calculate ratio between I1b and I1a
    I1a_I1b = np.interp(I1a.index.values, I1b.index.values,
                        I1b['df'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = I1a['df'].values / I1a_I1b
        ratios = ratios / ratios[peak_1D_v]
    last_high_ratio = I1a.index.values[peak_1D_v + np.nanargmax(ratios[peak_1D_v:] < 0.8)]
    return ratios, I1a_I1b, last_high_ratio


def bimaxwellian_fit(vs, df, guesses):
    """
    Given a (n, 3) array of velocities and a (n, ) array of distribution
    function values, fit a bi-Maxwellian. guesses contains the initial
    parameter guesses.

    If fitting fails, returns None, otherwise returns the result of
    opt.curve_fit.
    """
    # Get rid of nans
    finite = np.isfinite(df)
    df = df[finite]
    vs = vs[finite, :]

    # Residuals to minimize
    def maxwell_to_fit(vs, *params):
        return helpers.bi_maxwellian_3D(vs[:, 0], vs[:, 1],
                                        vs[:, 2], *params)

    try:
        return opt.curve_fit(maxwell_to_fit, vs, df, p0=guesses)
    except Exception as e:
        warnings.warn("Fitting failed with the following error:\n{}".format(e))
        return


def bi_maxwellian_instrument_frame(vx, vy, vz, vth_perp, vth_par,
                                   vbx, vby, vbz, A, B):
    R = helpers.rotationmatrix(B.values)
    # Calculate bi-maxwellian parameters in field aligned frame
    '''A = n * 1e6 / (np.power(np.pi, 1.5) *
                   vth_perp * 1e3 *
                   vth_perp * 1e3 *
                   vth_par * 1e3)'''
    vbulkBframe = np.dot(R, np.array([vbx, vby, vbz]))
    df = helpers.bi_maxwellian_3D(vx, vy, vz,
                                  A, vth_perp, vth_par,
                                  *vbulkBframe)
    return df


def check_speed_cut(speed_cut, corefit, threshold=1e-3):
    '''
    Check if the speed *speed_cut* is close to the proton core distribution
    or not. Returns a bool.
    '''
    logger.info('Un-adjusted speed cut at {} km/s'.format(speed_cut))

    n = 200
    phis = np.linspace(-np.pi, np.pi, n)
    thetas = np.arcsin(np.linspace(-1, 1, n))
    thetas, phis = np.meshgrid(thetas, phis)
    vx, vy, vz = vis_helpers.sph2cart(speed_cut, thetas, phis)
    # Check that speed cut isn't too close to protons
    out = bi_maxwellian_instrument_frame(
        vx, vy, vz, corefit['vth_p_perp'], corefit['vth_p_par'],
        corefit['vp_x'], corefit['vp_y'], corefit['vp_z'],
        1, corefit[['Bx', 'By', 'Bz']])
    max_df = np.max(out)

    logger.info(
        'Maximum distribution funciton value at speed cut: {}'.format(max_df))
    if max_df > threshold:
        return False
    return True


def adjust_speed_cut(speed_cut, I1a, corefit):
    '''
    Check that the speed cut is not taken too close to the proton core
    population. If it is, return a higher speed that is not too close.
    '''
    vs = I1a.index.values
    vs = vs[vs >= speed_cut]
    for v in vs:
        if check_speed_cut(v, corefit):
            return v


def fit_single_dist(probe, time, dist3D, I1a, I1b, corefit, params):
    ratios, I1a_I1b, speed_cut = find_speed_cut(I1a, I1b)
    speed_cut = adjust_speed_cut(speed_cut, I1a, corefit)
    # Cut out what we think is the alpha distribution
    alpha_dist = helpers.dist_cut(dist3D, speed_cut + 1)
    df = alpha_dist['pdf'].values
    vs = alpha_dist[['vx', 'vy', 'vz']].values
    # Convert to km/s
    vs /= 1e3
    # sqrt(2) charge to mass ratio correction
    vs /= np.sqrt(2)
    df *= 4

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
    result = bimaxwellian_fit(vprime, df, guesses)
    fit_dict = {}
    if result is None:
        status = 3
    else:
        popt, pcov = result
        fit_dict = helpers.process_fitparams(popt, 'a', vs, magempty, params, R)
        if magempty:
            status = 2
        else:
            status = 1

    plotfigs = False
    if plotfigs and not isinstance(fit_dict, int) and not magempty:
        I1a['Ratio'] = ratios
        I1a['I1b'] = I1a_I1b
        kwargs = {'last_high_ratio': speed_cut,
                  'alpha_dist': alpha_dist,
                  'fit_dict': fit_dict}
        from plot_fitted_dist_alphas import plot_dist
        import matplotlib.pyplot as plt
        plot_dist(time, probe, dist3D, params, corefit, I1a, I1b,
                  **kwargs)
        plt.show()

    fit_dict = check_output(fit_dict, status)
    return fit_dict


# Loop through each timestamp
def fit_rows(x):
    time_rows, dists_3D, I1as, I1bs, distparams, probe = x
    fitlist = []
    for [time, row] in time_rows:
        # Only do alpha fitting if high data mode
        if time not in distparams.index:
            warnings.warn('Could not find time {} in distparams'.format(time))
            fit_dict = check_output({}, -1)
        elif not (distparams.loc[time]['data_rate'] == 1):
            fit_dict = check_output({}, 4)
        # Only do alpha fitting if fitting proton core velocity was successful
        elif not np.isfinite(row['vp_x']):
            fit_dict = check_output({}, 5)
        else:
            dist3D = dists_3D.loc[time]
            I1a = I1as.loc[time]
            I1b = I1bs.loc[time]
            params = distparams.loc[time]
            fit_dict = fit_single_dist(probe, time, dist3D, I1a, I1b,
                                       row, params)

        fit_dict.update({'Time': time})
        fitlist.append(fit_dict)
        print(time)
    return fitlist


def fit_single_day(year, doy, probe, startdelta=None, enddelta=None):
    """
    Method to fit a single day of Helios distribution functions. This function
    is responsible for saving the results for the given day to a file.
    """
    starttime, endtime = helpers.doy2stime_etime(year, doy)
    if starttime.year != year:
        return
    if startdelta is not None:
        parallel = False
        starttime += startdelta
        input('Manually setting starttime, press enter to continue')
        if enddelta is not None:
            endtime = starttime + enddelta
    else:
        parallel = True

    corefit = helios.corefit(probe, starttime, endtime)
    distparams = helios.distparams(probe, starttime, endtime, verbose=True)
    distparams = distparams.sort_index()
    # Load a days worth of ion distribution functions
    try:
        dists_3D, I1as, I1bs, distparams = helpers_data.load_dists(
            probe, starttime, endtime)
    except RuntimeError as err:
        print(str(err))
        if 'No data available for times' in str(err):
            return
        raise

    rows = list(corefit.iterrows())
    if parallel:
        import multiprocessing
        npools = 4
        # Split up times
        rows = [rows[i::4] for i in range(npools)]
        # Add distribution functions to each list of rows
        rows = [(row, dists_3D, I1as, I1bs, distparams, probe) for row in rows]
        with multiprocessing.Pool(npools) as p:
            mapped = p.map(fit_rows, rows)
        fitlist = []
        for l in mapped:
            fitlist += l
    else:
        fitlist = fit_rows((rows, dists_3D, I1as, I1bs, distparams, probe))
    # End of a single day, put each day into its own DataFrame
    fits = pd.DataFrame(fitlist).set_index('Time', drop=True).sort_index()
    assert fits.shape[0] == corefit.shape[0]
    # Make directory to save fits
    fdir = output_dir / 'helios{}'.format(probe) / 'fits' / str(year)
    save_fits(fits, probe, year, doy, fdir)


def do_fitting(probes, years, doys, startdelta=None, enddelta=None):
    '''
    Main method for doing all the fitting.
    '''
    # Loop through each probe
    for probe in probes:
        # Loop through years
        for year in years:
            # Loop through days of the year
            for doy in doys:
                fit_single_day(year, doy, probe, startdelta, enddelta)


if __name__ == '__main__':
    probes = ['2', ]
    years = range(1976, 1977)
    doys = range(108, 109)
    startdelta = None
    # startdelta = timedelta(seconds=1)
    # startdelta = timedelta(hours=1, minutes=29)
    enddelta = None
    # enddelta = timedelta(hours=1)
    do_fitting(probes, years, doys, startdelta, enddelta)
