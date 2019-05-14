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

sys.path.append('visualising/')

import helpers_fit as helpers
import helpers_data
import vis_helpers

from config import get_dirs
output_dir, _ = get_dirs()
output_dir = output_dir / 'alphas'

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Increase output verbosity",
                    action="store_true")
parser.add_argument("-p", "--plot", help="Plot figures",
                    action="store_true")
args = parser.parse_args()

# Set the logger if verbose has been requested
logger = logging.getLogger(__name__)
if args.verbose:
    logging.basicConfig(level=logging.INFO)

# This file sets the intervals during which fitting is done
# See 'intervals_example.csv' for an example file
interval_file = ('fitting/intervals_example.csv')
# Set input parameters
days = helpers.read_intervals(interval_file)

# Status dictionary to map status integers to descriptions
status_dict = {1: 'Fitting successful',
               2: 'No magnetic field available',
               3: 'Curve fitting failed',
               4: 'Low data rate distribution',
               5: 'No proton corefit data available',
               6: 'No distribution left after cutting',
               7: 'Two distributions found in one file',
               8: 'I1b distribution function corrupted',
               9: 'Velocity out of range',
               10: 'Error on fitted parameters too high'
               }

# Expected parameters from the fitting process
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
    """
    Save fitted paramters to file
    """
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    # Save fits
    fname = 'h{}_{}_{}_alpha_fits.csv'.format(probe, year, str(doy).zfill(3))
    fits.to_csv(fdir / fname, mode='w')


def find_speed_cut(I1a, I1b, min_speed=0):
    I1a_df = I1a['df'].values
    # Get index of peak velocity
    peak_1D_v_idx = np.nanargmax(I1a_df)
    peak_1D_v = I1a.index.values[peak_1D_v_idx]

    # If this peak velocity is less than required speed, take minimum speed
    # instead
    min_speed = max(peak_1D_v, min_speed)
    min_speed_idx = np.nanargmin(np.abs(I1a.index.values - min_speed))
    min_speed_v = I1a.index.values[peak_1D_v_idx]

    # Calculate ratio between I1b and I1a
    I1a_I1b = np.interp(I1a.index.values, I1b.index.values,
                        I1b['df'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = I1a_df / I1a_I1b
        ratios = ratios / ratios[peak_1D_v_idx]
        threshold = 0.8
        good_ratio_idx = ((ratios[min_speed_idx:] < threshold) &
                          (ratios[min_speed_idx:] > 0.1))
    # Get the energy bin number of the speed cut
    ratio_idx = min_speed_idx + np.nanargmax(good_ratio_idx)
    speed_cut = I1a.index.values[ratio_idx]
    return ratios, I1a_I1b, speed_cut, ratio_idx


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


def check_speed_cut(speed_cut, corefit, threshold=1e-2):
    '''
    Check if the speed *speed_cut* is close to the proton core distribution
    or not. Returns a bool.
    '''
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


def valid_ratio(rs, idx):
    """
    Return True if r is a valid ratio in an alpha distribution.
    """
    n = rs.size
    if idx + 2 > n:
        return False
    if rs[idx] < 0.8 and rs[idx] > 0.1:
        return True
    return False


def fit_single_dist(probe, time, dist3D, I1a, I1b, corefit, params):
    """
    Fit a single distribution.
    """
    dist3D = dist3D.loc[dist3D['counts'] >= 2]
    # First, find speed at which to cut the distribution
    ratios, I1a_I1b, speed_cut, ratio_idx = find_speed_cut(I1a, I1b)
    # If the energy bin being cut is >= 30, there will me at most 3 energy bins
    # and fitting won't work
    if ratio_idx >= 30:
        return check_output({}, 8)
    # If the mass per charge ratio is not as expected for alphas, assume
    # something has gone wrong and return
    if not valid_ratio(ratios, ratio_idx):
        return check_output({}, 7)
    logger.info('Un-adjusted speed cut at {} km/s'.format(speed_cut))

    # Check that the speed cut is not too close to the proton core. If it is,
    # move it up so it is far enough away.
    speed_cut = adjust_speed_cut(speed_cut, I1a, corefit)
    if speed_cut is None:
        return check_output({}, 6)
    logger.info('Adjusted speed cut at {} km/s'.format(speed_cut))

    # Re-iterate the speed cut procedure with a new minimum speed.
    _, _, speed_cut, _ = find_speed_cut(I1a, I1b, min_speed=speed_cut)
    logger.info('Final speed cut at {} km/s'.format(speed_cut))

    # Cut out what we think is the alpha distribution
    # NOTE: this is in the instrument frame of reference and has no alpha
    # particle corrections applied
    alpha_dist = helpers.dist_cut(dist3D, speed_cut + 1)
    if alpha_dist.empty:
        return check_output({}, 6)
    df = alpha_dist['pdf'].values
    vs = alpha_dist[['vx', 'vy', 'vz']].values

    # Convert velocities from m/s to km/s
    vs /= 1e3
    # Do alpha particle corrections on the distibution values, since the values
    # are dervied assuming protons
    vs, df = helpers.distribution_function_correction(vs, df, 2)

    # Rotate velocities into field aligned co-ordinates
    B = corefit[['Bx', 'By', 'Bz']].values
    magempty = not np.isfinite(B[0])
    if not magempty:
        R = helpers.rotationmatrix(B)
        vprime = np.dot(R, vs.T).T
    else:
        R = None
        vprime = vs

    # Initial alpha parameter guesses
    # Take maximum of distribution function for amplitude
    Aa_guess = np.max(df)
    # Take numerical ion velocity moment for v_a
    va_guess = [np.sum(df * vprime[:, 0]) / np.sum(df),
                np.sum(df * vprime[:, 1]) / np.sum(df),
                np.sum(df * vprime[:, 2]) / np.sum(df)]
    # Take proton thermal speeds as initial guesses for alpha thermal speeds
    vtha_perp_guess = corefit['vth_p_perp']
    vtha_par_guess = corefit['vth_p_par']
    guesses = (Aa_guess, vtha_perp_guess, vtha_par_guess,
               va_guess[0], va_guess[1], va_guess[2])
    logger.info('Initial parameters guesses are {}'.format(guesses))

    # Do fitting
    result = bimaxwellian_fit(vprime, df, guesses)
    fit_dict = {}
    if result is None:
        status = 3
    else:
        popt, pcov = result
        perr = np.sqrt(np.diag(pcov))
        logger.info(f'Absolute error is {perr}')
        relerr = np.abs(perr) / popt
        logger.info(f'Relative error is {relerr}')
        if np.any(relerr > 1):
            status = 10
        else:
            # Convert the output of the fitting process to a sensible dict with
            # the alpha parameters.
            fit_dict = helpers.process_fitparams(popt, 'a', vs, magempty, params,
                                                 R, particle_mass=4)
            logger.info(f'Final parameters are {fit_dict}')
            if magempty:
                status = 2
            else:
                status = 1

    # If requested, visualise the resulting fit and original distributions
    if (args.plot and
            not isinstance(fit_dict, int) and
            status == 1):
        I1a['Ratio'] = ratios
        I1a['I1b'] = I1a_I1b
        kwargs = {'last_high_ratio': speed_cut,
                  'alpha_dist': alpha_dist,
                  'fit_dict': fit_dict}
        from plot_fitted_dist_alphas import plot_dist, plot_angular_cuts
        import matplotlib.pyplot as plt
        plot_angular_cuts(alpha_dist, fit_dict, R, m=4, moverq=2)
        plot_dist(time, probe, dist3D, params, corefit, I1a, I1b,
                  **kwargs)
        # plt.savefig('{}.png'.format(time))
        plt.show()

    fit_dict = check_output(fit_dict, status)
    # If the radial component of velocity is lies outside the array of
    # experimental data, reject the fit as it is probably overfitted in the
    # radial direction
    if fit_dict['va_x'] < np.min(vs[:, 0]):
        return check_output({}, 9)
    return fit_dict


# Loop through each timestamp
def fit_rows(x):
    time_rows, dists_3D, I1as, I1bs, distparams, probe = x
    fitlist = []
    for [time, row] in time_rows:
        # Only do alpha fitting if high data mode
        if (distparams.loc[time]['data_rate'].size != 1):
            fit_dict = check_output({}, 7)
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
        print(time, fit_dict['Status'])
    return fitlist


def fit_single_day(year, doy, probe, startdelta=None, enddelta=None):
    """
    Method to fit a single day of Helios distribution functions. This function
    is responsible for saving the results for the given day to a file.
    """
    starttime, endtime = helpers.doy2stime_etime(year, doy)
    if starttime.year != year:
        return
    parallel = True
    if args.plot:
        parallel = False
    if startdelta is not None:
        starttime += startdelta
        input('Manually setting starttime, press enter to continue')
        if enddelta is not None:
            endtime = starttime + enddelta

    try:
        corefit = helios.corefit(probe, starttime, endtime).data
        # The first corefit release has a sign error in the magnetic field.
        # If using v1, correct this
        if 'data_rate' not in corefit.keys():
            corefit[['Bx', 'By', 'Bz']] = corefit[['Bx', 'By', 'Bz']] * -1
    except RuntimeError:
        return
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
    fdir = output_dir / 'helios{}'.format(probe) / str(year)
    save_fits(fits, probe, year, doy, fdir)


def do_fitting(days, startdelta=None, enddelta=None):
    '''
    Main method for doing all the fitting.
    '''
    for [probe, year, doy] in days:
        fit_single_day(year, doy, probe, startdelta, enddelta)


if __name__ == '__main__':
    startdelta = None
    # startdelta = timedelta(seconds=1)
    # startdelta = timedelta(minutes=9, seconds=30)
    enddelta = None
    # enddelta = timedelta(hours=1)
    do_fitting(days, startdelta, enddelta)
