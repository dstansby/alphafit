# Script to read and process 3D Helios ion distribution functions.
#
# David Stansby 2017
from datetime import timedelta

import pandas as pd
import numpy as np
import scipy.optimize as opt

import heliopy.data.helios as HeliosImport
import heliopy.constants as const
import heliopy.plasma as helioplas

from helpers import rotationmatrix


# Dictionary mapping status codes to messages explaining the codes
statusdict = {1: 'Fitting successful',
              2: 'No magnetic field data available',
              3: 'Magnetic field varies too much for a reliable fit',
              4: 'Fitted bulk velocity outside distribution velocity bounds',
              5: 'Less than 6 points available for fitting',
              6: 'Least square fitting failed',
              9: 'Think there is more than one distribution in file',
              10: 'Proton peak not present in 3D distribution',
              11: 'Number density physically unrealistic',
              12: 'Less than 3 angular bins available in either direction',
              13: 'Temperature physically unrealistic'}

# List of keys that must be present in output
keys = ['n_p', 'vth_p_perp', 'vth_p_par', 'Tp_perp', 'Tp_par',
        'vp_x', 'vp_y', 'vp_z',
        'Time', 'Status', 'Ion instrument', 'B instrument',
        'Bx', 'By', 'Bz', 'sigma B']


def return_nans(status, time, instrument):
    '''
    Return np.nan for all parameters apart from "Time", "Status",
    and "Ion instrument"
    '''
    assert type(status) == int, 'Status code must be an integer'
    fitparams = {}
    for key in keys:
        fitparams[key] = np.nan
    fitparams['Time'] = time
    fitparams['Status'] = status
    fitparams['Ion instrument'] = instrument
    fitparams['B instrument'] = -1
    return fitparams


def bi_maxwellian_3D(vx, vy, vz, A, vth_perp, vth_z, vbx, vby, vbz):
    '''
    Return distribution function at (vx, vy, vz),
    given 6 distribution parameters.
    '''
    # Put in bulk frame
    vx = vx - vbx
    vy = vy - vby
    vz = vz - vbz
    exponent = (vx / vth_perp)**2 + (vy / vth_perp)**2 + (vz / vth_z)**2
    return A * np.exp(-exponent)


def iondistfitting(dist, params, fit_1D, mag4hz, mag6s, starttime, I1a, I1b,
                   plotfigs=False):
    '''
    Method to do 3D fitting to an individual ion distribution
    '''
    output = {}
    instrument = int(params['ion_instrument'])
    output['Ion instrument'] = instrument
    # Return if the 1D fit thinks there are two distributions functions in
    # one file
    if fit_1D['status'] == 9:
        return return_nans(9, starttime, instrument)

    # Return if any of the counts are less than zero (indicates corruped file)
    if (dist['counts'] < 0).any():
        return return_nans(9, starttime, instrument)
    # Get rid of velocities higher than 100*velocities in the
    # I1a 1D distribution
    dist = dist[dist['|v|'] <= I1a['v'].max() * 1e3]

    # Return if not enought points to do fitting
    if dist.shape[0] <= 6:
        return return_nans(5, starttime, instrument)

    # Work out number of angular bins in each direction
    n_phi_bins = len(dist.index.get_level_values('Az').unique())
    n_theta_bins = len(dist.index.get_level_values('El').unique())
    # If less than a 3x3 grid, return
    if n_phi_bins < 3 or n_theta_bins < 3:
        return return_nans(12, starttime, instrument)

    # Return if the minimum velocity in the 3D distribution is not below
    # the velocity of the peak in I1b (assumed to be the proton peak)
    vs_3D = dist['|v|'] / 1e3
    if len(I1a) != 0:
        if vs_3D.min() > I1a['df'].argmax():
            return return_nans(10, starttime, instrument)

    # Estimate the times during which the distribution was measured
    #
    # Assumes that the timestamp in the distribution is when measurement was
    # started
    #
    # Also assume that bins were measured from low to high velocities. See
    # the blue books for more information
    E_bins = dist.index.get_level_values('E_bin').values
    # Energy bin in which the peak of the distribution was measured
    peak_Ebin = int(dist['pdf'].argmax()[1])
    min_Ebin = int(np.min(E_bins))
    max_Ebin = int(np.max(E_bins))
    # Take 3 energy bins either side of the peak distribution function
    min_dist_Ebin = peak_Ebin - max(peak_Ebin - 3, min_Ebin)
    max_dist_Ebin = peak_Ebin + max(peak_Ebin + 3, max_Ebin)

    dist_peaktime = starttime + timedelta(seconds=peak_Ebin)
    dist_starttime = starttime + timedelta(seconds=min_dist_Ebin)
    dist_endtime = starttime + timedelta(seconds=max_dist_Ebin + 1)

    print('Fitting distribution measured from',
          dist_starttime, '-->', dist_endtime)

    # Distribution function in s**3 / cm**6
    df = dist['pdf'].values
    # Spacecraft frame velocities in km/s
    vs = dist[['vx', 'vy', 'vz']].values / 1e3

    if mag4hz is None:
        mag4hzempty = True
    else:
        # Get magnetic field whilst distribution was built up
        mag = mag4hz[np.logical_and(mag4hz.index > dist_starttime,
                                    mag4hz.index < dist_endtime)]
        mag4hzempty = mag.empty
        # 4Hz data available
        if not mag4hzempty:
            output['B instrument'] = 1
            magempty = False

    # If no 4Hz data and no 6s data available
    if mag4hzempty and (mag6s is None):
        magempty = True
    elif mag4hzempty and (mag6s is not None):
        mag = mag6s[np.logical_and(mag6s.index > dist_starttime,
                                   mag6s.index < dist_endtime)]
        magempty = mag.empty
        # No 4Hz or 6s data
        if not magempty:
            output['B instrument'] = 2

    if not magempty:
        # TODO: Check magnetic field is static enough
        mag = mag[['Bx', 'By', 'Bz']].values
        # Use average magnetic field
        B = np.mean(mag, axis=0)
        sigmaB = np.std(mag, axis=0)
        sigmaB = np.linalg.norm(sigmaB)
        output['Bx'] = B[0]
        output['By'] = B[1]
        output['Bz'] = B[2]
        output['sigma B'] = sigmaB
        # Rotation matrix that rotates into field aligned frame where B = zhat
        R = rotationmatrix(B)
        # Rotate velocities into field aligned co-ordinates
        vprime = np.dot(R, vs.T).T
    else:
        output['B instrument'] = -1
        output['Bx'] = np.nan
        output['By'] = np.nan
        output['Bz'] = np.nan
        output['sigma B'] = np.nan
        # If no magnetic field, we can still get velocities
        vprime = vs

    # Initial proton parameter guesses
    # Take maximum of distribution function for amplitude
    Ap_guess = np.max(df)
    # Take numerical ion velocity moment for v_p
    vp_guess = [np.sum(df * vprime[:, 0]) / np.sum(df),
                np.sum(df * vprime[:, 1]) / np.sum(df),
                np.sum(df * vprime[:, 2]) / np.sum(df)]
    # Take proton temperature in distribution parameters for T_p (par and perp)
    # If no guess, or guess < 10km/s or guess > 100km/s take 40km/s for guess
    vthp_guess = helioplas.temp2vth(fit_1D['T_p'], const.m_p)
    if (not np.isfinite(vthp_guess)) or vthp_guess < 10 or vthp_guess > 100:
        vthp_guess = 40
    guesses = (Ap_guess, vthp_guess, vthp_guess,
               vp_guess[0], vp_guess[1], vp_guess[2])

    # Residuals to minimize
    def resid(maxwell_params, vprime, df):
        fit = bi_maxwellian_3D(vprime[:, 0], vprime[:, 1],
                               vprime[:, 2], *maxwell_params)
        return df - fit

    fitout = opt.leastsq(resid, guesses, args=(vprime, df),
                         full_output=True)

    fitmsg = fitout[3]
    fitstatus = fitout[4]
    fitparams = fitout[0]

    # Check on fit result status. Return if not successfull
    if fitstatus not in (1, 2, 3, 4):
        return return_nans(6, starttime, instrument)

    # Return if number density is physically unreasonable
    # (> 20 or < 0.1 times the peak in the distribution function)
    if (((fitparams[0] > 20 * np.max(df)) or
         (fitparams[0] < 0.1 * np.max(df))) and
            not magempty):
        return return_nans(11, starttime, instrument)

    # If thermal velocities are less than 5km/s throw away
    # TODO: Check me
    if (((fitparams[1] < 5) or
         (fitparams[2] < 5)) and
            not magempty):
        return return_nans(13, starttime, instrument)

    def process_fitparams(fitparams, species):
        v = fitparams[3:6]

        # If no magnetic field data, set temperatures to nans
        if magempty:
            fitparams[1:3] = np.nan
        else:
            # Otherwise transformt bulk velocity back into spacecraft frame
            v = np.dot(R.T, v)

        # Check that fitted bulk velocity is within the velocity range of
        # the distribution function, and return if any one component is outside
        for i in range(0, 3):
            if v[i] < np.min(vs[:, i]) or v[i] > np.max(vs[:, i]):
                return 4
            elif np.linalg.norm(v) < np.min(np.linalg.norm(vs, axis=1)):
                return 4

        fit_dict = {'vth_' + species + '_perp': np.abs(fitparams[1]),
                    'vth_' + species + '_par': np.abs(fitparams[2])}
        m = const.m_p
        fit_dict['T' + species + '_perp'] =\
            helioplas.vth2temp(fit_dict['vth_' + species + '_perp'], m)
        fit_dict['T' + species + '_par'] =\
            helioplas.vth2temp(fit_dict['vth_' + species + '_par'], m)
        # Original distribution has units s**3 / m**6
        # Get n_p in 1 / m**3
        n = (fitparams[0] * np.power(np.pi, 1.5) *
             np.abs(fitparams[1]) * 1e3 *
             np.abs(fitparams[1]) * 1e3 *
             np.abs(fitparams[2]) * 1e3)
        # Convert to 1 / cm**3
        n *= 1e-6
        fit_dict.update({'n_' + species: n})

        # Remove spacecraft abberation
        # Velocities here are all in km/s
        v_x = v[0] + params['helios_vr']
        v_y = v[1] + params['helios_v']
        v_z = v[2]
        fit_dict.update({'v' + species + '_x': v_x,
                         'v' + species + '_y': v_y,
                         'v' + species + '_z': v_z})

        return fit_dict

    # Put outputted fit parameters into a dictionary
    fit_dict = process_fitparams(fitparams, 'p')
    if isinstance(fit_dict, int):
        return return_nans(fit_dict, starttime, instrument)
    output.update(fit_dict)

    if magempty:
        status = 2
    else:
        status = 1
    output.update({'Time': starttime,
                   'Status': status,
                   'Ion instrument': instrument})

    #########################
    # Fitting finishes here #
    #########################
    if plotfigs:
        print(guesses)
        print(output)
        # from plot_fitted_dist import plot_dist
        # plot_dist(starttime, dist, params, pd.Series(output), I1a, I1b)

    # Check the keys in output dictionary
    outputkeys = list(output.keys())
    if sorted(outputkeys) != sorted(keys):
        print(sorted(outputkeys))
        print(sorted(keys))
        raise RuntimeError('Output keys different from expected keys')
    return output
