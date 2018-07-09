# Script to process 1D ion distribution functions
#
# David Stansby 2017
import numpy as np
import pandas as pd
from datetime import datetime

import helpers
import heliopy.data.helios as helios

import scipy.optimize as opt


# Dictionary mapping status codes to messages explaining the codes
statusdict = {1: 'Fitting successful',
              5: 'Less than 6 points available for fitting',
              7: 'No non-zero I1a data available',
              8: 'No non-zero I1b data available',
              9: 'Corrupted distribution file'}


def maxwellian_1D(v, n, v0, vth):
    prefactor = n * 4 * np.pi * v**2 * np.power(np.pi * vth**2, -1.5)
    return prefactor * np.exp(-((v - v0) / vth)**2)


def return_nans(status, time, instrument):
    '''Return np.nan for all parameters apart from "Time" and "status"'''
    assert type(status) == int, 'Status code must be an integer'
    fitparams = {'n_p': np.nan,
                 'v_p': np.nan,
                 'vth_p': np.nan,
                 'T_p': np.nan,

                 'Time': time,
                 'status': status,
                 'Instrument': instrument}
    return fitparams


def remove_n_counts(dist, n):
    '''Removes all bins with a pdf below *n* counts (relative to the lowest
    count)'''
    proptocounts = (dist['df'].values * (dist['v'].values)**2)
    dist = dist[proptocounts >= (n + 0.01) * np.min(proptocounts)]
    return dist


def oned_fitting(I1a, I1b, distparams, starttime, plotfigs=False):
    '''
    Method to return parameters from fitting one-dimensional Helios
    distributions.
    '''
    instrument = int(distparams['ion_instrument'])
    if I1a.empty:
        return return_nans(7, starttime, instrument)
    elif I1b.empty:
        return return_nans(8, starttime, instrument)

    I1a = remove_n_counts(I1a.copy(), 1)
    I1b = remove_n_counts(I1b.copy(), 2)

    if I1a.empty:
        return return_nans(7, starttime, instrument)
    elif I1b.empty:
        return return_nans(8, starttime, instrument)
    elif I1b.shape[0] < 6:
        return return_nans(5, starttime, instrument)

    # Convert temperature to thermal speed
    distparams['vth_i1a'] = helpers.temp2vth(distparams['Tp_i1a'])

    # Get maximum distribution function (assumed to be proton peak)
    maxdf_v = I1b['df'].idxmax()
    maxdf = I1b['df'].max() / (4 * np.pi * maxdf_v**2 *
                               np.power(np.pi * 40**2, -1.5))

    if instrument == 1:
        # If the difference between the maximum values of I1a and I1b is too
        # large, assume we have a garbled I1a distribution function
        if I1b['df'].max() / I1a['df'].max() > 10:
            return return_nans(9, starttime, instrument)
        elif np.any((I1a.loc[(I1a['v'] -
                             (I1b['df'].idxmax())).abs().idxmin(), 'df'] /
                     I1b['df'].max()) < 0.01):
            return return_nans(9, starttime, instrument)
    else:
        if I1a['df'].max() / I1b['df'].max() > 5:
            return return_nans(9, starttime, instrument)
    # Initial proton parameter guess
    p0 = (maxdf, maxdf_v, 40)

    # Residuals to minimise
    def resid(args, v, df):
        fit = maxwellian_1D(v, *args[:3])
        resids = df - fit
        return resids

    fitout = opt.leastsq(resid, p0, args=(I1b['v'].values, I1b['df'].values),
                         full_output=True)

    fitmsg = fitout[3]
    fitstatus = fitout[4]
    popt = fitout[0]
    if popt[0] < 0:
        return return_nans(11, starttime, instrument)
    fitparams = {'n_p': popt[0],
                 'v_p': popt[1],
                 'vth_p': popt[2]}

    fitparams['T_p'] = helpers.vth2temp(fitparams['vth_p'])
    fitparams['status'] = 1
    fitparams['Time'] = starttime
    fitparams['Instrument'] = instrument

    if plotfigs:
        print(fitparams, '\n')
        import matplotlib.pyplot as plt
        fitv = np.arange(200, 1600, 1)
        title = 'Helios 2 ' + str(starttime)

        # Plot fitted maxwellian
        fig, ax = plt.subplots()
        labels = {1: 'I1a', 2: 'I3'}
        ax.plot(I1a['v'], I1a['df'], marker='+', label=labels[instrument])
        ax.plot(I1b['v'], I1b['df'], marker='+', label='I1b')
        ax.plot(fitv, maxwellian_1D(fitv, fitparams['n_p'],
                                    fitparams['v_p'], fitparams['vth_p']),
                label='Proton fit')

        ax.set_title(title)
        ax.set_xlabel(r'$\left | v \right |$' + ' (km/s)')

        ax.legend()
        ax.set_ylim(bottom=0.5 * I1b['df'].min(),
                    top=2 * I1b['df'].max())
        ax.set_yscale('log')
        fig.tight_layout()

    return fitparams
