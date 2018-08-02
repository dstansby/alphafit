# Methods to plot fitted distribution functions
#
# David Stansby 2017
from datetime import timedelta as dt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatch
import scipy.interpolate as interp
import numpy as np
import pandas as pd
import sys

sys.path.append('../fitting')

import heliopy.data.helios as helios

import vis_helpers as helpers
import helpers_fit as fit_helpers


def contour2d(x, y, pdf, showbins=True, levels=10, add1overe=False):
    """Perform a countour plot of 2D distribution function data."""
    ax = plt.gca()
    pdf = np.log(pdf)
    if type(levels) == int:
        levels = np.linspace(np.nanmin(pdf), np.nanmax(pdf), levels)
    ax.tricontourf(x, y, pdf, levels=levels, cmap='viridis')
    ax.tricontour(x, y, pdf, levels=levels, linestyles='-', colors='k',
                  linewidths=0.5, alpha=0.8)
    if add1overe:
        ax.tricontour(x, y, pdf, levels=[np.nanmax(levels) - 1],
                      linestyles='-', colors='k',
                      linewidths=1)
    if showbins:
        ax.scatter(x, y, color='k', marker='+', s=4, alpha=0.5)


def integrated_1D(vth_perp, vth_par, vbx, vby, vbz, n, params, B, moverq=1):
    '''
    Construct an integrated 1D distribution function from bi-Maxwellian
    parameters.

    Returns the integrated 1D function in instrument velocities. Use *moverq*
    in terms of proton units to normalise appropriately.
    '''
    squashing_factor = np.sqrt(moverq)
    R = helpers.rotationmatrix(B.values)
    vrminlim, vrmaxlim = 200, 1400
    # Calculate reduced 3D fit
    phis = np.linspace(-np.pi, np.pi, 100)
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 100)
    modvs = np.arange(vrminlim, vrmaxlim, 5.)
    modvs, thetas, phis = np.meshgrid(modvs, thetas, phis)
    modvs = modvs.flatten()
    thetas = thetas.flatten()
    phis = phis.flatten()
    vx, vy, vz = helpers.sph2cart(modvs, thetas, phis)
    v = np.array([vx, vy, vz]).T
    # Transform into rotated frame
    v = np.dot(R, v.T).T

    # Calculate bi-maxwellian parameters in field aligned frame
    A = n * 1e6 / (np.power(np.pi, 1.5) *
                   vth_perp * 1e3 *
                   vth_perp * 1e3 *
                   vth_par * 1e3)
    vx -= params['helios_vr']
    vy -= params['helios_v']
    vbulkBframe = np.dot(R, np.array([vbx, vby, vbz]))
    df = bi_maxwellian_3D(v[:, 0], v[:, 1], v[:, 2],
                          A, vth_perp, vth_par,
                          *vbulkBframe)
    # Distribution function and velocities are in the alpha frame of reference
    # Take into account different mass per charge ratios, and transform
    # into the instrument frame of reference
    modvs, df = fit_helpers.distribution_function_correction(modvs, df, 1 / moverq)
    # df and modvs are now in the isntrument frame
    # Mutliply by area element
    df *= np.cos(thetas) * modvs**2
    index = pd.MultiIndex.from_arrays([modvs, thetas, phis],
                                      names=['v', 'theta', 'phi'])

    df = pd.DataFrame({'Reduced fit': df}, index=index)
    df = df['Reduced fit'].groupby(level='v').sum()
    # df /= df.max()
    # df[df < 1e-3] = np.nan
    return df


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


def perp_par_maxwellian(n, wperp, wpar, vperp, vpar):
    out = np.exp(-((vpar / wpar)**2 +
                   (vperp / wperp)**2))
    out *= 2 * np.power(np.pi, 0.5) * n / (wperp * wperp * wpar)
    return out


def perp_par_vels(vs, bulkv, R):
    # Get perp/parallel velocities
    for i in range(0, 3):
        vs[:, i] -= bulkv[i]
    vs = np.dot(R, vs.T).T
    vpar = vs[:, 2]
    vperp = np.linalg.norm(vs[:, 0:2], axis=1)
    vpar = np.concatenate((vpar, vpar))
    vperp = np.concatenate((vperp, -vperp))
    return vperp, vpar


def plot_dist_time(probe, time, **kwargs):
    # Calls plot_dist with a given time. Uses already processed values
    corefit = helios.corefit(probe,
                             time - dt(seconds=20),
                             time + dt(seconds=20)).data
    corefit = corefit.loc[corefit.index == time]
    if corefit.shape[0] != 1:
        raise ValueError('Could not find fitted parameters at requested time')
    corefit = pd.Series(corefit.iloc[0])
    args = (probe, time.year, int(time.strftime('%j')),
            time.hour, time.minute, time.second)
    dist = helios.ion_dist_single(*args)
    params = helios.distparams_single(*args)
    I1a, I1b = helios.integrated_dists_single(*args)

    plot_dist(time, probe, dist, params, corefit, I1a, I1b, **kwargs)


def slice_dist(vs, pdf, plane):
    '''
    Get distribution slices. Interpolates on to either the x-y, x-z or y-z
    plane.

    Parameters
    ----------
    plane : int
        0 for y-z, 1 for x-z, 2 for y-z.
    '''
    vlim = 2000
    nbins = 100
    dim1, dim2 = np.meshgrid(np.linspace(-vlim, vlim, nbins + 1),
                             np.linspace(-vlim, vlim, nbins + 1))
    zeros = np.zeros(dim1.shape)
    if plane == 0:
        sampling_points = [zeros, dim1, dim2]
    elif plane == 1:
        sampling_points = [dim1, zeros, dim2]
    elif plane == 2:
        sampling_points = [dim1, dim2, zeros]
    else:
        raise ValueError('plane must be 1, 2 or 3')
    pdf = interp.griddata(vs, pdf, np.array(sampling_points).T,
                          method='linear').T
    dim1 = dim1.ravel()
    dim2 = dim2.ravel()
    pdf = pdf.ravel()
    dim1 = dim1[np.isfinite(pdf)]
    dim2 = dim2[np.isfinite(pdf)]
    pdf = pdf[np.isfinite(pdf)]
    return dim1, dim2, pdf


def plot_RTN_cuts(dist, ax1, ax2):
    '''
    Plot slices of dist on two different axes. ax1 is the RT plane, ax2 is the
    RN plane.
    '''
    vs = dist[['vx', 'vy', 'vz']].values
    pdf = dist['pdf'].values
    levels = np.linspace(np.log(pdf).min(),
                         np.log(pdf).max(), 20)
    # Slice along B (which is along z-axis)
    x, z, slice_pdf = slice_dist(vs, pdf, 1)
    plt.sca(ax1)
    contour2d(z, x, slice_pdf, levels=levels, showbins=False, add1overe=True)

    # Slice perp to B
    x, y, slice_pdf = slice_dist(vs, pdf, 2)
    plt.sca(ax2)
    contour2d(y, x, slice_pdf, levels=levels, showbins=False, add1overe=True)
    for a in [ax1, ax2]:
        a.set_aspect('equal', 'datalim')
    ax1.set_ylabel(r'$v_{r}$ (km/s)')
    ax1.set_xlabel(r'$v_{n}$ (km/s)')
    ax2.set_xlabel(r'$v_{t}$ (km/s)')


def plot_alpha_dist(dist, ax1, ax2):
    plot_RTN_cuts(dist, ax1, ax2)


def plot_dist(time, probe, dist, params, output, I1a, I1b,
              last_high_ratio=np.nan,
              alpha_dist=None,
              fit_dict=None):
    dist = dist.copy()
    magempty = np.any(~np.isfinite(output[['Bx', 'By', 'Bz']].values))
    if magempty:
        raise RuntimeError('No magnetic field present')

    title = 'Helios {} '.format(probe) + str(time)
    fig = plt.figure(figsize=(12, 10))
    spec = gridspec.GridSpec(ncols=4, nrows=3)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(spec[1, 0:2])
    ax4 = fig.add_subplot(spec[2, 0:2], sharex=ax3)
    ax5 = fig.add_subplot(spec[0, 2], sharey=ax1)
    ax6 = fig.add_subplot(spec[0, 3], sharey=ax1)
    ax = [ax1, ax2, ax3, ax4, ax5, ax6]

    # fig.suptitle(title)
    dist[['vx', 'vy', 'vz', '|v|']] /= 1e3
    alpha_dist[['vx', 'vy', 'vz', '|v|']] /= (np.sqrt(2) * 1e3)
    dist_vcentre = dist.copy()
    # Distribution is in spacecraft frame, but output velocities are in
    # solar wind frame, so correct
    dist_vcentre['vx'] += params['helios_vr']
    dist_vcentre['vy'] += params['helios_v']
    alpha_dist['vx'] += params['helios_vr']
    alpha_dist['vy'] += params['helios_v']
    sqrt2 = np.sqrt(2)
    plot_RTN_cuts(dist_vcentre, ax[0], ax[1])
    ax[0].scatter(fit_dict['va_z'], fit_dict['va_x'], marker='+', color='r')
    ax[1].scatter(fit_dict['va_y'], fit_dict['va_x'], marker='+', color='r')
    ax[0].scatter(fit_dict['va_z'] * sqrt2, fit_dict['va_x'] * sqrt2, marker='+', color='k')
    ax[1].scatter(fit_dict['va_y'] * sqrt2, fit_dict['va_x'] * sqrt2, marker='+', color='k')
    ax[0].scatter(output['vp_z'], output['vp_x'], marker='x', color='k')
    ax[1].scatter(output['vp_y'], output['vp_x'], marker='x', color='k')

    print(output)

    # Plot formatting
    ax[1].tick_params(axis='y', labelleft=False, labelright=True,
                      left=False, right=True)

    # Calculate 1D reduced distribution from data
    vs = dist['|v|'].groupby(level=['E_bin']).mean()
    pdf = dist['pdf'].groupby(level=['E_bin']).sum() * vs**2

    ax[2].plot(I1a['df'] / I1a['df'].max(),
               marker='x', label='I1a')
    ax[2].plot(I1b['df'] / I1b['df'].max(),
               marker='x', label='I1b')

    for axnum in [0, 1]:
        circ = mpatch.Circle((0, 0), last_high_ratio,
                             edgecolor='k', facecolor='none')
        ax[axnum].add_patch(circ)

    ax[2].plot(I1a['I1b'] / I1a['I1b'].max(),
               marker='x', label='I1b interp')

    ax[3].plot(I1a['Ratio'], marker='x')
    for axnum in [2, 3]:
        ax[axnum].axvline(last_high_ratio, color='k')
    ax[3].axhline(1, color='k')

    if not magempty:
        protons_1d = integrated_1D(output['vth_p_perp'], output['vth_p_par'],
                                   output['vp_x'], output['vp_y'], output['vp_z'],
                                   output['n_p'], params, output[['Bx', 'By', 'Bz']])
        ax[2].plot(protons_1d.index.values,
                   protons_1d / protons_1d.max(), label='Proton fit')
        alphas_1d = integrated_1D(helpers.temp2vth(fit_dict['Ta_perp'], m=4),
                                  helpers.temp2vth(fit_dict['Ta_par'], m=4),
                                  fit_dict['va_x'], fit_dict['va_y'], fit_dict['va_z'],
                                  fit_dict['n_a'], params, output[['Bx', 'By', 'Bz']],
                                  moverq=2)
        ax[2].plot(alphas_1d.index.values,
                   alphas_1d / (protons_1d.max()), label='Alpha fit')
        # Formatting
        ax[2].legend(frameon=False)
        ax[2].set_yscale('log')
        ax[2].set_xlabel(r'$|v|$' + ' (km/s)')
        ax[2].set_ylim(1e-3, 2)
        ax[2].set_xlim(400, 1600)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

    plot_RTN_cuts(alpha_dist, ax[4], ax[5])
    ax[4].scatter(fit_dict['va_z'], fit_dict['va_x'], marker='+', color='r')
    ax[5].scatter(fit_dict['va_y'], fit_dict['va_x'], marker='+', color='r')
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)

    ax[0].set_xlim(-500, 500)
    ax[1].set_xlim(-500, 500)
    fig.tight_layout()


def bimax_angular_cut(theta, phi, modv, fit_dict, R, m=1):
    vx, vy, vz = helpers.sph2cart(modv, theta, phi)
    vs = np.column_stack((vx, vy, vz))
    vprime = np.dot(R, vs.T).T
    vbulk = np.dot(R, np.array([fit_dict['va_x'], fit_dict['va_y'], fit_dict['va_z']]).T)

    vth_perp = helpers.temp2vth(fit_dict['Ta_perp'], m=m)
    vth_par = helpers.temp2vth(fit_dict['Ta_par'], m=m)
    A = fit_dict['n_a'] / (np.power(np.pi, 1.5) *
                           vth_perp *
                           vth_perp *
                           vth_par) * 1e-3
    df = bi_maxwellian_3D(vprime[:, 0], vprime[:, 1], vprime[:, 2], A,
                          vth_perp, vth_par,
                          *vbulk)
    return df


def plot_angular_cuts(dist, fit_dict, R, moverq=1, m=1):
    nel = len(dist.groupby(level='El'))
    naz = len(dist.groupby(level='Az'))
    fig, axs = plt.subplots(nrows=naz, ncols=nel, figsize=(10, 10),
                            sharex=True, sharey=True)
    for i, (el_bin, el_cut) in enumerate(dist.groupby(level='El')):
        for j, (az_bin, az_cut) in enumerate(el_cut.groupby(level='Az')):
            ax = axs[i, j]
            df = az_cut['pdf'].values
            modvs = az_cut['|v|'].values / 1e3

            # Plot data
            ax.plot(modvs, df, marker='+')

            # Calculate and plot fits
            theta = az_cut['theta'].iloc[0]
            phi = az_cut['phi'].iloc[0]
            vs_fit = np.linspace(600, 1600, 100)
            df_fit = bimax_angular_cut(theta, phi, vs_fit, fit_dict, R, m=m)
            vs_fit, df_fit = fit_helpers.distribution_function_correction(vs_fit, df_fit, 1 / moverq)
            ax.plot(vs_fit, df_fit, scaley=False)
            ax.text(0.1, 0.1,
                    '$\\theta$ = {:.1f}\n$\phi$ = {:.1f}'.format(
                        np.rad2deg(theta), np.rad2deg(phi)),
                    transform=ax.transAxes, fontsize=8)

    ax.set_yscale('log')
    ax.set_ylim(1e-12, 1e-9)
    ax.set_xlim(600, 1200)


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import matplotlib.pyplot as plt
    description = ('Plot a single Helios 3D ion distribution '
                   'along with fitted distribution.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('probe', metavar='p', type=str, nargs=1,
                        help='Helios probe')
    parser.add_argument('date', metavar='d', type=str, nargs=1,
                        help='Date - must be formatted as YYYY/MM/DD')
    parser.add_argument('time', metavar='t', type=str, nargs=1,
                        help='Time - must be formatted as HH:MM:SS')

    args = parser.parse_args()
    date = datetime.strptime(args.date[0] + ' ' + args.time[0],
                             '%Y/%m/%d %H:%M:%S')
    plot_dist_time(args.probe[0], date)
    plt.show()
