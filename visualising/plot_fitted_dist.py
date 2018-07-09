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

import heliopy.data.helios as helios

import helpers


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


def plot_dist_time(probe, time):
    # Calls plot_dist with a given time. Uses already processed values
    corefit = helpers.load_corefit(probe, time - dt(seconds=20),
                                   time + dt(seconds=20))
    corefit = corefit.loc[corefit.index == time]
    if corefit.shape[0] != 1:
        raise ValueError('Could not find fitted parameters at requested time')
    corefit = pd.Series(corefit.iloc[0])
    args = (probe, time.year, int(time.strftime('%j')),
            time.hour, time.minute, time.second)
    dist = helios.ion_dist_single(*args)
    params = helios.distparams_single(*args)
    I1a, I1b = helios.integrated_dists_single(*args)

    plot_dist(time, probe, dist, params, corefit, I1a, I1b)


def slice_dist(vs, pdf, plane):
    '''
    Get distribution slices. Interpolates on to either the x-y, x-z or y-z
    plane.

    Parameters
    ----------
    plane : int
        0 for y-z, 1 for x-z, 2 for y-z.
    '''
    vlim = 400
    nbins = 200
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
    pdf = interp.griddata(vs, pdf.values, np.array(sampling_points).T,
                          method='linear').T
    dim1 = dim1.ravel()
    dim2 = dim2.ravel()
    pdf = pdf.ravel()
    dim1 = dim1[np.isfinite(pdf)]
    dim2 = dim2[np.isfinite(pdf)]
    pdf = pdf[np.isfinite(pdf)]
    return dim1, dim2, pdf


def plot_dist(time, probe, dist, params, output, I1a, I1b):
    magempty = np.any(~np.isfinite(output[['Bx', 'By', 'Bz']].values))
    if magempty:
        raise RuntimeError('No magnetic field present')
    R = helpers.rotationmatrix(output[['Bx', 'By', 'Bz']].values)

    title = 'Helios {} '.format(probe) + str(time)
    fig = plt.figure(figsize=(6, 6))
    spec = gridspec.GridSpec(ncols=2, nrows=2)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(spec[1, :])
    ax = [ax1, ax2, ax3]

    fig.suptitle(title)
    vrminlim = 200
    vrmaxlim = 1000
    dist[['vx', 'vy', 'vz', '|v|']] /= 1e3
    dist_vcentre = dist.copy()
    for comp in ['x', 'y', 'z']:
        dist_vcentre['v' + comp] -= output['vp_' + comp]
    # Distribution is in spacecraft frame, but output velocities are in
    # solar wind frame, so correct
    dist_vcentre['vx'] += params['helios_vr']
    dist_vcentre['vy'] += params['helios_v']
    vs = np.dot(R, dist_vcentre[['vx', 'vy', 'vz']].values.T).T
    levels = np.linspace(np.log(dist_vcentre['pdf']).min(),
                         np.log(dist_vcentre['pdf']).max(), 20)

    # Slice along B (which is along z-axis)
    x, z, pdf = slice_dist(vs, dist_vcentre['pdf'], 1)
    plt.sca(ax[0])
    contour2d(z, x, pdf, levels=levels, showbins=False, add1overe=True)
    # Plot thermal speeds
    ax[0].plot((0, 0), (-output['vth_p_perp'], output['vth_p_perp']), color='k')
    ax[0].plot((-output['vth_p_par'], output['vth_p_par']), (0, 0), color='k')

    # Slice perp to B
    x, y, pdf = slice_dist(vs, dist_vcentre['pdf'], 2)
    plt.sca(ax[1])
    contour2d(y, x, pdf, levels=levels, showbins=False, add1overe=True)
    ax[1].plot((0, 0), (-output['vth_p_perp'], output['vth_p_perp']),
               color='k')
    ax[1].plot((-output['vth_p_perp'], output['vth_p_perp']), (0, 0),
               color='k')

    # Plot formatting
    ax[0].set_ylabel(r'$v_{\perp ,1}$ (km/s)')
    ax[0].set_xlabel(r'$v_{\parallel}$ (km/s)')
    ax[1].set_xlabel(r'$v_{\perp ,2}$ (km/s)')
    for a in ax[0:2]:
        a.set_aspect('equal', 'datalim')
        a.set_ylim(-400, 400)
    ax[1].tick_params(axis='y', labelleft=False, labelright=True,
                      left=False, right=True)
    trans = ax[0].transAxes
    # Add magnetic field arrows
    arrow = mpatch.FancyArrowPatch((0.6, 0.87), (0.95, 0.87),
                                   arrowstyle='-|>', mutation_scale=20,
                                   facecolor='k', transform=trans)
    ax[0].add_patch(arrow)
    ax[0].text(0.73, 0.9,
               'B', fontsize=14,
               transform=trans)
    Tani = output['Tp_perp'] / output['Tp_par']
    ax[0].text(0.05, 0.05,
               r'$T_{\perp} / T_{\parallel} =$' + '{:.01f}'.format(Tani),
               transform=trans)
    ax[1].text(0.8, 0.9, 'B ⊗', fontsize=14, transform=ax[1].transAxes)
    # ax[1].text(0.02, 0.05,
    #            r'$\mathbf{B}_{rtn}$ = ' +
    #            '({:.01f}, {:.01f}, {:.01f}) nT'.format(output['Bx'],
    #                                                    output['By'],
    #                                                    output['Bz']),
    #            transform=ax[1].transAxes)
    # Calculate 1D reduced distribution from data
    vs = dist['|v|'].groupby(level=['E_bin']).mean()
    pdf = dist['pdf'].groupby(level=['E_bin']).sum() * vs**2

    ax[2].plot(I1a['v'], I1a['df'] / I1a['df'].max(),
               marker='x', label='Data')
    # ax[2].plot(vs, pdf / np.max(pdf),
    #            marker='+', label='Integrated 3D')
    if not magempty:
        # Calculate reduced 3D fit
        phis = np.linspace(-np.pi, np.pi, 100)
        thetas = np.linspace(-np.pi / 2, np.pi / 2, 100)
        modvs = np.arange(vrminlim, vrmaxlim, 5)
        modvs, thetas, phis = np.meshgrid(modvs, thetas, phis)
        modvs = modvs.flatten()
        thetas = thetas.flatten()
        phis = phis.flatten()
        vx, vy, vz = helpers.sph2cart(modvs, thetas, phis)
        v = np.array([vx, vy, vz]).T
        # Transform into rotated frame
        v = np.dot(R, v.T).T

        # Calculate bi-maxwellian parameters in field aligned frame
        A = output['n_p'] * 1e6 / (np.power(np.pi, 1.5) *
                                   output['vth_p_perp'] * 1e3 *
                                   output['vth_p_perp'] * 1e3 *
                                   output['vth_p_par'] * 1e3)
        output['vp_x'] -= params['helios_vr']
        output['vp_y'] -= params['helios_v']
        vbulkBframe = np.dot(R, output[['vp_x', 'vp_y', 'vp_z']].values)
        df = bi_maxwellian_3D(v[:, 0], v[:, 1], v[:, 2],
                              A, output['vth_p_perp'], output['vth_p_par'],
                              *vbulkBframe)
        index = pd.MultiIndex.from_arrays([modvs, thetas, phis],
                                          names=['v', 'theta', 'phi'])
        df *= np.cos(phis) * modvs
        df = pd.DataFrame({'Reduced fit': df}, index=index)
        df = df['Reduced fit'].groupby(level='v').sum()
        df /= df.max()
        df[df < 1e-3] = np.nan

        ax[2].plot(df.index.values, df, label='Fit')

        # Formatting
        ax[2].legend(frameon=False)
        ax[2].set_yscale('log')
        ax[2].set_xlabel(r'$|v|$' + ' (km/s)')
        ax[2].set_ylim(1e-3, 2)
        ax[2].set_xlim(400, 1600)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        vperp, vpar = perp_par_vels(dist[['vx', 'vy', 'vz']].values,
                                    output[['vp_x', 'vp_y', 'vp_z']].values, R)
        levels = np.linspace(-5, 0, 20)
        fig, axs2 = plt.subplots(2, 1, sharex=True, sharey=True)
        axs2[0].set_title(title)

        plt.sca(axs2[0])
        contour2d(
            vpar, vperp,
            np.concatenate((dist['pdf'].values,
                            dist['pdf'].values)) / dist['pdf'].max(),
            levels=levels, showbins=True)
        plt.sca(axs2[1])
        fitted_bimax = perp_par_maxwellian(
            output['n_p'], output['vth_p_perp'], output['vth_p_par'],
            vperp, vpar)
        contour2d(vpar, vperp, fitted_bimax / np.max(fitted_bimax),
                  levels=levels)

        for ax2 in axs2:
            # ax2.set_aspect('equal', 'datalim')
            ax2.set_xlabel(r'$v_{\parallel}$')
            ax2.set_ylabel(r'$v_{\perp}$')


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
