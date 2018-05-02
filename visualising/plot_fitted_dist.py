# Methods to plot fitted distribution functions
#
# David Stansby 2017
from datetime import timedelta as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import heliopy.data.helios as helios

import helpers


def contour2d(x, y, pdf, showbins=True, levels=10):
    """Perform a countour plot of 2D distribution function data."""
    ax = plt.gca()
    pdf = np.log10(pdf)
    if type(levels) == int:
        levels = np.linspace(np.nanmin(pdf), np.nanmax(pdf), levels)
    ax.tricontourf(x, y, pdf, levels=levels, cmap='viridis')
    ax.tricontour(x, y, pdf, levels=levels, linestyles='-', colors='k', linewidths=0.5, alpha=0.8)
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
    corefit = corefit[corefit.index == time]
    if corefit.shape[0] != 1:
        raise ValueError('Could not find fitted parameters at requested time')
    corefit = pd.Series(corefit.iloc[0])
    args = (probe, time.year, int(time.strftime('%j')),
            time.hour, time.minute, time.second)
    dist = helios.ion_dist_single(*args)
    params = helios.distparams_single(*args)
    I1a, I1b = helios.integrated_dists_single(*args)

    plot_dist(time, dist, params, corefit, I1a, I1b)


def slice_dist(vs, pdf, plane):
    vlim = 300
    x, y = np.meshgrid(np.linspace(-vlim, vlim, 100),
                       np.linspace(-vlim, vlim, 100))
    sampling_points = [x, y, y]
    sampling_points[plane] = np.zeros(x.shape)
    xyinterp = interp.LinearNDInterpolator(vs, pdf)
    pdf = xyinterp(np.array(sampling_points).T).T
    x = x.ravel()
    y = y.ravel()
    pdf = pdf.ravel()
    x = x[np.isfinite(pdf)]
    y = y[np.isfinite(pdf)]
    pdf = pdf[np.isfinite(pdf)]
    return x, y, pdf


def plot_dist(time, dist, params, output, I1a, I1b):
    magempty = np.any(output[['Bx', 'By', 'Bz']] == np.nan)
    R = helpers.rotationmatrix(output[['Bx', 'By', 'Bz']].values)

    title = 'Helios 2 ' + str(time)
    fig, ax = plt.subplots(3, 1)
    ax[0].set_title(title)
    vrminlim = 200
    vrmaxlim = 1000

    dist[['vx', 'vy', 'vz', '|v|']] /= 1e3
    dist_vcentre = dist.copy()
    for comp in ['x', 'y', 'z']:
        dist_vcentre['v' + comp] -= output['vp_' + comp]
    print(dist_vcentre.head())
    vs = np.dot(R, dist_vcentre[['vx', 'vy', 'vz']].T).T
    x, y, pdf = slice_dist(vs, dist_vcentre['pdf'], 1)

    plt.sca(ax[0])
    contour2d(x.ravel(), y.ravel(), pdf.ravel(), levels=20, showbins=False)
    ax[0].set_aspect('equal', adjustable='datalim')


    x, y, pdf = slice_dist(vs, dist_vcentre['pdf'], 2)
    plt.sca(ax[1])
    contour2d(x, y, pdf, levels=20, showbins=False)
    '''ax[1].scatter(output['vp_x'], output['vp_y'],
                  marker='x', color='r')
    ax[1].set_ylabel(r'$v_{t}$' + ' (km/s)')
    ax[1].set_xlim(vrminlim, vrmaxlim)'''
    ax[1].set_aspect('equal', 'datalim')

    # Calculate 1D reduced distribution from data
    vs = dist['|v|'].groupby(level=['E_bin']).mean()
    pdf = dist['pdf'].groupby(level=['E_bin']).sum() * vs**2

    ax[2].plot(I1a['v'], I1a['df'] / I1a['df'].max(),
               marker='x', label='1D')
    ax[2].plot(vs, pdf / np.max(pdf),
               marker='+', label='Integrated 3D')
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

        ax[2].plot(df.index.values, df)
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

    ax[2].legend()
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$v_{r}$' + ' (km/s)')


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
