import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta as dt

import heliopy.plot.particles as partplt
import heliopy.vector.transformations as heliotrans
import heliopy.data.helios as helios


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


def plot_dist_time(probe, time):
    # Calls plot_dist with a given time. Uses already processed values
    corefit = helios.ion_fitparams_3D(probe, time - dt(seconds=20),
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


def plot_dist(time, dist, params, output, I1a, I1b):
    magempty = np.any(output[['Bx', 'By', 'Bz']] == np.nan)
    R = heliotrans.rotationmatrix(output[['Bx', 'By', 'Bz']].values)

    title = 'Helios 2 ' + str(time)
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].set_title(title)
    vrminlim = 200
    vrmaxlim = 600

    # Calculate reduced out of ecliptic distribution function
    pdf = dist['pdf'].groupby(level=['E_bin', 'El']).sum()
    vs = dist['|v|'].groupby(level=['E_bin', 'El']).mean() / 1e3
    theta = dist['theta'].groupby(level=['E_bin', 'El']).mean()
    vr = np.cos(theta) * vs + params['helios_vr']
    vn = np.sin(theta) * vs

    plt.sca(ax[0])
    partplt.contour2d(vr, vn, pdf, levels=20, showbins=True)
    ax[0].scatter(output['vp_x'], output['vp_z'],
                  marker='x', color='r')
    ax[0].set_ylabel(r'$v_{n}$' + ' (km/s)')
    ax[0].set_xlim(vrminlim, vrmaxlim)
    ax[0].set_aspect('equal', 'datalim')

    # Calculate reduced ecliptic distribution function
    pdf = dist['pdf'].groupby(level=['E_bin', 'Az']).sum()
    vs = dist['|v|'].groupby(level=['E_bin', 'Az']).mean() / 1e3
    phi = dist['phi'].groupby(level=['E_bin', 'Az']).mean()
    vr = np.cos(phi) * vs + params['helios_vr']
    vt = np.sin(phi) * vs + params['helios_v']

    plt.sca(ax[1])
    partplt.contour2d(vr, vt, pdf, levels=20, showbins=True)
    ax[1].scatter(output['vp_x'], output['vp_y'],
                  marker='x', color='r')
    ax[1].set_ylabel(r'$v_{t}$' + ' (km/s)')
    ax[1].set_xlim(vrminlim, vrmaxlim)
    ax[1].set_aspect('equal', 'datalim')

    # Calculate 1D reduced distribution from data
    vs = dist['|v|'].groupby(level=['E_bin']).mean() / 1e3
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
        vx, vy, vz = heliotrans.sph2cart(modvs, thetas, phis)
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

        vperp, vpar = perp_par_vels(dist[['vx', 'vy', 'vz']].values / 1e3,
                                    output[['vp_x', 'vp_y', 'vp_z']], R)
        levels = np.linspace(-5, 0, 20)
        fig, axs2 = plt.subplots(2, 1, sharex=True, sharey=True)
        axs2[0].set_title(title)

        plt.sca(axs2[0])
        partplt.contour2d(vpar, vperp,
                          np.concatenate((dist['pdf'].values,
                                          dist['pdf'].values)) / dist['pdf'].max(), levels=levels, showbins=True)
        plt.sca(axs2[1])
        fitted_bimax = perp_par_maxwellian(output['n_p'], output['vth_p_perp'], output['vth_p_par'], vperp, vpar)
        partplt.contour2d(vpar, vperp, fitted_bimax / np.max(fitted_bimax), levels=levels)

        for ax2 in axs2:
            ax2.set_aspect('equal', 'datalim')
            ax2.set_xlabel(r'$v_{\parallel}$')
            ax2.set_ylabel(r'$v_{\perp}$')

    ax[2].legend()
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$v_{r}$' + ' (km/s)')


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
