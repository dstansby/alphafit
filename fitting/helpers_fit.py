# Helper methods for distribution fitting
#
# David Stansby 2017
from datetime import datetime, timedelta, time

from scipy.integrate import simps
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const


def manual_nbeam(df, bimax_proton, params):
    """
    Do a manual numerical integration to estimate the proton beam number
    density.

    bimax_proton : BiMax
        Proton bi-Maxwellian.
    """
    # Hardcode dtheta and dphi
    # phis = np.unique(df['phi'].values)
    # dphis = np.sort(np.diff(phis))
    dphi = 9.81743247e-02
    # assert np.allclose(dphis[-1], dphi)

    # thetas = np.unique(df['theta'].values)
    # dthetas = np.sort(np.diff(thetas))
    dtheta = 8.88372572e-02
    # assert np.allclose(dthetas[-1], dtheta)

    # Convet from SI to (km/s)^{-1} cm^{-3}
    pdf = df['pdf'] * 1e3
    vs = df['|v|'].groupby(level='E_bin').apply(np.median) / 1e3
    # Multiply each point by the integration element
    dist_1d = pdf * np.cos(df['theta']) * (df['|v|'] / 1e3)**2
    # Integrate (ie. sum)
    dist_1d = dist_1d.groupby(level=['E_bin']).sum() * dtheta * dphi

    # Extract velocities of the 1D distribution
    vs = df['|v|'].groupby(level='E_bin').apply(np.median) / 1e3

    # Calculate the bimaxwellian sampled at the same 1D speeds
    bimax_sampled = bimax_proton.integrated_1D(params, vs.values)

    if dist_1d.size < 1:
        return np.nan
    ratio = dist_1d / bimax_sampled.values
    keep = (ratio > 1) & (dist_1d.index > dist_1d.idxmax())
    n_beam = np.trapz((dist_1d - bimax_sampled.values) * keep, x=vs)

    """
    fig, axs = plt.subplots(nrows=2, sharex=True)
    ax = axs[0]
    ax.plot(vs, estimate, lw=1, marker='o')
    ax.plot(vs, bimax_sampled, lw=1, marker='o')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-3)

    ax = axs[1]
    print(estimate.shape, bimax_sampled.shape)
    ax.plot(vs, np.abs(estimate / bimax_sampled.values), lw=1, marker='o')
    ax.axhline(1, color='k', lw=1, ls='-')
    ax.axhline(2, color='k', lw=1, ls='--')
    ax.set_yscale('log')
    ax.set_ylim(top=1e4)
    print(np.trapz(estimate - bimax_sampled.values, vs))
    plt.show()
    """

    return n_beam


def distribution_function_correction(vs, df, moverq):
    '''
    Correct a distribution function for different mass per charge ratios.

    vs :
        Velocities that correspond to proton velocities.
    df :
        Distribution function measurements that assume protons have been
        measured.
    moverq :
        Mass per charge ratio of particles, relative to the mass per charge
        ratio of protons.
    '''
    vs_corrected = vs / np.sqrt(moverq)
    df_corrected = df * np.sqrt(moverq)**4
    return vs_corrected, df_corrected


def _columndotproduct(v1, v2):
    out = np.zeros(v1.shape[0])
    for i in range(v1.shape[0]):
        out[i] = np.dot(v1[int(i), :], v2[int(i), :])
    return out


def process_fitparams(fitparams, species, dist_vs, magempty, params, R,
                      particle_mass=1):
    '''
    Process the output of a bi-Maxwellian fitting routine into a sensible
    dictionary of parameters.

    particle_mass should be as a fraction of proton mass.
    '''
    v = fitparams[3:6]

    # If no magnetic field data, set temperatures to nans
    if magempty:
        fitparams[1:3] = np.nan
    else:
        # Otherwise transform bulk velocity back into spacecraft frame
        v = np.dot(R.T, v)
        pass

    # Speed is less than lowest speed in distribution function
    # if np.linalg.norm(v) < np.min(np.linalg.norm(dist_vs, axis=1)):
    #     return 4

    fit_dict = {}
    fit_dict['T' + species + '_perp'] =\
        vth2temp(np.abs(fitparams[1]), particle_mass)
    fit_dict['T' + species + '_par'] =\
        vth2temp(np.abs(fitparams[2]), particle_mass)
    # Original distribution has units s**3 / m**6
    # Get n_p in 1 / m**3
    n = (fitparams[0] * np.power(np.pi, 1.5) *
         np.abs(fitparams[1]) * 1e3 *
         np.abs(fitparams[1]) * 1e3 *
         np.abs(fitparams[2]) * 1e3) * 1e-6
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


def dist_cut(dist3D, velocity, keep='above'):
    """
    Returns the portion of dist3D that has speeds >= velocity.

    Parameters
    ----------
    keep : {'above', 'below'}
        Whether to keep above or below the *velocity* threshold.
    """
    dist3D = dist3D.copy()
    if keep == 'above':
        tokeep = dist3D['|v|'] / 1e3 >= velocity
        return dist3D.loc[dist3D['|v|'] / 1e3 >= velocity]
    elif keep == 'below':
        tokeep = dist3D['|v|'] / 1e3 <= velocity
    else:
        raise ValueError

    return dist3D.loc[tokeep]


def maxwellian_1D(v, n, v0, v_th):
    prefactor = n * 4 * np.pi * v**2 * np.power(np.pi * v_th**2, -1.5)
    return prefactor * np.exp(-((v - v0) / v_th)**2)


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


mp = const.m_p.value
kB = const.k_B.value


def vth2temp(vth, m=1):
    """
    Assumes velocities are floating point numbers in km/s.
    m is in fractions of proton mass.
    Returns tempearture in degrees kelvin
    """
    return (mp * m * ((vth * 1e3)**2) /
            (2 * kB))


def temp2vth(temp, m=1):
    """
    Assumes velocities are floating point numbers in degrees Kelvin.
    m is in fractions of proton mass.
    Returns thermal speed in km/s
    """
    return np.sqrt(2 * const.k_B * temp * u.K /
                   const.m_p).to(u.km / u.s).value


def rotationmatrixangle(axis, theta):
    """
    Return the rotation matrix about a given axis.

    The rotation is taken to be counterclockwise about the given axis. Uses the
    Euler-Rodrigues formula.

    Parameters
    ----------
        axis : array_like
            Axis to rotate about.
        theta : float
            Angle through which to rotate in radians.

    Returns
    -------
        R : array_like
            Rotation matrix resulting from rotation about given axis.
    """
    assert axis.shape == (3, ), 'Axis must be a single 3 vector'
    assert np.dot(axis, axis) != 0, 'Axis has zero length'

    normaxis = axis / (np.sqrt(np.dot(axis, axis)))

    a = np.cos(theta / 2)
    b, c, d = -normaxis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    out = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return out[:, :, 0]


def angle(v1, v2):
    """
    Return angle between vectors v1 and v2 in radians.

    `n` is the number of components each vector has, and `m` is the number of
    vectors.

    Parameters
    ----------
        v1 : array_like
            Vector 1. Can be shape `(n, )` or shape `(m, n)`.
        v2: array_like
            Vector 2. Can be shape `(n, )` or shape `(m, n)`.

    Returns
    -------
        phi : array_like or float
            Angle between two vectors in radians. Shape will be `(m, )`.
    """
    def ncomps(v):
        """Work out how many components a vector has, and make v 2d"""
        if len(v.shape) == 1:
            n = v.shape[0]
        elif len(v.shape) == 2:
            n = v.shape[1]
        else:
            raise ValueError('Input array must be 1D or 2D, but is %sD'
                             % (len(v.shape)))
        return n, np.atleast_2d(np.array(v))

    v1comps, v1 = ncomps(v1)
    v2comps, v2 = ncomps(v2)
    if v1.shape != v2.shape:
        if v1.shape[0] == 1 and v2.shape[0] != 1:
            v1 = np.repeat(v1, v2.shape[0], axis=0)
        elif v1.shape[0] != 1 and v2.shape[0] == 1:
            v2 = np.repeat(v2, v1.shape[0], axis=0)
        assert v1comps == v2comps,\
            'Input vectors must have same nubmer of components'

    v1mag = np.linalg.norm(v1, axis=1)
    v2mag = np.linalg.norm(v2, axis=1)
    v1dotv2 = _columndotproduct(v1, v2)

    phi = np.arccos(v1dotv2 / (v1mag * v2mag))
    return phi


def rotationmatrix(v):
    """
    Returns the rotation matrix that maps the input vector on to the z-axis.

    Parameters
    ----------
        v : array_like
            Input vector.

    Returns
    -------
        R : array_like
            Rotation matrix.
    """
    assert v.shape == (3, ), "Input must be a 3 component vector"
    v = np.float64(v)
    zaxis = np.array([0, 0, 1])
    if np.array_equal(v, zaxis):
        return np.ma.identity(3)

    # Calculate orthogonal axis
    orthvec = np.cross(zaxis, v)
    phi = angle(v, zaxis)

    R = rotationmatrixangle(orthvec, -phi)

    newzaxis = np.dot(R, v)
    newzaxis = newzaxis / np.linalg.norm(newzaxis)

    return R


def doy2dtime(y, doy):
    return datetime.strptime(str(y) + '-' + str(doy).zfill(3), '%Y-%j')


def dtime2ydoy(dtime):
    return dtime.year, int(dtime.strftime('%j'))


def doy2stime_etime(y, doy):
    '''
    Return start and end of day in datetime form.
    '''
    starttime = doy2dtime(y, doy)
    endtime = starttime + timedelta(hours=24) -\
        timedelta(microseconds=1)
    return starttime, endtime


def read_intervals(f):
    """
    Read in a text file with intervals and return a list of tuples giving all
    the ``(probe, year, doy)`` days of the year contained within the intervals.
    """
    df = pd.read_csv(f, parse_dates=[1, 2])
    out = []
    for i, row in df.iterrows():
        stime = row['Start']
        etime = row['End']
        stime.time = time.min
        etime.time = time.min
        while stime < etime:
            out.append([str(row['Probe']), *dtime2ydoy(stime)])
            stime += timedelta(days=1)
        out.append([str(row['Probe']), *dtime2ydoy(stime)])
    return out
