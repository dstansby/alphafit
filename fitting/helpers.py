# Helper methods for distribution fitting
#
# David Stansby 2017
from datetime import datetime

import numpy as np
import astropy.units as u
import astropy.constants as const


def _columndotproduct(v1, v2):
    out = np.zeros(v1.shape[0])
    for i in range(v1.shape[0]):
        out[i] = np.dot(v1[int(i), :], v2[int(i), :])
    return out


def vth2temp(vth):
    """
    Assumes velocities are floating point numbers in km/s.
    """
    return (const.m_p * ((float(vth) * (u.km / u.s))**2) /
            (2 * const.k_B)).to(u.K).value


def temp2vth(temp):
    """
    Assumes velocities are floating point numbers in km/s.
    """
    return np.sqrt(2 * const.k_B * float(temp) * u.K /
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
