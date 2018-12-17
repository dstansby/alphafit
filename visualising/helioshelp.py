from datetime import datetime
import multiprocessing

import numpy as np
import pandas as pd

import astropy.constants as const
import astropy.units as u
from heliopy.data import helios


def p_mag(B):
    return (B * 1e-9)**2 / (2 * const.mu0.value)


def p_th(n, T):
    return n * 1e6 * const.k_B.value * T


def beta(n, T, B):
    return p_th(n, T) / p_mag(B)


def mass_flux(n, vr, r):
    return n * vr * r**2


def calculate_derived(data):
    data = data.copy()

    # Calculate plasma stuff
    data['|B|'] = np.linalg.norm(data[['Bx', 'By', 'Bz']].values, axis=1)
    data['|v|'] = np.linalg.norm(data[['vp_x', 'vp_y', 'vp_z']].values, axis=1)

    # Calculate pressures
    data['Tp_tot'] = (2 * data['Tp_perp'] + data['Tp_par']) / 3
    data['p_mag'] = p_mag(data['|B|'])
    data['p_th_par'] = p_th(data['n_p'], data['Tp_par'])
    data['p_th_tot'] = p_th(data['n_p'], data['Tp_tot'])
    data['Beta'] = (data['p_th_par'] / data['p_mag'])
    data['Beta_tot'] = (data['p_th_tot'] / data['p_mag'])
    data['Tani'] = data['Tp_perp'] / data['Tp_par']
    data['Tp_tot'] = (2 * data['Tp_perp'] + data['Tp_par']) / 3
    # Number density compensated for radial expansion
    data['n_p_norm'] = data['n_p'] * data['r_sun']**2
    data['mass_flux'] = mass_flux(data['n_p'].values * u.cm**-3,
                                     data['vp_x'].values * u.km / u.s,
                                     data['r_sun'].values * const.au).to(1 / u.s).value
    # Specific entropy
    data['Entropy'] = data['Tp_tot'] / data['n_p']**0.5
    for comp in ['x', 'y', 'z']:
        data['va_' + comp] = (data['B' + comp] * 1e-9 * 1e-3 /
                                 np.sqrt(const.m_p.value * data['n_p'] *
                                         1e6 * const.mu0.value))
    data['|va|'] = np.linalg.norm(data[['va_x', 'va_y', 'va_z']].values, axis=1)
    print('New keys:\n', data.keys())
    return data
