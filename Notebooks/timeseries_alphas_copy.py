import sys
sys.path.append('../fitting')
sys.path.append('../visualising')
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as const

from heliopy.data import helios

import vis_helpers as helpers
from helpers import mplhelp
from helpers import helioshelp
from plot_fitted_dist_alphas import plot_dist_time

probe = '2'
starttime = datetime(1976, 4, 14)
endtime = starttime + timedelta(days=10)

alphas = helpers.load_alphafit(probe, starttime, endtime)
protons = helios.corefit(probe, starttime, endtime, try_download=False)
protons = protons.reindex(alphas.index)
protons = helioshelp.calculate_derived(protons)
protons['Bphi'] = np.rad2deg(np.arctan2(protons['By'], protons['Bx']))
protons.loc[protons['Bphi'] < 0, 'Bphi'] += 360
protons['Btheta'] = np.rad2deg(np.arcsin(protons['Bz'] / protons['|B|']))

alphas[['Bx', 'By', 'Bz', '|B|']] = protons[['Bx', 'By', 'Bz', '|B|']]
alphas['Beta_par'] = alphas['Ta_par'] * const.k_B.value * alphas['n_a'] * 1e6 / (alphas['|B|']**2 * 1e-18 / (2 * const.mu0))
alphas['Tani'] = alphas['Ta_perp'] / alphas['Ta_par']

akwargs = {'alpha': 1, 'color': 'C0', 's': 1}
pkwargs = {'alpha': 0.5, 'color': 'C3', 's': 1}
fig, axs = plt.subplots(nrows=6, sharex=True, figsize=(10, 12))

ax = axs[0]
ax.scatter(alphas.index, alphas['va_x'], **akwargs)
ax.scatter(protons.index, protons['vp_x'], **pkwargs)
ax.set_ylim(200, 900)
ax.set_ylabel(r'$v_{x}$ (km/s)')

ax = axs[1]
ax.scatter(alphas.index, alphas['n_a'], **akwargs)
ax.scatter(protons.index, protons['n_p'], **pkwargs)
ax.set_yscale('log')
ax.set_ylim(1e-2, 1e2)
ax.set_ylabel(r'$n$ (cm$^{-3}$)')

ax = axs[2]
ax.scatter(alphas.index, alphas['Ta_par'], **akwargs)
ax.scatter(protons.index, protons['Tp_par'], **pkwargs)
ax.set_yscale('log')
ax.set_ylim(1e4, 1e7)
ax.set_ylabel(r'$T_{\parallel}$ (K)')

ax = axs[3]
ax.scatter(alphas.index, alphas['Ta_perp'], **akwargs)
ax.scatter(protons.index, protons['Tp_perp'], **pkwargs)
ax.set_yscale('log')
ax.set_ylim(1e4, 1e7)
ax.set_ylabel(r'$T_{\perp}$ (K)')

ax = axs[4]
ax.scatter(alphas.index, alphas['Ta_par'] / protons['Tp_par'], s=1, color='C1', label=r'$\parallel$')
ax.scatter(alphas.index, alphas['Ta_perp'] / protons['Tp_perp'], s=1, color='C2', label=r'$\perp$')

ax.legend()
ax.set_yscale('log')
ax.set_ylim(1e0, 1e2)
ax.axhline(4, color='k', alpha=0.5)
# ax.set_ylabel(r'$T_{\alpha \perp}$')

ax = axs[5]
ax.plot(protons['r_sun'], color='k')
ax.set_ylabel('r (AU)')

ax.set_xlim(alphas.index.min(), alphas.index.max())

fig, axs = plt.subplots(ncols=2, sharey=True)
ax = axs[0]
ax.scatter(protons['Bphi'], alphas['Ta_par'], s=1, label=r'$T_{\alpha \parallel}$', alpha=0.5)
ax.scatter(protons['Bphi'], alphas['Ta_perp'], s=1, label=r'$T_{\alpha \perp}$', alpha=0.5)
ax.set_yscale('log')
ax.legend()

ax = axs[1]
ax.scatter(protons['Btheta'], alphas['Ta_par'], s=1, alpha=0.1)
ax.scatter(protons['Btheta'], alphas['Ta_perp'], s=1, alpha=0.1)

plt.show()
