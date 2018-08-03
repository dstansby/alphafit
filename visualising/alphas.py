from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from heliopy.data import helios

import vis_helpers as helpers
from helpers import mplhelp
from plot_fitted_dist_alphas import plot_dist_time

# Set probe and dates to compare here
probe = '2'
starttime = datetime(1976, 4, 15, 0, 0, 0)
endtime = starttime + timedelta(days=10)

alphas = helpers.load_alphafit(probe, starttime, endtime)
protons = helios.corefit(probe, starttime, endtime)

print('Alpha points:', alphas.shape[0])

fig, axs = plt.subplots(nrows=6, sharex=True, figsize=(6, 10))

ax = axs[0]
ax.plot(protons['vp_x'])
ax.plot(alphas.index, alphas['va_x'])

ax = axs[1]
ax.plot(protons['vp_y'])
ax.plot(alphas['va_y'])

ax = axs[2]
ax.plot(protons['vp_z'])
ax.plot(alphas['va_z'])

ax = axs[3]
ax.plot(protons['n_p'])
ax.plot(alphas['n_a'])
ax.set_yscale('log')

ax = axs[4]
ax.plot(protons['Tp_perp'], color='C0')
ax.plot(alphas['Ta_perp'], color='C3')
ax.plot(protons['Tp_par'], color='C0', alpha=0.5)
ax.plot(alphas['Ta_par'], color='C3', alpha=0.5)
ax.legend()
ax.set_yscale('log')

ax = axs[5]
ax.scatter(alphas.index, alphas['Status'])

fig, ax = plt.subplots()
kwargs = {'s': 1}
ax.scatter(protons['vp_y'], protons['vp_x'], **kwargs)
ax.scatter(alphas['va_y'], alphas['va_x'], **kwargs)

fig, axs = plt.subplots(ncols=2)
histkwargs = dict(histtype='step')
ax = axs[0]
protons['Tani'] = protons['Tp_perp'] / protons['Tp_par']
alphas['Tani'] = alphas['Ta_perp'] / alphas['Ta_par']

bins = np.logspace(-2, 2, 100)
ax.hist(protons['Tani'].dropna(), bins=bins, label='p', **histkwargs)
ax.hist(alphas['Tani'].dropna(), bins=bins, label=r'$\alpha$', **histkwargs)
ax.set_xscale('log')
ax.legend()
ax.set_xlabel(r'$T_{\perp} / T_{\parallel}$')

ax = axs[1]
bins = np.logspace(-2, 1, 20)
ax.hist((alphas['n_a'] / protons['n_p']).dropna(), bins=bins, histtype='step')
ax.set_xscale('log')
ax.set_xlabel(r'$n_{\alpha} / n_{p}$')

plt.show()
