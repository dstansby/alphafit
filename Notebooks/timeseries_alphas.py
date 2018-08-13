
# coding: utf-8

# In[165]:


get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[154]:


# Set probe and dates to compare here
probe = '2'
starttime = datetime(1976, 1, 17)
endtime = starttime + timedelta(days=100)

merged = helios.merged(probe, starttime, endtime)
alphas = helpers.load_alphafit(probe, starttime, endtime)
protons = helios.corefit(probe, starttime, endtime, try_download=False)


# In[155]:


protons = protons.reindex(alphas.index)
protons = helioshelp.calculate_derived(protons)

alphas[['Bx', 'By', 'Bz', '|B|']] = protons[['Bx', 'By', 'Bz', '|B|']]
alphas['Beta_par'] = alphas['Ta_par'] * const.k_B.value * alphas['n_a'] * 1e6 / (alphas['|B|']**2 * 1e-18 / (2 * const.mu0))
alphas['Tani'] = alphas['Ta_perp'] / alphas['Ta_par']


# In[156]:


merged.keys()


# Reprodcue Marsch et al. 1982b Figure 1
# ---

# In[157]:


fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(14, 10))

ax = axs[0]
ax.scatter(alphas.index, alphas['va_x'], s=1, alpha=0.5)
ax.set_ylim(200, 900)
ax.set_ylabel(r'$v_{\alpha x}$ (km/s)')

ax = axs[1]
ax.scatter(alphas.index, alphas['n_a'], s=1)
ax.set_yscale('log')
ax.set_ylim(1e-1, 1e1)
ax.set_ylabel(r'$n_{\alpha}$ (cm$^{-3}$)')

ax = axs[2]
ax.scatter(merged.index, merged['Tal'], s=1, c='C1')
ax.scatter(alphas.index, alphas['Ta_par'], s=1, alpha=0.5)

ax.set_yscale('log')
ax.set_ylim(1e4, 1e7)
ax.set_ylabel(r'$T_{\alpha \parallel}$ (K)')
ax.set_xlim(alphas.index.min(), alphas.index.max())


# Extend previous figure
# ---

# In[166]:


akwargs = {'alpha': 0.5, 'color': 'C0', 's': 1}
pkwargs = {'alpha': 0.1, 'color': 'C3', 's': 1}
fig, axs = plt.subplots(nrows=6, sharex=True, figsize=(14, 16))

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
ax.scatter(alphas.index, alphas['Ta_par'] / protons['Tp_par'], s=1, color='C1', alpha=0.5)
ax.scatter(alphas.index, alphas['Ta_perp'] / protons['Tp_perp'], s=1, color='C2', alpha=0.5)

ax.set_yscale('log')
ax.set_ylim(1e0, 1e2)
ax.axhline(4, color='k', alpha=0.5)
# ax.set_ylabel(r'$T_{\alpha \perp}$')

ax = axs[5]
ax.plot(protons['r_sun'], color='k')
ax.set_ylabel('r (AU)')

ax.set_xlim(alphas.index.min(), alphas.index.max())

