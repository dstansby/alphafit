
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sys
sys.path.append('../fitting')
sys.path.append('../visualising')
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import pandas as pd
import numpy as np
from plasmapy.physics import dimensionless
import astropy.units as u
import astropy.constants as const

from heliopy.data import helios

import vis_helpers as helpers
from helpers import mplhelp
from helpers import helioshelp
from plot_fitted_dist_alphas import plot_dist_time


# In[17]:


# Set probe and dates to compare here
probe = '2'
starttime = datetime(1976, 1, 1, 0, 0, 0)
endtime = starttime + timedelta(days=366)

alphas = helpers.load_alphafit(probe, starttime, endtime)
protons = helios.corefit(probe, starttime, endtime, try_download=False)
protons = protons.reindex(alphas.index)
protons = helioshelp.calculate_derived(protons)


# In[18]:


alphas[['Bx', 'By', 'Bz', '|B|']] = protons[['Bx', 'By', 'Bz', '|B|']]
alphas['Beta_par'] = alphas['Ta_par'] * const.k_B.value * alphas['n_a'] * 1e6 / (alphas['|B|']**2 * 1e-18 / (2 * const.mu0))
alphas['Tani'] = alphas['Ta_perp'] / alphas['Ta_par']
alphas['T_tot'] = (2 * alphas['Ta_perp'] + alphas['Ta_par']) / 2


# In[23]:


keep = protons['vp_x'] > 500
alphas = alphas.loc[keep]
protons = protons.loc[keep]


# Radial distance plots
# ---

# In[24]:


fig, axs = plt.subplots(nrows=4, sharex=True)
kwargs = dict(marker='o', markersize=1, linewidth=0)

ax = axs[0]
ax.plot(protons['r_sun'], alphas['n_a'], **kwargs)
ax.set_yscale('log')

ax = axs[1]
ax.plot(protons['r_sun'], alphas['Ta_par'], **kwargs)
ax.plot(protons['r_sun'], alphas['Ta_perp'], **kwargs)
ax.set_yscale('log')

ax = axs[2]
ax.plot(protons['r_sun'], alphas['Ta_perp'] / alphas['Ta_par'], **kwargs)
ax.set_yscale('log')
ax.set_ylim(1e-2, 1e2)

ax = axs[3]
ax.plot(protons['r_sun'], alphas['Ta_perp'] / protons['Tp_perp'], **kwargs)
ax.plot(protons['r_sun'], alphas['Ta_par'] / protons['Tp_par'], **kwargs)
ax.set_yscale('log')


# Temperatures plots
# ---

# In[25]:


fig, axs = plt.subplots(nrows=7, figsize=(8, 16))
far = protons['r_sun'] > 0.9
close = protons['r_sun'] < 0.4

linekwargs = dict(color='k', linewidth=1)
kwargs = dict(histtype='step')
lw = 1
for loc, s in zip((far, close), ('(1 AU)', '(0.3 AU)')):
    a = alphas.loc[loc]
    p = protons.loc[loc]
    
    ax = axs[0]
    ax.hist((a['n_a'] / p['n_p']).dropna().values, bins=np.logspace(-2, 1, 100),
            label=r'$n_{\alpha} / n_{p}$' + s, **kwargs)
    ax.set_xscale('log')
    
    ax = axs[1]
    ax.hist((a['T_tot'] / p['Tp_tot']).dropna().values, bins=np.logspace(0, 2, 100),
            label=r'$T_{\alpha} / T_{p}$' + s, **kwargs)
    ax.set_xscale('log')

    ax = axs[2]
    Tbins = np.logspace(4, 8, 200)
    ax.hist(a['Ta_perp'].dropna().values, bins=Tbins,
            label=r'$T_{\alpha \perp}$' + s, color='C0', linewidth=lw, **kwargs)
    ax.hist(p['Tp_perp'].dropna().values, bins=Tbins,
            label=r'$T_{p \perp}$' + s, color='C1', linewidth=lw, **kwargs)
    ax.set_xscale('log')
    ax.set_xlabel(r'$^{\circ}$K')

    ax = axs[3]
    ax.hist(a['Ta_par'].dropna().values, bins=Tbins,
            label=r'$T_{\alpha \parallel}$' + s, color='C0', linewidth=lw, **kwargs)
    ax.hist(p['Tp_par'].dropna().values, bins=Tbins,
            label=r'$T_{p \parallel}$' + s, color='C1', linewidth=lw, **kwargs)
    ax.set_xscale('log')
    ax.set_xlabel(r'$^{\circ}$K')

    ax = axs[4]
    ax.hist((a['Ta_par'] / p['Tp_par']).dropna(), bins=np.logspace(-1, 2, 100),
            label=r'$T_{\alpha \parallel} / T_{p \parallel}$' + s, **kwargs)
    ax.set_xscale('log')
    ax.axvline(1, **linekwargs)
    ax.axvline(4, **linekwargs)

    ax = axs[5]
    ax.hist((a['Ta_perp'] / p['Tp_perp']).dropna(), bins=np.logspace(-1, 2, 100),
            label=r'$T_{\alpha \perp} / T_{p \perp}$' + s, **kwargs)
    ax.set_xscale('log')
    ax.axvline(1, **linekwargs)
    ax.axvline(4, **linekwargs)

    ax = axs[6]
    ax.hist((a['Ta_perp'] / a['Ta_par']).dropna(), bins=np.logspace(-1, 1, 100),
            label=r'$T_{\alpha \perp} / T_{a \parallel}$' + s, **kwargs)
    ax.set_xscale('log')
    ax.axvline(1, **linekwargs)
    lw += 1

for ax in axs:
    ax.legend()
fig.tight_layout()
# fig.savefig('alpha_temps.pdf', bbox_inches='tight')


# In[26]:


kwargs = dict(marker='o', markersize=0.1, linewidth=0)


fig, ax = plt.subplots()
ax.plot(protons['r_sun'], alphas['n_a'] / protons['n_p'], **kwargs)
ax.set_ylim(1e-2, 1e2)
ax.set_yscale('log')
ax.grid()


# In[27]:


kwargs = dict(marker='o', markersize=1, linewidth=0)
fig, ax = plt.subplots()
for loc, s, color in zip((far, close), ('1 AU', '0.3 AU'), ('C0', 'C3')):
    a = alphas.loc[loc]
    toplot = a[['Beta_par', 'Tani']].dropna()
    hist, xbins, ybins, = np.histogram2d(a['Beta_par'], a['Tani'],
                                    bins=(np.logspace(-3, 3, 50), np.logspace(-2, 1, 50)))
    xcentres = (xbins[:-1] + xbins[1:]) / 2
    ycentres = (ybins[:-1] + ybins[1:]) / 2
    hist = hist / np.nanmax(hist)
    ax.contour(xcentres, ycentres, hist.T, colors=color, levels=np.logspace(-1, 0, 10))
    ax.plot([0], [0], color=color, label=s)
ax.set_xscale('log')
ax.set_yscale('log')
ax.axhline(1, **linekwargs)
ax.axvline(1, **linekwargs)
ax.set_ylabel(r'$T_{\alpha \perp} / T_{\alpha \parallel}$')
ax.set_xlabel(r'$\beta_{\alpha \parallel}$')
ax.legend()
# fig.savefig('alpha_ani_beta.pdf'.format(s))


# In[82]:


fig, ax = plt.subplots()
ax.scatter(alphas['Ta_par'], alphas['Ta_perp'], s=0.1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e4, 1e8)
ax.set_ylim(1e4, 1e8)


# In[60]:


tokeep = (protons['vp_x'] > 0) & (protons['r_sun'] < 0.4)
a = alphas.loc[tokeep]
p = protons.loc[tokeep]

fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
fig.subplots_adjust(hspace=0, wspace=0)

for i, comp in enumerate(['x', 'y', 'z']):
    ax = axs[0, i]
    ax.scatter(p['B' + comp], a['Ta_par'], s=0.1)
    ax.set_yscale('log')
    ax.set_ylim(1e4, 1e9)

    ax = axs[1, i]
    ax.scatter(p['B' + comp], a['Ta_perp'], s=0.1)
    ax.set_yscale('log')
    
    axs[1, i].set_xlabel('B' + comp)

axs[0, 0].set_ylabel(r'$T_{\alpha \parallel}$')
axs[1, 0].set_ylabel(r'$T_{\alpha \perp}$')


# In[79]:


protons['Bphi'] = np.rad2deg(np.arctan2(protons['By'], protons['Bx']))
protons['Btheta'] = np.rad2deg(np.arcsin(protons['Bz'] / protons['|B|']))
tokeep = (protons['vp_x'] > 0) & (protons['r_sun'] < 1)
a = alphas.loc[tokeep]
p = protons.loc[tokeep]

fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(14, 8))
for i, temp in enumerate(['Ta_par', 'Ta_perp']):
    for j, angle in enumerate(['theta', 'phi']):
        ax = axs[i, j]
        ax.scatter(p['B' + angle], a[temp], s=0.1)
        ax.set_yscale('log')
        ax.set_ylim(1e5, 1e8)

for ax in axs[:, 0]:
    ax.set_xlim(-90, 90)
    ax.set_xlabel(r'$B_{\theta}$')
for ax in axs[:, 1]:
    ax.set_xlim(-180, 180)
    ax.set_xlabel(r'$B_{\phi}$')
for ax in axs[0, :]:
    ax.set_ylabel(r'$T_{\alpha \parallel}$')
for ax in axs[1, :]:
    ax.set_ylabel(r'$T_{\alpha \perp}$')

