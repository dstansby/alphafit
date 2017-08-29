# Compare merged data set with our corefit data set
#
# David Stansby 2017
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from heliopy.data import helios

import helpers

# Set probe and dates to compare here
probe = '1'
starttime = datetime(1974, 12, 1, 0, 0, 0)
endtime = starttime + timedelta(days=31)


def plot_status(ax, params, key):
    ax.plot(params[key], marker='o', linewidth=0)


params_3D = helpers.load_corefit(probe, starttime, endtime)
merged = helios.merged(probe, starttime, endtime,
                       try_download=False, verbose=False)

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(merged['Tp1'], label='Merged T')
axs[0].plot(params_3D['Tp_perp'], label=r'3D $T_{\perp}$')
axs[0].plot(params_3D['Tp_par'], label=r'3D $T_{\parallel}$')
axs[0].legend()
axs[0].set_yscale('log')

axs[1].plot(merged['Tp1'], label='Merged T')
axs[1].plot((2 * params_3D['Tp_perp'] + params_3D['Tp_par']) / 3,
            label='3D T total')
axs[1].set_yscale('log')
axs[1].legend()

axs[2].plot(params_3D['Tp_perp'] / params_3D['Tp_par'],
            label=r'$T_{\perp} / T_{\parallel}$')
axs[2].legend()
axs[2].set_yscale('log')

# Velocity
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(merged['vp1r'], label=r'Merged $v_{r}$')
axs[0].plot(params_3D['vp_x'], label=r'3D $v_{r}$', marker='+', markersize=1)

axs[1].plot(merged['vp1t'], label=r'Merged $v_{t}$')
axs[1].plot(params_3D['vp_y'], label=r'3D $v_{t}$')

axs[2].plot(-merged['vp1n'], label=r'Merged $v_{n}$')
axs[2].plot(params_3D['vp_z'], label=r'3D $v_{n}$')
# plot_status(axs[3], params_3D, 'Status')
for i in range(0, 3):
    axs[i].set_ylabel('km/s')
    axs[i].legend()

fig, axs = plt.subplots(2, 1, sharex=True)
# axs = [axs]
axs[0].plot(merged['np1'], label=r'Merged $n_{p}$')
axs[0].plot(params_3D['n_p'], label=r'3D $n_{p}$')
axs[0].set_yscale('log')
plot_status(axs[1], params_3D, 'Status')
axs[0].legend()
axs[0].set_ylabel('cm' + r'$^{-3}$')


def scatter(x, y, ax, xlabel, ylabel):
    minval = min(np.min(x), np.min(y))
    maxval = max(np.max(x), np.max(x))
    ax.scatter(x, y, color='k', s=1)
    ax.plot((minval, maxval), (minval, maxval), linewidth=1, color='r')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# Scatter plots
merge = pd.concat([merged, params_3D], axis=1)

# Number density and velocity
fig, axs = plt.subplots(2, 2)
scatter(merge['np1'], merge['n_p'], axs[0, 0],
        r'Merged $n_{p}$', r'3D $n_{p}$')
scatter(merge['vp1r'], merge['vp_x'], axs[0, 1],
        r'Merged $v_{r}$', r'3D $v_{r}$')
scatter(merge['vp1t'], merge['vp_y'], axs[1, 0],
        r'Merged $v_{t}$', r'3D $v_{t}$')
scatter(-merge['vp1n'], merge['vp_z'], axs[1, 1],
        r'Merged $v_{n}$', r'3D $v_{n}$')
fig.tight_layout()

# Temperature
fig, axs = plt.subplots(3, 1, sharex=True)
scatter(merge['Tp1'], merge['Tp_perp'], axs[0],
        'Merged T', r'3D $T_{\perp}$')
scatter(merge['Tp1'], merge['Tp_par'], axs[1],
        'Merged T', r'3D $T_{\parallel}$')
scatter(merge['Tp1'],
        (merge['Tp_par'] + 2 * merge['Tp_perp']) / 3, axs[2],
        'Merged T', '3D T')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[2].set_xscale('log')
axs[2].set_yscale('log')

fig.tight_layout()
ndists = params_3D.shape[0]


def summarystr(code, string):
    n = (params_3D['Status'] == code).sum()
    return (str(n) + '\t' + string + '\t' +
            '(' + str(100 * n / ndists)[:4] + '%)')


ngoodfits = (params_3D['Status'] == 1).sum()
nnomag = (params_3D['Status'] == 2).sum()
print('\n')
print('Summary')
print('=======')
print(ndists, 'total points')
print('\n')
print(summarystr(1, 'successful fits'))
print(summarystr(2, 'no B data'))
print(summarystr(5, '< 6 points'))
print(summarystr(6, 'fitting failed'))
print(summarystr(9, 'corrupted file'))
print(summarystr(10, 'no proton peak'))
print(summarystr(11, 'n overestimated'))
print(summarystr(12, 'anglular bins'))

plt.show()
