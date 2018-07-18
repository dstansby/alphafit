from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from heliopy.data import helios

import vis_helpers as helpers
from plot_fitted_dist_alphas import plot_dist_time

# Set probe and dates to compare here
probe = '2'
starttime = datetime(1976, 4, 17, 0, 0, 0)
endtime = starttime + timedelta(days=1)

alphas = helpers.load_alphafit(probe, starttime, endtime)
protons = helios.corefit(probe, starttime, endtime).data

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
ax.plot(protons['Tp_perp'])
ax.plot(alphas['Ta_perp'])
ax.plot(protons['Tp_par'])
ax.plot(alphas['Ta_par'])
ax.legend()
ax.set_yscale('log')

ax = axs[5]
ax.scatter(alphas.index, alphas['Status'])

fig, ax = plt.subplots()
kwargs = {'s': 1}
ax.scatter(protons['vp_y'], protons['vp_x'], **kwargs)
ax.scatter(alphas['va_y'], alphas['va_x'], **kwargs)
plt.show()
