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
protons = helios.corefit(probe, starttime, endtime)

fig, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(alphas['va_x'])
axs[0].plot(protons['vp_x'])

plt.show()
