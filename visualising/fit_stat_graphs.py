'''
Output stats from helios distribution fitting
'''
import matplotlib.pyplot as plt
import heliopy.data.helios as helios
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np

fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
for probe in ['1', '2']:
    times = []
    n_fit = []
    n_nomag = []
    n_dists = []
    maxpoints = []

    for year in range(1974, 1985):
        for month in range(1, 12):
            starttime = datetime(year, month, 1, 0, 0, 0)
            if month == 12:
                year += 1
                month = 0
            endtime = datetime(year, month + 1, 1, 0, 0, 0) - timedelta(seconds=1)

            print(starttime)
            ndays = (endtime - starttime).days + 1
            maxpoints.append(ndays * 24 * 60 * 60 / 40.5)
            times.append(starttime + (endtime - starttime) / 2)
            try:
                params_3D = helios.ion_fitparams_3D(probe, starttime, endtime)
            except Exception:
                n_fit.append(0)
                n_nomag.append(0)
                n_dists.append(0)
                continue
            n_dists.append(params_3D.shape[0])
            n_fit.append((params_3D['Status'] == 1).sum())
            n_nomag.append((params_3D['Status'] == 2).sum())

    ax1.plot(times, (np.array(n_fit) + np.array(n_nomag)) / np.array(maxpoints), label='Helios ' + probe)
    ax2.plot(times, np.array(n_fit) / np.array(n_dists), label='Helios ' + probe)
    # ax2.plot(times, np.array(n_nomag) / np.array(n_dists), label='Helios ' + probe)

ax1.set_ylabel(r'fraction of all possible distributions fitted')
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax1.legend()
plt.show()
