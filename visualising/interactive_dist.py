# Script for interactively plotting an experimentally measured Helios
# velocity distribution function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
from heliopy.data import helios


class Distribution:
    def __init__(self, df):
        self.df = df
        self.index = df.index

    @property
    def vx(self):
        return self.df['vx'].values / 1e3

    @property
    def vy(self):
        return self.df['vy'].values / 1e3

    @property
    def vz(self):
        return self.df['vz'].values / 1e3

    @property
    def pdf(self):
        return self.df['pdf'].values

    @property
    def az_levels(self):
        return np.unique(self.index.get_level_values('Az'))

    def elevation_slice(self, n):
        '''
        Returns a VDF slice along the elevation bin given by *n*.
        '''

        return self.df.loc[self.index.get_level_values('El') == n]

    def azimuth_slice(self, n):
        '''
        Returns a VDF slice along the azimiuthal bin given by *n*.
        '''
        return self.df.loc[self.index.get_level_values('Az') == n]

    def plot_az_slice(self, n, ax):
        data = self.azimuth_slice(n)

        pdf = data['pdf'].values
        theta = data['theta'].values
        v = data['|v|'].values / 1e3

        # Transform to cartesian coordinates
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)

        levels = np.logspace(np.log10(np.max(self.pdf)) - 3,
                             np.log10(np.max(self.pdf)), 10)
        norm = mcolor.LogNorm()
        ax.tricontourf(vx, vy, pdf, levels=levels, norm=norm)
        ax.tricontour(vx, vy, pdf, levels=levels,
                      colors='k', linewidths=1, linestyles='-',
                      )
        ax.scatter(vx, vy, color='k', s=10, edgecolor='w')

        levels = self.az_levels

        for i, level in enumerate(levels):
            t = ax.text(0.7 + 0.05 * i, 1.05, str(level),
                        transform=ax.transAxes)
            if level == n:
                t.set_weight('bold')

        # Fix axes limits
        ax.set_xlim(np.min(self.vx), np.max(self.vx))
        ax.set_ylim(np.min(self.vz), np.max(self.vz))


class SlicePlotter:
    def __init__(self, df):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.df = df
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self)

        az_levels = self.df.az_levels
        self.az_slice = az_levels[az_levels.size // 2]
        self.df.plot_az_slice(self.az_slice, ax)
        # fig.colorbar()

    def __call__(self, event):
        key = event.key
        if key not in ('left', 'right'):
            return

        if event.key == 'left':
            if self.az_slice > np.min(self.df.az_levels):
                self.az_slice -= 1
        elif event.key == 'right':
            if self.az_slice < np.max(self.df.az_levels):
                self.az_slice += 1

        self.ax.cla()
        df.plot_az_slice(self.az_slice, self.ax)
        self.fig.canvas.draw()


if __name__ == '__main__':
    probe = '1'
    year = 1975
    doy = 87
    hour = 0
    minute = 0
    second = 8
    df = helios.ion_dist_single(probe, year, doy, hour, minute, second,
                                remove_advect=False)
    df = Distribution(df)
    plotter = SlicePlotter(df)
    plt.show()
