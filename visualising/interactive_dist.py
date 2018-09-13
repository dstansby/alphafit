# Script for interactively plotting an experimentally measured Helios
# velocity distribution function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
from heliopy.data import helios


def get_middle_value(a):
    return a[a.size // 2]


class SlicePlotter:
    def __init__(self, df):
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        self.fig = fig
        self.axs = axs
        self.df = df
        self.index = df.index
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self)

        # Azimuth
        az_levels = self.az_levels
        self.az_slice = get_middle_value(az_levels)
        self.plot_az_slice()

        # Elevation
        el_levels = self.el_levels
        self.el_slice = get_middle_value(el_levels)
        self.plot_el_slice()

    def __call__(self, event):
        key = event.key
        if key not in ('left', 'right', 'up', 'down'):
            return

        if event.key == 'left':
            if self.az_slice > np.min(self.az_levels):
                self.az_slice -= 1
        elif event.key == 'right':
            if self.az_slice < np.max(self.az_levels):
                self.az_slice += 1
        elif event.key == 'up':
            if self.el_slice < np.max(self.el_levels):
                self.el_slice += 1
        elif event.key == 'down':
            if self.el_slice > np.min(self.el_levels):
                self.el_slice -= 1

        for ax in self.axs:
            ax.cla()

        self.plot_az_slice()
        self.plot_el_slice()
        self.fig.canvas.draw()

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
        """
        Unique azimuthal levels.
        """
        return np.unique(self.index.get_level_values('Az'))

    @property
    def el_levels(self):
        """
        Unique elevation levels.
        """
        return np.unique(self.index.get_level_values('El'))

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

    def plot_slice(self, v, angles, pdf, ax):
        # Transform to cartesian coordinates
        vx = v * np.cos(angles)
        vy = v * np.sin(angles)

        levels = np.logspace(np.log10(np.max(self.pdf)) - 3,
                             np.log10(np.max(self.pdf)), 10)
        norm = mcolor.LogNorm()
        ax.tricontourf(vx, vy, pdf, levels=levels,
                       norm=norm, cmap='viridis', alpha=0.8)
        ax.tricontour(vx, vy, pdf, levels=levels,
                      colors='k', linewidths=0.5, linestyles='-',
                      )
        ax.scatter(vx, vy, color='k', s=1, alpha=0.5)

    def add_angle_line(self, angle, ax):
        ax.plot([0, 1e4 * np.cos(angle)],
                [0, 1e4 * np.sin(angle)],
                color='C3')

    def plot_az_slice(self):
        n = self.az_slice
        ax = self.axs[0]
        data = self.azimuth_slice(n)

        pdf = data['pdf'].values
        theta = data['theta'].values
        phi = data['phi'].values[0]
        v = data['|v|'].values / 1e3
        self.plot_slice(v, theta, pdf, ax)
        self.add_angle_line(phi, self.axs[1])

        # Fix axes limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(np.min(self.vx), np.max(self.vx))
        ax.set_ylim(np.min(self.vz), np.max(self.vz))
        ax.set_xlabel(r'$v_{r}$ (km/s)')
        ax.set_ylabel(r'$v_{n}$ (km/s)')

    def plot_el_slice(self):
        n = self.el_slice
        ax = self.axs[1]
        data = self.elevation_slice(n)

        pdf = data['pdf'].values
        phi = data['phi'].values
        theta = data['theta'].values[0]
        v = data['|v|'].values / 1e3
        self.plot_slice(v, phi, pdf, ax)
        self.add_angle_line(theta, self.axs[0])

        # Fix axes limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(np.min(self.vx), np.max(self.vx))
        ax.set_ylim(np.min(self.vy), np.max(self.vy))
        ax.set_xlabel(r'$v_{r}$ (km/s)')
        ax.set_ylabel(r'$v_{t}$ (km/s)')


if __name__ == '__main__':
    probe = '2'
    year = 1976
    doy = 108
    hour = 6
    minute = 36
    second = 48
    df = helios.ion_dist_single(probe, year, doy, hour, minute, second,
                                remove_advect=False)
    df = df.loc[df['counts'] > 5]
    plotter = SlicePlotter(df)
    plt.show()
