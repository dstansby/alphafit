# Script to convert hdf files to csv files
import os
import pandas as pd
import numpy as np
from config import get_dirs

output_dir = get_dirs()


probe = '2'
for year in range(1976, 1981):
    fdir = os.path.join(output_dir, 'helios' + probe, 'fits', str(year))
    if not os.path.exists(os.path.join(fdir, 'csv')):
        os.makedirs(os.path.join(fdir, 'csv'))
    for doy in range(1, 367):
        print(probe, year, doy)
        hdf_fname = 'h{}_{}_{:03}_3D_fits.h5'.format(probe, year, doy)
        csv_fname = 'h{}_{}_{:03}_3D_fits.csv'.format(probe, year, doy)

        hdf_file = os.path.join(fdir, hdf_fname)
        csv_file = os.path.join(fdir, 'csv', csv_fname)
        try:
            data = pd.read_hdf(hdf_file)
        except Exception as err:
            print('No data available')
            continue
        data = data.set_index('Time')
        # Convert all bad error codes to error code 3
        data.loc[data['Status'] > 2, 'Status'] = 3
        data = data.rename(columns={'Instrument': 'Ion instrument'})
        # Convert B instrument column to int
        data.loc[data['Status'] > 1, 'B instrument'] = -1
        data['B instrument'] = data['B instrument'].astype(int)

        data = data[['B instrument', 'Bx', 'By', 'Bz',
                     'Ion instrument', 'Status', 'Tp_par', 'Tp_perp',
                     'carrot', 'r_sun', 'clat', 'clong', 'earth_he_angle',
                     'n_p', 'sigma B', 'vp_x', 'vp_y', 'vp_z',
                     'vth_p_par', 'vth_p_perp']]

        def sigfigs(x, n):
            return '%s' % float(('%.' + str(n) + 'g') % x)

        def sigfigvars(data, varbls, n):
            data[varbls] = data[varbls].applymap(lambda x: sigfigs(x, n))
            return data

        data = sigfigvars(data, ['Bx', 'By', 'Bz', 'sigma B'], 6)
        data = sigfigvars(data, ['Tp_perp', 'Tp_par'], 6)
        data = sigfigvars(data, ['vp_x', 'vp_y', 'vp_z'], 6)
        data = sigfigvars(data, ['vth_p_par', 'vth_p_perp'], 6)
        data = sigfigvars(data, ['n_p'], 6)

        data.to_csv(csv_file, na_rep='NaN')
