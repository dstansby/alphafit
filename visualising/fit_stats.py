# Output stats from helios distribution fitting
#
# David Stansby 2017
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import heliopy.data.helios as helios

fitparamlist = []
starttime = datetime(1974, 1, 1, 0, 0, 0)
endtime = starttime + timedelta(days=11 * 365)

'''
print('Loading 1D...')
params_1D = helios.ion_fitparams_1D(probe, starttime, endtime)
'''
params_3D = []
print('Loading 3D...')
for probe in ['1', '2']:
    print(probe)
    params_3D.append(helios.ion_fitparams_3D(probe, starttime, endtime))
params_3D = pd.concat(params_3D)

ndists = params_3D.shape[0]


def summarystr(code, string):
    n = (params_3D['Status'] == code).sum()
    return (str(n) + '\t' + string + '\t' +
            '(' + str(100 * n / ndists)[:4] + '%)')


print('\n')
print('3D Summary')
print('=======')
print(ndists, 'total points')
print('\n')
n = (params_3D['Status'] == 1).sum() + (params_3D['Status'] == 2).sum()
print(str(n) + '\t' + 'successful fits' + '\t' +
      '(' + str(100 * n / ndists)[:4] + '%)')
print(summarystr(1, 'including n/T'))
print(summarystr(2, 'no B data'))
print(summarystr(4, 'v out of range'))
print(summarystr(5, '< 6 points'))
print(summarystr(6, 'fitting failed'))
print(summarystr(9, 'corrupted file'))
print(summarystr(10, 'no proton peak'))
print(summarystr(11, 'n overestimated'))
print(summarystr(12, 'anglular bins'))

print('\n')
print('I1a:', (params_3D['Instrument'] == 1).sum() / ndists)
print('I3: ', (params_3D['Instrument'] == 2).sum() / ndists)
