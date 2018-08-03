
# coding: utf-8

# In[1]:


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


# In[4]:


# Set probe and dates to compare here
probe = '2'
starttime = datetime(1976, 3, 8)
endtime = starttime + timedelta(days=30)

merged = helios.merged(probe, starttime, endtime)
alphas = helpers.load_alphafit(probe, starttime, endtime)
protons = helios.corefit(probe, starttime, endtime, try_download=False)


# In[3]:


protons = protons.reindex(alphas.index)
protons = helioshelp.calculate_derived(protons)

alphas[['Bx', 'By', 'Bz', '|B|']] = protons[['Bx', 'By', 'Bz', '|B|']]
alphas['Beta_par'] = alphas['Ta_par'] * const.k_B.value * alphas['n_a'] * 1e6 / (alphas['|B|']**2 * 1e-18 / (2 * const.mu0))
alphas['Tani'] = alphas['Ta_perp'] / alphas['Ta_par']

