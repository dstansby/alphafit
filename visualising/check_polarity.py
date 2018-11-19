# A script to plot some Helios and IMP8 data side by side to double check
# that manually flipping the Helios magnetic field data values is the
# correct thing to do.

from datetime import datetime
import matplotlib.pyplot as plt
import heliopy.data.imp as imp
import heliopy.data.helios as helios

starttime = datetime(1976, 1, 17)
endtime = datetime(1976, 3, 1)
iprobe = '8'

heliosdata = helios.mag_ness('2', starttime, endtime)
heliosdata.data = heliosdata.data.resample('2H').mean()

# Import IMP8 data in the solar wind. These data are in GSE coordinates,
# which have the x-axis pointing towards the Sun
impdata = imp.merged(iprobe, starttime, endtime)
impdata.data = impdata.data.loc[impdata.data['sw_flag'] == 0]
impdata.data = impdata.data.resample('2H').mean()

# Compare data
fig, axs = plt.subplots(nrows=2, sharex=True)
ax = axs[0]
ax.plot(impdata.index, impdata.quantity('Bx_gse'))
ax.plot(impdata.index, impdata.quantity('By_gse'))
ax = axs[1]
ax.plot(heliosdata.index, heliosdata.quantity('Bx'))
ax.plot(heliosdata.index, heliosdata.quantity('By'))
plt.show()
