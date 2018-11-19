# A script to plot some Helios and IMP8 data side by side to double check
# that manually flipping the Helios magnetic field data values is the
# correct thing to do.

from datetime import datetime
import matplotlib.pyplot as plt
import heliopy.data.imp as imp
import heliopy.data.helios as helios

hprobe = '1'
starttime = datetime(1974, 12, 12)
endtime = datetime(1975, 1, 1)

'''hprobe = '2'
starttime = datetime(1976, 1, 17)
endtime = datetime(1976, 2, 10)'''
iprobe = '8'

sixs = helios.mag_ness(hprobe, starttime, endtime)
sixs.data = sixs.data.resample('2H').mean()
fourhz = helios.mag_4hz(hprobe, starttime, endtime)
fourhz.data = fourhz.data.resample('2H').mean()

# Import IMP8 data in the solar wind. These data are in GSE coordinates,
# which have the x-axis pointing towards the Sun
impdata = imp.merged(iprobe, starttime, endtime)
impdata.data = impdata.data.loc[impdata.data['sw_flag'] == 0]
impdata.data = impdata.data.resample('2H').mean()

# Compare data
fig, axs = plt.subplots(nrows=3, sharex=True)
ax = axs[0]
ax.plot(impdata.index, impdata.quantity('Bx_gse'))
ax.plot(impdata.index, impdata.quantity('By_gse'))
ax.plot(impdata.index, impdata.quantity('Bz_gse'))
ax.set_title('IMP8 (GSE)')
ax = axs[1]
ax.plot(sixs.index, sixs.quantity('Bx'))
ax.plot(sixs.index, sixs.quantity('By'))
ax.plot(sixs.index, sixs.quantity('Bz'))
ax.set_title('Helios 6s')
ax = axs[2]
ax.plot(fourhz.index, fourhz.quantity('Bx'))
ax.plot(fourhz.index, fourhz.quantity('By'))
ax.plot(fourhz.index, fourhz.quantity('Bz'))
ax.set_title('Helios 4Hz')

fig.suptitle('Helios {}'.format(hprobe))
plt.show()
