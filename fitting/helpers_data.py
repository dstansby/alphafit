from heliopy.data import helios


def get_mag(probe, starttime, endtime):
    try:
        mag4hz = helios.mag_4hz(probe, starttime, endtime,
                                try_download=True).data

        # The 4Hz data is given in SSE, so flip to get in ~RTN
        mag4hz[['Bx', 'By', 'Bz']] *= -1
        # Helios 2 was "upside down" relative to Helios 1, so the spin plane
        # components do not need flipping
        if probe == '2':
            mag4hz['By'] *= -1
            mag4hz['Bz'] *= -1

    except Exception as err:
        if str(err) != 'No raw 4Hz mag data available':
            print(str(err))
        mag4hz = None

    # Also load 6s data as backup
    try:
        mag6s = helios.mag_ness(probe, starttime, endtime,
                                try_download=True).data
        # The 6s data is given in SSE, so flip to get in ~RTN
        mag6s[['Bx', 'By', 'Bz']] *= -1
    except Exception as err:
        if 'No 6s mag data avaialble' not in str(err):
            print(str(err))
        mag6s = None
    return mag4hz, mag6s


def clean3D(dists_3D):
    dists_3D = dists_3D[dists_3D['counts'] != 1]

    # A handful of files seem to have some garbage counts in them
    dists_3D = dists_3D[dists_3D['counts'] < 32768]

    # Throw away high and low energy bins
    # (which contain just noise)
    dists_3D = dists_3D[
        dists_3D.index.get_level_values('E_bin') > 3]
    dists_3D = dists_3D[
        dists_3D.index.get_level_values('E_bin') < 32]
    return dists_3D


def clean1D(I1as, I1bs):
    I1as = I1as[I1as['df'] != 0]
    I1bs = I1bs[I1bs['df'] != 0]
    return I1as, I1bs


def load_dists(probe, starttime, endtime):
    # Load a days worth of ion distribution functions
    dists_3D = helios.ion_dists(probe,
                                starttime, endtime,
                                verbose=True)
    print('Loaded 3D dists')
    dists_1D = helios.integrated_dists(probe,
                                       starttime, endtime,
                                       verbose=True)
    print('Loaded 1D dists')
    distparams = helios.distparams(probe,
                                   starttime, endtime,
                                   verbose=True)
    print('Loaded distribution parameters')

    I1as = dists_1D['a']
    I1bs = dists_1D['b']
    # Re-order 3D index levels
    dists_3D = dists_3D.reorder_levels(
        ['Time', 'E_bin', 'El', 'Az'], axis=0)
    return dists_3D, I1as, I1bs, distparams
