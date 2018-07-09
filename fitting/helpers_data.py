from heliopy.data import helios


def get_mag(probe, starttime, endtime):
    try:
        mag4hz = helios.mag_4hz(probe, starttime, endtime,
                                try_download=True)

        # Deal with the flipped 4Hz data on Helios 2
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
                                verbose=False, try_download=False)
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