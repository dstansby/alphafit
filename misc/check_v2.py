# Script to validate version 2 of the data against version 1
import pathlib
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

old_dir = pathlib.Path('/Users/dstansby/Desktop/old_corefit')
new_dir = pathlib.Path('/Users/dstansby/Data/helios/new_corefit/csv')


def check_all_data():
    missing_counter = 0
    for (dirpath, dirnames, fnames) in os.walk(old_dir):
        if not len(fnames):
            continue
        relative_dir = pathlib.Path(dirpath).parts[-2:]
        relative_dir = pathlib.Path(relative_dir[0]) / relative_dir[1]

        oldpath = old_dir / relative_dir
        newpath = new_dir / relative_dir
        for fname in fnames:
            oldfile = oldpath / fname
            newfile = newpath / fname

            if not newfile.exists():
                print('Cannot find {}'.format(newfile))
                missing_counter += 1
                continue
            print(newfile)
            olddata = pd.read_csv(oldfile)
            newdata = pd.read_csv(newfile)
            try:
                check_single_day(olddata, newdata)
            except Exception as err:
                print(str(err))
                raise RuntimeError('Comparision for file {} failed'.format(
                    newfile))
    return missing_counter


def check_single_day(df1, df2):
    # Check that number of rows is the same
    if df1.shape[0] != df2.shape[0]:
        raise RuntimeError('v1: {}, v2: {}'.format(df1.size, df2.size))

    # Check that new data has 1 extra column
    if df1.shape[1] != df2.shape[1] - 1:
        raise RuntimeError('Column mismatch')
    for key in ['n_p']:
        if np.sum(~np.isclose(df1[key].values,
                              df2[key].values,
                              rtol=1e-3, atol=0, equal_nan=True)) > 5:
            fig, axs = plt.subplots(nrows=2, sharex=True)
            axs[0].scatter(df1.index, df1[key])
            axs[0].scatter(df1.index, df2[key])
            axs[1].scatter(df1.index, df1[key] - df2[key])
            plt.show()


if __name__ == '__main__':
    missing_counter = check_all_data()
    print('Missing {} files'.format(missing_counter))
