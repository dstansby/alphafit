# Script to validate version 2 of the data against version 1
import pathlib
import os

import pandas as pd

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
    if df1.shape[0] != df2.shape[0]:
        raise RuntimeError('v1: {}, v2: {}'.format(df1.size, df2.size))


if __name__ == '__main__':
    missing_counter = check_all_data()
    print('Missing {} files'.format(missing_counter))
