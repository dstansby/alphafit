# Script to read in output directory configuration file
#
# David Stansby 2017
import os
import pathlib
import configparser


def get_dirs():
    config = configparser.ConfigParser()
    config.read('config.ini')
    if not os.path.isfile('config.ini'):
        raise FileNotFoundError('Could not find a config.ini file. '
                                'Try renaming the config.ini.template file to '
                                'config.ini in the current directory')

    output_dir = os.path.expanduser(config['data_dirs']['output_dir'])
    if not os.path.isdir(output_dir):
        raise FileNotFoundError('Output directory "{}" specified in '
                                'config.ini does not exist'.format(output_dir))

    corefit_code_dir = os.path.expanduser(
        config['data_dirs']['corefit_code_dir'])
    corefit_code_dir = pathlib.Path(corefit_code_dir)
    if not os.path.isdir(corefit_code_dir):
        raise FileNotFoundError('Corefit code directory "{}" specified in '
                                'config.ini does not exist'.format(
                                    corefit_code_dir))

    return output_dir, corefit_code_dir
