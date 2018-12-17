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
        raise FileNotFoundError('\nCould not find a config.ini file.'
                                '\nTry renaming the config.ini.template file to '
                                'config.ini in the current directory')

    output_dir = os.path.expanduser(config['data_dirs']['output_dir'])
    output_dir = pathlib.Path(output_dir)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError('Output directory "{}" specified in '
                                'config.ini does not exist'.format(output_dir))

    alphafit_code_dir = os.path.expanduser(
        config['data_dirs']['alphafit_code_dir'])
    alphafit_code_dir = pathlib.Path(alphafit_code_dir)
    if not os.path.isdir(alphafit_code_dir):
        raise FileNotFoundError('alphafit code directory "{}" specified in '
                                'config.ini does not exist'.format(
                                    alphafit_code_dir))

    return output_dir, alphafit_code_dir
