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

    alpha_code_dir = os.path.expanduser(
        config['data_dirs']['alpha_code_dir'])
    alpha_code_dir = pathlib.Path(alpha_code_dir)
    if not os.path.isdir(alpha_code_dir):
        raise FileNotFoundError('alphafit code directory "{}" specified in '
                                'config.ini does not exist'.format(
                                    alpha_code_dir))

    return output_dir, alpha_code_dir
