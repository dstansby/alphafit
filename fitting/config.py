# Script to read in output directory configuration file
#
# David Stansby 2017
import os
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

    return output_dir
