This repository contains the code used to create the Helios 'corefit' dataset.

### Reproducing the data set
The software is a series of scripts written in Python. The following steps can
be used to download the code, original data files, and required dependencies:

1. Dowload a copy of this repository
2. Install [python](https://www.python.org/)
3. Install the required python dependencies by running
`pip install -r requirements.txt` from the `corefit` directory
4. Configure heliopy
5. Download the original distribution function files from ftp://apollo.ssl.berkeley.edu/pub/helios-data/E1_experiment/helios_raw/
6. In the `corefit/fitting` directory, change the name of
`config.ini.template` to `config.ini` and fill in the directory where you
would like the output data to be saved to

The corefit data set can now be re-generated by changing to the
`corefit/fitting` directory and running

```bash
python save_dist_params.py
```

This will generate the data set in hdf files, which are easily read by python.
To clean and convert the files to ascii .csv files run

```bash
python convert_distparams.py
```

### Acknowledgements
The code was written by myself (David Stansby). The majority of it was written
during a 2 week visit to UC Berkeley (USA) hosted by Chadi Salem. Thanks to
Chadi Salem, Tim Horbury, and Lorenzo Matteini for useful discussions and
advice concerning distribution functions and the Helios mission.
