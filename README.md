### Supplementary code for Sture et Al. 2019 "Identification of Hyperspectral Signatures for Underwater Massive Sulphide Exploration"

The repository contains the following python scripts. 
Only python 3.6 or newer is supported. Python must have the following libraries installed.

- h5py (tested: 2.9.0)
- imageio (tested: 2.5.0)
- matplotlib (tested: 3.1.1)
- numpy (tested: 1.17.1)

Optional interaction with Gaussian process models requires the following additional packages.

- tensorflow (tested: 1.14.0)
- GPFlow (tested: 1.5.0)
- pandas (tested: 0.25.0)
- scikit-learn (tested: 0.21.3)


### Data
The default paths of the scripts in this repository expects a folder called *data* with the following subdirectories; *masks*, *model* and *samples*. These folders contain png-files denoting the masks in which reflectance curves are calculated from, a Gaussian process model / calibration data, and UHI data from the respective samples. The paths can be modified in the scripts if necessary. 

### Scripts
The following main scripts are available

#### download.py
Downloads a zip archive containing UHI data and pre-computed calibration data.
The archive is extracted in the project folder.

#### viewer.py
A simple viewer in matplotlib for the UHI-data.

#### refl_curves.py
Create the reflectance plots in figure 6 and 7 in the paper.

The reflectance is calculated either based on calibration data stored in the UHI-file,
or from a stored Gaussian process model.


#### train_model.py


