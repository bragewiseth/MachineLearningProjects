# FYS-STK3155 Projects
This repository is a collection of projects in FYS-STK 3155/4155 Applied Data Analysis and 
Machine Learning at the University of Oslo

## ⚠️ IMPORTANT
For our projects we use the local python package
ADAMLL - Applied Data Analysis and Machine Learning Library
you can find the source code for this package in `ADAMLL-package\`
To install it, run the following command in the terminal:
```
pip install ADAMLL-package\
```
alternativly you can navigate to this folder and run
```
pip install .
```
if you want to make changes to the package we advise that you install the package with the `-e` flag.
Any changes to the package requires a rebulid to take effect

The package will install any dependencies that are not already installed
in addition to the package itself you need to install the following packages in order to run the code in 
this repo:

- `matplotlib`
- `seaborn`
- `tensorflow`


The main code is found in the 'src' folder in the notebook 'waveequation.ipynb'
Animations of the results are found in the 'runsAndFigures' folder