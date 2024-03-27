# ES-LM

## Descriptions
The Levenberg-Marquardt based iterative Ensemble Smoother (ES-LM) belongs to the family of iterative Ensemble Smoother (iES) methods, which are widely used for history matching and uncertainty quantification in groundwater hydrology. The ES-LM method is developed by Ma and Bi (2019) and implemented by Yang et al. (2023). This repository provides python codes for the ES-LM and examples of its implementation of inverse modeling.

## Requirements
The following requirements need to be installed.
* Python >= 3.9
* numpy >= 1.22
* scipy >= 1.7.1
* matplotlib >= 3.4.3
* os
* subprocess
* csv
* flopy >= 3.3.5

## Examples
`Example1` demonstrates the application of ES-LM to fit a quadratic curve given scatter data. `Example2` demonstrates the application of ES-LM to calibrate a 2D groundwater model given observation data. 

Python codes and data for these examples are provided in respective folders. Implementation of the ES-LM codes are explained in the python script of each example. To run these examples, the module `IES_lib.py` in `Source` needs to be copied to the work directory.

## Citation
If you find the codes helpful, please cite:

* Yang S., Tsai, F.T.-C., Bacopoulos, P., & Kees, C.E., 2023. Comparative analyses of covariance matrix adaptation and iterative ensemble smoother on high-dimensional inverse problems in high-resolution groundwater modeling, Journal of Hydrology, 130075. https://doi.org/10.1016/j.jhydrol.2023.130075
* Ma, X., & Bi, L., 2019. A robust adaptive iterative ensemble smoother scheme for practical history matching applications. Computational Geosciences, 23(3), 415-442. https://doi.org/10.1007/s10596-018-9786-9

## Liscense
MIT.
