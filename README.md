This repository contains the codes referenced in Nye et al. (2024). The full reference for the paper is....

**tsuquakes/figures** contains the scripts used to make the figures in the paper. 

**tsuquakes/src** contains the scripts used to generate the simulated data and perform subsequent analyses. 
- The scripts in this directory are organized numerically by topic and the order in which they are run. For example, scripts starting with "0" should be run before those starting with "1". Scripts with a letter following the number should be run in order. 
- Scripts starting with "0" are used to prepare the observed data and finite fault inversion data from Yu et al. (2014) for this study.
- Scripts starting with "1" are used to modify the rise time and rupture velocity before generati simulations.
- The script starting with "2" is used to run the simulations.
- Scripts starting with "3" are used to evalaute the intensity measures and perform thje Gaussian process regressions.
- Scripts starting with "4" are used to compute Mpga-Mpgd for the near-real time analysis.

The data for this project can be found (include Zenodo reference once published)
