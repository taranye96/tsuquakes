#!/bin/bash

model_name=model4.6.7

# Make files for station names and coords
python /Users/tnye/kappa/code/site_res/vs30_stns.py ${model_name}

# Grid search
cd /Users/tnye/kappa/data/Vs30
gmt grdtrack ${model_name}_stn_coords_culled.txt -Gglobal_vs30.grd -h -Z > /Users/tnye/kappa/data/Vs30/${model_name}_vs30.txt
