#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:13:10 2023

@author: tnye
"""

###############################################################################
# Script used to set up a dataframe with all the parameter combinations.
###############################################################################

# Imports
import numpy as np
import pandas as pd

# Initialize arrays
risetime = []
vrupt = []
stress = []

# List of parameters
rt_vals = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
sf_vals = [0.37, 0.40, 0.43, 0.46, 0.49]
sd_vals = [0.1, 0.5, 1.0, 1.5, 2.0]

# Loop over parameter combinations
for rt_val in rt_vals:
    for sf_val in sf_vals:
        for sd_val in sd_vals:
            risetime.append(f'rt{rt_val}x')
            vrupt.append(f'sf{sf_val}')
            stress.append(f'sd{sd_val}')

# Set up dataframe
data = {'Risetime':risetime, 'Vrupt':vrupt, 'Stress Drop':stress}
df = pd.DataFrame.from_dict(data)

for i in range(int(275/5)):
    print(i)
    
    if i == 0:
        df_small = df.loc[:4]
        
    elif i == 54:
        df_small = df.loc[270:]
       
    else:
        df_small = df.loc[5*i:5*i+4]
    
    df_small.to_csv(f'/Users/tnye/tsuquakes/files/parameter_files/final_suite_parameter_file_{i}.csv')

