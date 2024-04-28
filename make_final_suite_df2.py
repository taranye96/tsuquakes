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

#%%

# Initialize arrays
risetime = []
vrupt = []

# List of parameters
rt_vals = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3]
sf_vals = [0.37,0.4,0.43,0.46,0.49]

# Loop over parameter combinations
for rt_val in rt_vals:
    for sf_val in sf_vals:
        risetime.append(f'rt{rt_val}x')
        vrupt.append(f'sf{sf_val}')

# Set up dataframe
data = {'Risetime':risetime, 'Vrupt':vrupt}
df = pd.DataFrame.from_dict(data)

sub_num = round(len(df)/4)

for i in range(4):
    
    if i == 0:
        df_small = df.loc[:sub_num-1]
    elif i == 1:
        df_small = df.loc[sub_num:2*sub_num-1]
    elif i == 2:
        df_small = df.loc[2*sub_num:3*sub_num-1]
    else:
        df_small = df.loc[3*sub_num:]
    
    df_small.to_csv(f'/Users/tnye/FakeQuakes/files/parameter_flatfiles/final_suite_tPGD_{i}.csv')


#%%

# Initialize arrays
stress = []

# List of parameters
sd_vals = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# Loop over parameter combinations
for sd_val in sd_vals:
    stress.append(f'sd{sd_val}')

# Set up dataframe
data = {'Stress Drop':stress}
df = pd.DataFrame.from_dict(data)

for i in range(4):
    
    if i == 0:
        df_small = df.loc[:2]
    elif i == 1:
        df_small = df.loc[3:4]
    elif i == 2:
        df_small = df.loc[5:5]
    else:
        df_small = df.loc[6:6]
    
    df_small.to_csv(f'/Users/tnye/FakeQuakes/files/parameter_flatfiles/final_suite_HF_{i}.csv')

