#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:16:19 2020

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import os

# Columns are number of subfaults
col_nums = np.arange(0,963)
columns = []
for col in col_nums:
    col = 'Subfault' + str(col) + ' slip(m)'
    columns.append(col)

# Initialize dataframe
ann_df = pd.DataFrame(columns=columns)

rootdir = '/Users/tnye/Classes/Geol610/project/Cascadia/data/'

# Loop through runs
runs = []
for subdir, dirs, files in os.walk(rootdir):
    for i, run in enumerate(dirs):
        runs.append(run)
        run_dir = f'/Users/tnye/Classes/Geol610/project/Cascadia/data/{run}/'

        # Calcualte slip on each subfault
        slip_df = pd.read_csv(run_dir + '_slip.csv')
        ss_slip = slip_df.iloc[:,0]
        ds_slip = slip_df.iloc[:,1]
        slip = np.sqrt(ss_slip**2 + ds_slip**2)
        
        # Append slip to dataframe
        ann_df.loc[i] = slip

# Replace NaNs with 0s
for col in columns:
    ann_df[col] = ann_df[col].fillna(0)
    
ann_df.insert(0, 'Run', runs, True) 


