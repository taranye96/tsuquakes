#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:43:47 2023

@author: tnye
"""

###############################################################################
# This script turns a .rupt file into a txt file for plotting in GMT.
###############################################################################

# Iports 
import numpy as np
import pandas as pd
from glob import glob
from os import path, makedirs
import matplotlib.pyplot as plt

# Gather .rupt files
rupt_files = sorted(glob('/Users/tnye/tsuquakes/simulations/test_simulations/standard/output/ruptures/mentawai*.rupt'))
max_slip_list = []

# Loop over .rupt files
for file in rupt_files:
    
    # Get run name
    run = file.split('/')[-1].strip('.rupt')
    
    # Get rupture data
    rupt = np.genfromtxt(file)
    lon = rupt[:,1]
    lat = rupt[:,2]
    ss_slip = rupt[:,8]
    ds_slip = rupt[:,9]
    slip = np.sqrt(ss_slip**2 + ds_slip**2)
    
    max_slip = np.max(slip)
    print(max_slip)
    max_slip_list.append(max_slip)
    
    # Create output directory
    if not path.exists('/Users/tnye/tsuquakes/rupture_models/test_simulations'):
        makedirs('/Users/tnye/tsuquakes/rupture_models/test_simulations')
    
    # Turn into a dataframe
    data = {'Longitude':lon,'Latitude':lat,'Slip':slip}
    df = pd.DataFrame.from_dict(data)
    df.to_csv(f'/Users/tnye/tsuquakes/rupture_models/test_simulations/{run}.txt',header=False,index=False)
    

plt.figure()
plt.hist(max_slip_list)
plt.title('')
plt.xlabel('Max Slip (km)')
    