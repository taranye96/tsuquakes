#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:29:18 2023

@author: tnye
"""

###############################################################################
# This script estimates Mpgd for the different simulations. 
###############################################################################

# Imports
import numpy as np
import pandas as pd
from glob import glob
import tsuquakes_main_fns as tmf
from scipy import linalg

# Set up working directory
home_dir = '/Users/tnye/tsuquakes/simulations/tse_simulations'

# Grab intensity measure flatfiles for the GNSS stations
all_files = sorted(glob(f'{home_dir}/*/flatfiles/IMs/*_gnss.csv'))

# Initialize lists
Mpgd_list = []
events = []

# Loop over event IM files
for file in all_files:
    
    # Get rupt file
    project = file.split('/')[-4]
    run = file.split('/')[-1].split('_')[0]
    rupt_file = f'{home_dir}/{project}/output/ruptures/{run}.rupt'

    # Get station coords
    df = pd.read_csv(file)
    gnss_stns = df['station'].values
    stlons = df['stlon'].values
    stlats = df['stlat'].values
    stelevs = df['stelev'].values
    
    # Get PGD in cm
    pgd_list = 100*df['pgd'].values
    
    # Get Rp distance metric
    Rp_list = []
    for i in range(len(gnss_stns)):
        Rp = tmf.get_Rp(stlons[i], stlats[i], stelevs[i], rupt_file, exp=-2.3)
        Rp_list.append(Rp)
    
    # Initialize arrays for least dquares inversion
    G =  np.zeros((len(gnss_stns), 1))
    d =  np.zeros((len(gnss_stns), 1))
    
    # Loop over stations
    for i in range(len(gnss_stns)):
        
        # Set up G matrix (coefficients and distance) and d vector (PGD)
            # Coefficients are from the Goldberg et al. (2021) study
        G[i][0] = 1.303 + (-0.168 * np.log10(Rp_list[i]))
        d[i][0] = np.log10(pgd_list[i]) + 5.902
        
    # Perform inversion
    G_inv = np.linalg.pinv(G, rcond=1e-13)
    m = G_inv @ d
    
    # Append values to lists
    Mpgd_list.append(round(m[0][0],2))
    events.append(f'{project}_{run}')

# Save to dataframe
data = {'Event':events,'Mpgd':Mpgd_list}
Mpgd_df = pd.DataFrame(data)
Mpgd_df.to_csv('/Users/tnye/tsuquakes/realtime_analysis/Mpgd.csv',index=None)
    
    
