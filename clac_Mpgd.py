#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:29:18 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from glob import glob
import valid_fns as valid
from scipy import linalg

# Set up working directory
home_dir = '/Users/tnye/FakeQuakes/simulations/ideal_runs_m7.8'
# home_dir = '/Users/tnye/FakeQuakes/simulations/ideal_runs_m7.8/standard'
all_files = sorted(glob(f'{home_dir}/*/flatfiles/IMs/*_gnss.csv'))

# # Get intensity measure files
# TsE_IM_files = sorted(glob(f'{home_dir}/*/flatfiles/IMs/*_gnss.csv'))
# standard_IM_files = sorted(glob(f'{home_dir}/standard/flatfiles/IMs/*_gnss.csv'))
# all_files = TsE_IM_files + standard_IM_files
Mpgd_list = []
events = []

# Loop over event IM files
for file in all_files:
    
    # Get rupt file
    project = file.split('/')[-4]
    run = file.split('/')[-1].split('_')[0]
    rupt_file = f'{home_dir}/{project}/output/ruptures/{run}.rupt'
    # rupt_file = f'{home_dir}/output/ruptures/{run}.rupt'

    # Get station coords
    df = pd.read_csv(file)
    gnss_stns = df['station'].values
    stlons = df['stlon'].values
    stlats = df['stlat'].values
    stelevs = df['stelev'].values
    
    # Get PGD
    pgd_list = 100*df['pgd'].values
    
    # Get Rp
    Rp_list = []
    for i in range(len(gnss_stns)):
        Rp = valid.get_Rp(stlons[i], stlats[i], stelevs[i], rupt_file, exp=-2.3)
        Rp_list.append(Rp)
    
    # M = get_pgd_scaling(Mw, R, model)
    
    # Set up least squares inversion for MPGD
    G =  np.zeros((len(gnss_stns), 1))
    d =  np.zeros((len(gnss_stns), 1))
    for i in range(len(gnss_stns)):
        # G[i][0] = 1.047 + (-0.168 * np.log10(Rp_list[i]))
        # d[i][0] = np.log10(pgd_list[i]) + 4.34
        
        G[i][0] = 1.303 + (-0.168 * np.log10(Rp_list[i]))
        d[i][0] = np.log10(pgd_list[i]) + 5.902
        
    
    G_inv = np.linalg.pinv(G, rcond=1e-13)
    m = G_inv @ d
    
    Mpgd_list.append(round(m[0][0],2))
    events.append(f'{project}_{run}')

data = {'Event':events,'Mpgd':Mpgd_list}
Mpgd_df = pd.DataFrame(data)
Mpgd_df.to_csv('/Users/tnye/tsuquakes/realtime_analysis/Mpgd_m7.8.csv',index=None)
    
    
