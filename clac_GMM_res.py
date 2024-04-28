#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 22:24:16 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from glob import glob
from pyproj import Geod   
import gmm_call as gmm
import scipy.constants as sp
import valid_fns as valid

# Set up working directory
# home_dir = '/Users/tnye/FakeQuakes/simulations/ideal_runs_m7.8/standard'

home_dir = '/Users/tnye/FakeQuakes/simulations/ideal_runs_m7.8'
all_files = sorted(glob(f'{home_dir}/*/flatfiles/IMs/*_sm.csv'))

# # Get intensity measure files
# TsE_IM_files = sorted(glob(f'{home_dir}/*/flatfiles/IMs/*_sm.csv'))
# standard_IM_files = sorted(glob(f'{home_dir}/standard/flatfiles/IMs/*_sm.csv'))
# all_files = TsE_IM_files + standard_IM_files

# Define parameters
trial_M = np.arange(5.5,8.5,0.1)
hypdepth = 8.82
vs30 = pd.read_csv('/Users/tnye/tsuquakes/data/vs30/sm_vs30_close.csv').vs30

def compute_rrup(rupt_file, stlon, stlat):
    
   
    #get rupture
    rupt = np.genfromtxt(rupt_file)
    Nsubfaults = len(rupt)
    
    #keep only those with slip
    i = np.where(rupt[:,12]>0)[0]
    rupt = rupt[i,:]
    
    #get Rrupt
    #projection obnject
    p = Geod(ellps='WGS84')
    
    #lon will have as many rows as Vs30 points and as many columns as subfautls in rupture
    Nsubfaults = len(rupt)
    lon_surface = np.tile(stlon,(Nsubfaults,1)).T
    lat_surface = np.tile(stlat,(Nsubfaults,1)).T
    lon_subfaults = np.tile(rupt[:,1],(len(stlon),1))-360
    lat_subfaults = np.tile(rupt[:,2],(len(stlon),1))
    az,baz,dist = p.inv(lon_surface,lat_surface,lon_subfaults,lat_subfaults)
    dist = dist/1000
    
    #get 3D distance
    z = np.tile(rupt[:,3],(len(stlon),1))
    xyz_dist = (dist**2 + z**2)**0.5
    rrup = xyz_dist.min(axis=1)
    
    return(rrup)

M_list = np.array([])
event_list = np.array([])
stn_list = np.array([])
pga_res_list = np.array([])
pga_std_list = np.array([])
params_list = np.array([])
runs_list = np.array([])

# Loop over different trial magnitudes
for M in trial_M:
    
    print(f'Working on mag {M}')

    # Loop over event IM files
    for file in all_files:
        
        # Get rupt file
        project = file.split('/')[-4]
        run = file.split('/')[-1].split('_')[0]
        rupt_file = f'{home_dir}/{project}/output/ruptures/{run}.rupt'
        # rupt_file = f'{home_dir}/output/ruptures/{run}.rupt'
        
        # Read in csv
        sm_df = pd.read_csv(file)
    
        # Get PGA
        syn_pga = sm_df['pga'].values
        
        # Get station coords
        stns = sm_df['station'].values
        stlon = sm_df['stlon'].values
        stlat = sm_df['stlat'].values
        
        # Get Rrup
        rrup = compute_rrup(rupt_file, stlon, stlat)
        
        # Compute Zhoa (2006) PGA prediction
        ln_pga_g, pga_std = gmm.zhao2006(M,hypdepth,rrup,vs30)
        zhao_pga = np.exp(ln_pga_g) * sp.g
        
        # Compute GMM residuals
        pga_res = np.log(syn_pga/zhao_pga)
        
        # Append values to lists
        M_list = np.append(M_list,[round(M,1)]*len(pga_res))
        event_list = np.append(event_list,[f'{project}_{run}']*len(pga_res))
        params_list = np.append(params_list,[project]*len(pga_res))
        runs_list = np.append(runs_list,[run]*len(pga_res))
        stn_list = np.append(stn_list,stns)
        pga_res_list = np.append(pga_res_list,pga_res)
        pga_std_list = np.append(pga_std_list,pga_std)

# Assemble to df
data = {'Mag':M_list,'Event ID':event_list,'Parameters':params_list,
        'Run':runs_list,'Station Name':stn_list,'lnZhao06_PGA_Res':pga_res_list,
        'Zhao06_PGA_std':pga_std_list,}
res_df = pd.DataFrame(data)
res_df.to_csv('/Users/tnye/tsuquakes/realtime_analysis/PGA_GMM_residuals_m7.8.csv')
