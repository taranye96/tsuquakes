#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:27:37 2023

@author: tnye
"""

# Imports
from glob import glob
import numpy as np
from numpy import genfromtxt
import pandas as pd
from os import path,makedirs,chdir
import residual_fns as res

#%%

# File paths
parameter = 'final_runs_m7.8'
home_dir = f'/Users/tnye/FakeQuakes/simulations'

# Read in residual files
gnss_res_files = sorted(glob(f'{home_dir}/{parameter}/rt*/flatfiles/residuals/*_gnss.csv'))

# Read in rupture list
rupture_list = genfromtxt(f'{home_dir}/{parameter}/standard/data/ruptures.list',dtype=str)


################################ tPGD Residuals ###############################

### Median residuals

# Initialize arrays
rise = np.array([])
vrupt = np.array([])
all_tPGD_res = np.array([])
run_name = np.array([])

# Loop over gnss files
for file in gnss_res_files:
    
    project = file.split('/')[-4]
    
    gnss_df = pd.read_csv(file)
    
    for rupture in rupture_list:
        
        run = rupture.strip('.rupt')
    
        tPGD_res = gnss_df['tPGD_res_linear'].values[np.where((gnss_df['run']==run) & (gnss_df['station']!='NGNG'))[0]]
      
        # tpgd_mean = np.mean(tPGD_res)
        tpgd_median = np.median(tPGD_res)
        
        run_name = np.append(run_name, run)
        rise = np.append(rise, float(project.split('_')[0][2:].strip('x')))
        vrupt = np.append(vrupt, float(project.split('_')[1][2:]))
        all_tPGD_res = np.append(all_tPGD_res, np.median(tpgd_median))

# Save to csv
gnss_data = {'k-factor':rise, 'ssf':vrupt,'tPGD res':all_tPGD_res}
gnss_df = pd.DataFrame(gnss_data)  
gnss_df.to_csv('/Users/tnye/tsuquakes/gaussian_process/tPGD_event_residuals_exNGNG.csv') 


# ### Median residual given for each set of parmaeters (median over all stations
#     # all ruptures scenarios)

# # Initialize arrays
# rise = np.array([])
# vrupt = np.array([])
# run_name = np.array([])
# all_tPGD_res = np.array([])

# # Loop over gnss files
# for file in gnss_res_files:
    
#     project = file.split('/')[-4]
    
#     gnss_df = pd.read_csv(file)
        
#     rise = np.append(rise, float(project.split('_')[0][2:].strip('x')))
#     vrupt = np.append(vrupt, float(project.split('_')[1][2:]))
#     pgd_res = gnss_df['pgd_res'].values[np.where(gnss_df['station']!='NGNG')[0]]
#     tPGD_res = gnss_df['tPGD_res_linear'].values[np.where(gnss_df['station']!='NGNG')[0]]
   
#     tpgd_median = np.median(tPGD_res)
#     all_tPGD_res = np.append(all_tPGD_res, tpgd_median)

# # Save to csv
# gnss_data = {'k-factor':rise, 'ssf':vrupt,'tPGD res':all_tPGD_res}
# gnss_df = pd.DataFrame(gnss_data)  
# gnss_df.to_csv('/Users/tnye/tsuquakes/gaussian_process/tPGD_parameter_residuals_exNGNG.csv') 


# # Initialize arrays
# rise = np.array([])
# vrupt = np.array([])
# run_name = np.array([])
# all_tPGD_res = np.array([])

# # Loop over gnss files
# for file in gnss_res_files:
    
#     project = file.split('/')[-4]
    
#     gnss_df = pd.read_csv(file)
        
#     rise = np.append(rise, float(project.split('_')[0][2:].strip('x')))
#     vrupt = np.append(vrupt, float(project.split('_')[1][2:]))
#     pgd_res = gnss_df['pgd_res'].values
#     tPGD_res = gnss_df['tPGD_res_linear'].values
   
#     tpgd_median = np.median(tPGD_res)
#     all_tPGD_res = np.append(all_tPGD_res, tpgd_median)

# # Save to csv
# gnss_data = {'k-factor':rise, 'ssf':vrupt,'tPGD res':all_tPGD_res}
# gnss_df = pd.DataFrame(gnss_data)  
# gnss_df.to_csv('/Users/tnye/tsuquakes/gaussian_process/tPGD_parameter_residuals.csv') 


#%%
################################# HF Residuals ################################

# File paths
parameter = 'final_runs_m7.8'
home_dir = f'/Users/tnye/FakeQuakes/simulations'

# Read in residual files
sm_res_files = sorted(glob(f'{home_dir}/{parameter}/sd*/flatfiles/residuals/*_sm.csv'))

# Read in rupture list
rupture_list = genfromtxt(f'{home_dir}/{parameter}/sd2.0/data/ruptures.list',dtype=str)

# Initialize arrays
stress = np.array([])
run_name = np.array([])
all_HF_res = np.array([])

# Loop over sm files
for file in sm_res_files:
    
    project = file.split('/')[-4]
    
    sm_df = pd.read_csv(file.replace('gnss','sm'))
    
    for rupture in rupture_list:
        
        run = rupture.strip('.rupt')
        run_name = np.append(run_name, run)
        
        stress = np.append(stress, float(project[2:]))
    
        pga_res = sm_df['pga_res'].values[np.where(sm_df['run']==run)[0]]
        
        acc_fas_res = sm_df.iloc[:,20:].values[np.where(sm_df['run']==run)[0]]
        mean_acc_fas = []
        for i in range(len(acc_fas_res)):
            mean_acc_fas.append(np.mean(acc_fas_res[i]))
        mean_HF_res = (pga_res+mean_acc_fas)/2
        median_HF_all_stns = np.median(mean_HF_res)
        all_HF_res = np.append(all_HF_res,median_HF_all_stns)

sm_data = {'run':run_name, 'stress drop':stress, 'HF res':all_HF_res}
sm_df = pd.DataFrame(sm_data)  
sm_df.to_csv('/Users/tnye/tsuquakes/gaussian_process/HF_mean_residuals_corrected.csv')    
        