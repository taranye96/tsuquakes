#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:27:37 2023

@author: tnye
"""

###############################################################################
# This script calculates the tPGD residuals and HF residuals that are used to
# perform the Gaussian process regression.
###############################################################################

# Imports
from glob import glob
import numpy as np
from numpy import genfromtxt
import pandas as pd

# File paths
parameter = 'gpr_simulations'
home_dir = '/Users/tnye/tsuquakes/simulations'

# Read in rupture list
rupture_list = genfromtxt(f'{home_dir}/{parameter}/standard/data/ruptures.list',dtype=str)


#%%

################################ tPGD Residuals ###############################

# Read in residual files
gnss_res_files = sorted(glob(f'{home_dir}/{parameter}/rt*/flatfiles/residuals/*_gnss.csv'))

# Initialize arrays
rise = np.array([])
vrupt = np.array([])
all_tPGD_res = np.array([])
run_name = np.array([])

# Loop over gnss files
for file in gnss_res_files:
    
    # Get project name
    project = file.split('/')[-4]
    
    # Read in GNSS residual file
    gnss_df = pd.read_csv(file)
    
    # Loop over ruptures
    for rupture in rupture_list:
        
        # Get run number
        run = rupture.strip('.rupt')
    
        # Get residuals for specific run 
            # Exclude residuals from station NGNG because the stations is pretty noisy
        tPGD_res = gnss_df['tPGD_res_linear'].values[np.where((gnss_df['run']==run) & (gnss_df['station']!='NGNG'))[0]]
      
        # Get median tPGD residual 
        tpgd_median = np.median(tPGD_res)
        
        # Append values to lists
        run_name = np.append(run_name, run)
        rise = np.append(rise, float(project.split('_')[0][2:].strip('x')))
        vrupt = np.append(vrupt, float(project.split('_')[1][2:]))
        all_tPGD_res = np.append(all_tPGD_res, np.median(tpgd_median))

# Save to csv
gnss_data = {'k-factor':rise, 'ssf':vrupt,'tPGD res':all_tPGD_res}
gnss_df = pd.DataFrame(gnss_data)  
gnss_df.to_csv('/Users/tnye/tsuquakes/gaussian_process/tPGD_parameter_residuals_exNGNG.csv') 


#%%

################################# HF Residuals ################################

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
    
    # Get project name
    project = file.split('/')[-4]
    
    # Read in strong motion residual file
    sm_df = pd.read_csv(file.replace('gnss','sm'))
    
    # Loop over ruptures
    for rupture in rupture_list:
        
        # Get run number
        run = rupture.strip('.rupt')
    
        # Get PGA  residuals for specific run 
        pga_res = sm_df['pga_res'].values[np.where(sm_df['run']==run)[0]]
        
        # Get average Fourier spectra residuals for each station for the
            # specific run
        acc_fas_res = sm_df.iloc[:,20:].values[np.where(sm_df['run']==run)[0]]
        mean_acc_fas = []
        for i in range(len(acc_fas_res)):
            mean_acc_fas.append(np.mean(acc_fas_res[i]))
        mean_HF_res = (pga_res+mean_acc_fas)/2
        
        # Get median HF residual 
        median_HF_all_stns = np.median(mean_HF_res)
        
        # Append values to lists
        run_name = np.append(run_name, run)
        stress = np.append(stress, float(project[2:]))
        all_HF_res = np.append(all_HF_res,median_HF_all_stns)

# Save to csv
sm_data = {'run':run_name, 'stress drop':stress, 'HF res':all_HF_res}
sm_df = pd.DataFrame(sm_data)  
sm_df.to_csv('/Users/tnye/tsuquakes/gaussian_process/HF_mean_residuals.csv')    
        