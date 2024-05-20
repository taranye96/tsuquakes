#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:27:46 2024

@author: tnye
"""

###############################################################################
# This script makes Figure S9 in the supporting information.
###############################################################################

# Imports
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# Read in GMM residual flatfile
df = pd.read_csv('/Users/tnye/tsuquakes/realtime_analysis/PGA_GMM_residuals.csv')

# Read in Mpgd flatfile
Mpgd_df = pd.read_csv('/Users/tnye/tsuquakes/realtime_analysis/Mpgd.csv')

# Define trial magnitudes
trial_M = np.arange(5.5,8.5,0.1)

# Get list of parameters and runs
params = np.unique(df['Parameters'])
runs = np.unique(df['Run'])

mu1 = 'rt1.234x_sf0.41_sd1.196'
mu2 = 'rt1.954x_sf0.469_sd1.196'
sig1 = 'rt1.4x_sf0.45_sd1.0'
sig2 = 'rt1.4x_sf0.45_sd2.0'
sig3 = 'rt1.75x_sf0.42_sd1.0'
sig4 = 'rt1.75x_sf0.42_sd2.0'

#%%

# Loop over paramter combinations
param = params[0]
    
# Loop over runs
run = runs[0]
    
# Get Mpgd for this parameter and run
Mpgd = Mpgd_df['Mpgd'].values[np.where(Mpgd_df['Event']==f'{param}_{run}')[0]]        

# Initialize K-S test lists
p_val_list = np.array([])
statistic_list = np.array([])

# Loop over trial magnitudes
for M in trial_M:
    
    # Find indices for the specific parameters, event, and trial magnitude
    ind = np.where((df['Mag']==round(M,1)) & (df['Parameters']==param) & 
                   (df['Run']==run))[0]
    
    # Get PGA residuals for this event
    run_res = df['lnZhao06_PGA_Res'].values[ind]
    
    # Define distribution for GMM
    gmm_std = df['Zhao06_PGA_std'].values[ind]
    gmm_distribution = np.random.normal(0, gmm_std[0], 10000)
    
    # Run K-S test and append values to lists
    statistic, p_value = ks_2samp(run_res, gmm_distribution)
    p_val_list = np.append(p_val_list,p_value)
    statistic_list = np.append(statistic_list,statistic)

# Get MPGA
Mpga = trial_M[np.where(np.abs(p_val_list-0.05) == np.min(np.abs(p_val_list-0.05)))][0]


#%%

# Plot results
fig, axs = plt.subplots(2,1,figsize=(6,5))
axs[0].semilogy(trial_M,p_val_list)
axs[0].axhline(0.05,c='gray',ls='--',lw=1,label='significance level')
axs[0].axvline(Mpgd,c='goldenrod',lw=0.8,label=r'$M_{PGD}$')
axs[0].axvline(Mpga,c='red',lw=0.8,label=r'$M_{PGA}$')
axs[0].set_xlabel('Magnitude')
axs[0].set_ylabel('P-value')
axs[0].legend()
axs[0].grid(alpha=0.75)
axs[0].set_xlim(5.5,8.4)

axs[1].plot(trial_M,statistic_list)
axs[1].axvline(Mpgd,c='goldenrod',lw=0.8,label=r'$M_{PGD}$')
axs[1].set_xlabel('Magnitude')
axs[1].set_ylabel('Statistic')
axs[1].legend()
axs[1].grid(alpha=0.75)
axs[1].set_xlim(5.5,8.4)

plt.subplots_adjust(left=0.125,bottom=0.11,top=0.975,right=0.975,hspace=0.4)
plt.savefig(f'/Users/tnye/tsuquakes/manuscript/figures/FigS9_example_Mpga.png',dpi=300)
