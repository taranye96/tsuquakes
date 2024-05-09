#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:40:15 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# Read in residual file
df = pd.read_csv('/Users/tnye/tsuquakes/realtime_analysis/PGA_GMM_residuals_standard_m7.8.csv')
Mpgd_df = pd.read_csv('/Users/tnye/tsuquakes/realtime_analysis/Mpgd_standard_m7.8.csv')

trial_M = np.arange(5.5,8.5,0.1)
params = np.unique(df['Parameters'])
runs = np.unique(df['Run'])

#%%

param_list = []
run_list = []
Mpga_list = []
Mpgd_list = []
Mdiff_list = []

param_res = []

for param in params:
    
    for run in runs:
        
        # Get Mpgd
        Mpgd = Mpgd_df['Mpgd'].values[np.where(Mpgd_df['Event']==f'{param}_{run}')[0]]        
    
        p_val_list = np.array([])
        statistic_list = np.array([])
        
        for M in trial_M:
            
            # Get residuals for this event
            ind = np.where((df['Mag']==round(M,1)) & (df['Parameters']==param) & 
                           (df['Run']==run))[0]
            run_res = df['lnZhao06_PGA_Res'].values[ind]
            
            # Define distribution for GMM
            gmm_std = df['Zhao06_PGA_std'].values[ind]
            gmm_distribution = np.random.normal(0, gmm_std[0], 10000)
            
            # Run K-S test
            statistic, p_value = ks_2samp(run_res, gmm_distribution)
            p_val_list = np.append(p_val_list,p_value)
            statistic_list = np.append(statistic_list,statistic)
        
        # Append values
        param_list.append(param)
        run_list.append(run)
        Mpga = trial_M[np.where(np.abs(p_val_list-0.05) == np.min(np.abs(p_val_list-0.05)))][0]
        Mpga_list.append(Mpga)
        Mpgd_list.append(Mpgd[0])
        Mdiff_list.append(Mpga - Mpgd[0])
        
        # # Plot results
        # fig, axs = plt.subplots(2,1)
        # axs[0].semilogy(trial_M,p_val_list)
        # axs[0].axhline(0.05,c='gray',ls='--',lw=0.8,label='significance level')
        # axs[0].axvline(Mpgd,c='goldenrod',lw=0.8,label=r'$M_{PGD}$')
        # axs[0].set_xlabel('Magnitude')
        # axs[0].set_ylabel('P-value')
        # axs[0].legend()
        # axs[1].plot(trial_M,statistic_list)
        # axs[1].axvline(Mpgd,c='goldenrod',lw=0.8,label=r'$M_{PGD}$')
        # axs[1].set_xlabel('Magnitude')
        # axs[1].set_ylabel('Statistic')
        # axs[1].legend()
        # plt.subplots_adjust(top=0.95,right=0.95,hspace=0.35)
        # plt.savefig(f'/Users/tnye/tsuquakes/realtime_analysis/Mpga_plots/{param}_{run}.png',dpi=300)
        # plt.close()

data = {'Parameters':param_list,'Run':run_list,'Mpgd':Mpgd_list,'Mpga':Mpga_list,
        'Mpga-Mpgd':Mdiff_list}
Mdiff_df = pd.DataFrame(data)
Mdiff_df.to_csv('/Users/tnye/tsuquakes/realtime_analysis/Mpga-Mpgd_results_standard_m7.8.csv')


