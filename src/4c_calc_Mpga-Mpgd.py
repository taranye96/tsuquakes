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

# Read in GMM residual flatfile
df = pd.read_csv('/Users/tnye/tsuquakes/realtime_analysis/PGA_GMM_residuals.csv')

# Read in Mpgd flatfile
Mpgd_df = pd.read_csv('/Users/tnye/tsuquakes/realtime_analysis/Mpgd.csv')

# Define trial magnitudes
trial_M = np.arange(5.5,8.5,0.1)

# Get list of parameters and runs
params = np.unique(df['Parameters'])
runs = np.unique(df['Run'])

#%%

# Initialize lists
param_list = []
run_list = []
Mpga_list = []
Mpgd_list = []
Mdiff_list = []
param_res = []

# Loop over paramter combinations
for param in params:
    
    # Loop over runs
    for run in runs:
        
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
        
        # Append values to lists
        param_list.append(param)
        run_list.append(run)
        Mpga = trial_M[np.where(np.abs(p_val_list-0.05) == np.min(np.abs(p_val_list-0.05)))][0]
        Mpga_list.append(Mpga)
        Mpgd_list.append(Mpgd[0])
        Mdiff_list.append(Mpga - Mpgd[0])
        
        # # Plot results
        # fig, ax = plt.subplots(1,1,figsize=(5,2.75))
        # ax.semilogy(trial_M,p_val_list)
        # ax.axhline(0.05,c='gray',ls='--',lw=1,label='significance level')
        # ax.axvline(Mpgd,c='goldenrod',lw=0.8,label=r'$M_{PGD}$')
        # ax.axvline(Mpga,c='red',lw=0.8,label=r'$M_{PGA}$')
        # ax.set_xlabel('Magnitude')
        # ax.set_ylabel('P-value')
        # ax.legend()
        # ax.set_xlim(5.5,8.4)
        # plt.subplots_adjust(left=0.15,bottom=0.175,top=0.95,right=0.95,hspace=0.35)
        # plt.savefig(f'/Users/tnye/tsuquakes/realtime_analysis/Mpga_plots/{param}_{run}.png',dpi=300)
        # plt.close()

data = {'Parameters':param_list,'Run':run_list,'Mpgd':Mpgd_list,'Mpga':Mpga_list,
        'Mpga-Mpgd':Mdiff_list}
Mdiff_df = pd.DataFrame(data)
Mdiff_df.to_csv('/Users/tnye/tsuquakes/realtime_analysis/Mpga-Mpgd_results_m7.8.csv')

#%%

# compute the statistical difference between the sets of data

sahakian_df1 = pd.read_csv('/Users/tnye/tsuquakes/files/sahakian2019/mag_differences_KStests_finalmag.csv')
sahakian_df2 = pd.read_csv('/Users/tnye/tsuquakes/files/sahakian2019/mag_differences_KStests_pgdmag.csv')
obs_M = sahakian_df1.mw.values
obs_Mdiff = sahakian_df2.magdiff_KSpvalue.values
obs_std = np.std(obs_Mdiff)

mentawai_distribution = np.random.normal(-1.25, 0.232140994565855, 180)

Mdiff_df = pd.read_csv('/Users/tnye/tsuquakes/realtime_analysis/Mpga-Mpgd_results_m7.8.csv')

val1 = Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.2_sf0.41_sd1.0')[0]]
val2 = Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.742_sf0.422_sd1.428')[0]]
val3 = Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.2_sf0.41_sd2.0')[0]]
val4 = Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt2.0_sf0.42_sd1.0')[0]]
val5 = Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt1.422_sf0.395_sd1.428')[0]]
val6 = Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='rt2.0_sf0.42_sd2.0')[0]]
std_data = Mdiff_df['Mpga-Mpgd'].values[np.where(Mdiff_df['Parameters']=='standard')[0]]

# val1 = Mdiff_df['Mpga-Mpgd'].values[np.where((Mdiff_df['Parameters']=='rt1.2_sf0.41_sd1.0')&(Mdiff_df['Run']=='mentawai.000000'))[0]]
# val2 = Mdiff_df['Mpga-Mpgd'].values[np.where((Mdiff_df['Parameters']=='rt1.742_sf0.422_sd1.428')&(Mdiff_df['Run']=='mentawai.000000'))[0]]
# val3 = Mdiff_df['Mpga-Mpgd'].values[np.where((Mdiff_df['Parameters']=='rt1.2_sf0.41_sd2.0')&(Mdiff_df['Run']=='mentawai.000000'))[0]]
# val4 = Mdiff_df['Mpga-Mpgd'].values[np.where((Mdiff_df['Parameters']=='rt2.0_sf0.42_sd1.0')&(Mdiff_df['Run']=='mentawai.000000'))[0]]
# val5 = Mdiff_df['Mpga-Mpgd'].values[np.where((Mdiff_df['Parameters']=='rt1.422_sf0.395_sd1.428')&(Mdiff_df['Run']=='mentawai.000000'))[0]]
# val6 = Mdiff_df['Mpga-Mpgd'].values[np.where((Mdiff_df['Parameters']=='rt2.0_sf0.42_sd2.0')&(Mdiff_df['Run']=='mentawai.000000'))[0]]
# std_data = Mdiff_df['Mpga-Mpgd'].values[np.where((Mdiff_df['Parameters']=='standard')&(Mdiff_df['Run']=='mentawai.000000'))[0]]
                                   
# tse_data = [np.mean(val1),np.mean(val2),np.mean(val3),np.mean(val4),np.mean(val5),np.mean(val6)]
concatenated_array = np.concatenate((val1, val2, val3, val4, val5, val6))
tse_data = concatenated_array.flatten()

statistic, p_value = ks_2samp(tse_data, obs_Mdiff, mode='asymp')

statistic, p_value = ks_2samp(tse_data, std_data, mode='asymp')

statistic, p_value = ks_2samp(obs_Mdiff, std_data, mode='asymp')

statistic, p_value = ks_2samp(tse_data, mentawai_distribution, mode='asymp')

plt.figure()
plt.hist(obs_Mdiff,alpha=0.5,label='Observed')
plt.hist(std_data,alpha=0.5,laebl='Standard Sim')
plt.hist(tse_data,alpha=0.5,label='TsE Sim')
plt.legend()

print('d-statistic, p-value')
print(f"TsE Sim.–Mentawai Dist.: {ks_2samp(tse_data, mentawai_distribution, mode='asymp')}")
print(f"TsE Sim.–Observed Events: {ks_2samp(tse_data, obs_Mdiff, mode='asymp')}")
print(f"Standard Sim.–Observed Events: {ks_2samp(std_data, obs_Mdiff, mode='asymp')}")
print(f"Standard Sim.–TsE Sim.: {ks_2samp(std_data, tse_data, mode='asymp')}")






