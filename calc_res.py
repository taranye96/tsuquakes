#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:54:47 2020

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project = '2min_16cores/'

# runs list

runs = ['run.000000', 'run.000001', 'run.000002', 'run.000003', 'run.000004',
        'run.000005', 'run.000006', 'run.000007', 'run.000008', 'run.000009',
        'run.000010', 'run.000011']

# Observed df
obs_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/spec_test/2min_16cores/obs.csv')


# Observed values
obs_pgd = np.array(obs_df['pgd'])
obs_pga = np.array(obs_df['pga'])
obs_pgv = np.array(obs_df['pgv'])
# obs_tPGD = np.array(obs_df['tPGD'])
# obs_tPGA = np.array(obs_df['tPGA'])
obs_spectra = np.array(obs_df.iloc[:,24:248])

# Get rid of duplicates and NaNs
obs_pgd = [x for x in obs_pgd if str(x) != 'nan']
obs_pgd = np.delete(obs_pgd, slice(None, None, 3))
obs_pgd = np.delete(obs_pgd, slice(None, None, 2))

obs_pga = [x for x in obs_pga if str(x) != 'nan']
obs_pga = np.delete(obs_pga, slice(None, None, 3))
obs_pga = np.delete(obs_pga, slice(None, None, 2))

obs_pgv = [x for x in obs_pgv if str(x) != 'nan']
obs_pgv = np.delete(obs_pgv, slice(None, None, 3))
obs_pgv = np.delete(obs_pgv, slice(None, None, 2))

pgd_run_list = []
pgd_res_list = []

pga_run_list = []
pga_res_list = []

pgv_run_list = []
pgv_res_list = []



for i, run in enumerate(runs):
    syn_df = pd.read_csv(('/Users/tnye/tsuquakes/flatfiles/' + project + '/' + run + '.csv'))
    syn_pgd = np.array(syn_df['pgd'])
    syn_pga = np.array(syn_df['pga'])
    syn_pgv = np.array(syn_df['pgv'])
    
    # Get rid of duplicates and NaNs
    syn_pgd = [x for x in syn_pgd if str(x) != 'nan']
    syn_pgd = np.delete(syn_pgd, slice(None, None, 3))
    syn_pgd = np.delete(syn_pgd, slice(None, None, 2))
    
    syn_pga = [x for x in syn_pga if str(x) != 'nan']
    syn_pga = np.delete(syn_pga, slice(None, None, 3))
    syn_pga = np.delete(syn_pga, slice(None, None, 2))
    
    syn_pgv = [x for x in syn_pgv if str(x) != 'nan']
    syn_pgv = np.delete(syn_pgv, slice(None, None, 3))
    syn_pgv = np.delete(syn_pgv, slice(None, None, 2))
    
    # calc res
    pgd_res = obs_pgd - syn_pgd
    pgd_res_list.append(pgd_res)
    
    pga_res = obs_pga - syn_pga
    pga_res_list.append(pga_res)
    
    pgv_res = obs_pgv - syn_pgv
    pgv_res_list.append(pgv_res)
    
    # get run number
    for j in range(len(pgd_res)):
        pgd_run_list.append(i)
    for j in range(len(pga_res)):
        pga_run_list.append(i)
    for j in range(len(pgv_res)):
        pgv_run_list.append(i)


# PGD boxplot
pgd_res_list = [val for sublist in pgd_res_list for val in sublist]
pgd_data = {'Run#':pgd_run_list, 'Residual(m)':pgd_res_list}
pgd_df = pd.DataFrame(data=pgd_data)       
        
ax = sns.catplot(x='Run#', y='Residual(m)', data=pgd_df)
ax = sns.boxplot(x='Run#', y='Residual(m)', data=pgd_df, boxprops=dict(alpha=.3))
ax.set_title('PGD')
ax.axhline(0, ls='--')
ax.set(ylim=(-0.8, 0.8))

figpath = '/Users/tnye/tsuquakes/plots/residuals/' + project + 'pgd_res.png'
plt.savefig(figpath, bbox_inches='tight', dpi=300)
plt.close()


# PGA boxplot
pga_res_list = [val for sublist in pga_res_list for val in sublist]
pga_data = {'Run#':pga_run_list, 'Residual(m/s/s)':pga_res_list}
pga_df = pd.DataFrame(data=pga_data)
                   
ax = sns.catplot(x='Run#', y='Residual(m/s/s)', data=pga_df)
ax = sns.boxplot(x='Run#', y='Residual(m/s/s)', data=pga_df, boxprops=dict(alpha=.3))
ax.set_title('PGA')
ax.axhline(0, ls='--')
ax.set(ylim=(-.8, .8))

figpath = '/Users/tnye/tsuquakes/plots/residuals/' + project + 'pga_res.png'
plt.savefig(figpath, bbox_inches='tight', dpi=300)
plt.close()


# PGV boxplot
pgv_res_list = [val for sublist in pgv_res_list for val in sublist]
pgv_data = {'Run#':pgv_run_list, 'Residual(m/s)':pgv_res_list}
pgv_df = pd.DataFrame(data=pgv_data)
                   
ax = sns.catplot(x='Run#', y='Residual(m/s)', data=pgv_df)
ax = sns.boxplot(x='Run#', y='Residual(m/s)', data=pgv_df, boxprops=dict(alpha=.3))
ax.set_title('PGV')
ax.axhline(0, ls='--')
ax.set(ylim=(-.8, .8))

figpath = '/Users/tnye/tsuquakes/plots/residuals/' + project + 'pgv_res.png'
plt.savefig(figpath, bbox_inches='tight', dpi=300)
plt.close()





            

