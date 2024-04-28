#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 23:12:47 2021

@author: tnye
"""

###############################################################################
# Script that plots PGD residuals for the FakeQuakes test making runs in
# magnitude bins ranging from M7.7-M7.9.
###############################################################################

# Imports 
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns


rupt_files = sorted(glob('/Users/tnye/FakeQuakes/parameters/test/mag_test/disp/output/ruptures/*.rupt'))
    
# Observed values
obs_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/obs_IMs.csv')
obs_pgd = np.array(obs_df['pgd'])

# Set up empty lists to store the project names in for the disp IMs
project_list = []
# Set up empty lists to store the current project residuals in
pgd_res_list = []

for i in range(len(rupt_files)):
    
    run = rupt_files[i].split('/')[-1].rsplit('.', 1)[0]
    
    # Synthetic values
    syn_df = pd.read_csv(f'/Users/tnye/FakeQuakes/parameters/test/mag_test/flatfiles/IMs/{run}.csv')
    syn_pgd = np.array(syn_df['pgd'])
    
    # calc res
    pgd_res = np.log(obs_pgd)[:13] - np.log(syn_pgd)
    pgd_res_list.append(pgd_res)

    if i < 4:
        # project_list.append('M7.7')
        project_list += 13 * ['M7.7']
    elif i >= 4 and i < 8:
        # project_list.append('M7.75')
        project_list += 13 * ['M7.75']
    elif i >= 8 and i < 12:
        # project_list.append('M7.8')
        project_list += 13 * ['M7.8']
    elif i >= 12 and i < 16:
        # project_list.append('M7.85')
        project_list += 13 * ['M7.85']
    else:
        # project_list.append('M7.9')
        project_list += 13 * ['M7.9']
        

############################## Plot PGD Residuals #############################
    
# Set up Figure
fig, ax = plt.subplots(1,1)

# PGD subplot
pgd_res_list = [val for sublist in pgd_res_list for val in sublist]
pgd_data = {'Parameter':project_list, 'ln PGD (m)':pgd_res_list}
pgd_df = pd.DataFrame(data=pgd_data)       
        
sns.boxplot(x='Parameter', y='ln PGD (m)', data=pgd_df, showfliers=True, boxprops=dict(alpha=.3),
            ax=ax)
yabs_max = abs(max(ax.get_ylim(), key=abs))
ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
ax.set_title('PGD')
ax.axhline(0, ls='--')

plt.savefig('/Users/tnye/tsuquakes/plots/residuals/test/mag_pgd.png', dpi=300)
plt.close()
