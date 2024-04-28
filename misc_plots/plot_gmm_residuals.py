#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 22:25:48 2022

@author: tnye
"""

###############################################################################
# Script that plots GMM residuals using a new Q model. 
###############################################################################

# Imports
from os import path, makedirs
import numpy as np
import pandas as pd
from os import path, makedirs
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt

batch = 'mentawai'

ruptures = np.genfromtxt(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.87/standard_parameters/data/ruptures.list',dtype=str)

pga_res = np.array([])
pgv_res = np.array([])
mag_range_list = np.array([])
mag_pgv = np.array([])
rrup_range = []
rrup_pgv = []
gmm = np.array([])
gmm_pgv = np.array([])

for rupture in ruptures:
    
    run = rupture.replace('.rupt','')

    try:
        df = pd.read_csv('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.87/standard_parameters/flatfiles/residuals/HF_residuals.csv')
        
        # NGA-Sub
        pga_res = np.append(pga_res, np.array(df['lnPGA_res_NGASub']))
        pgv_res = np.append(pgv_res, np.array(df['lnPGV_res_NGASub']))
        gmm = np.append(gmm, ['NGA-Sub']*len(df))
        gmm_pgv = np.append(gmm_pgv, ['NGA-Sub']*len(df))
        
        rrup = np.array(df['Rrup(km)'])
        for r in rrup:
            if r < 100:
                rrup_range.append(int(50))
                rrup_pgv.append(int(50))
            elif (r >= 100) and (r < 200):
                rrup_range.append(int(150))
                rrup_pgv.append(int(150))
            elif (r >= 200) and (r < 300):
                rrup_range.append(int(250))
                rrup_pgv.append(int(250))
            elif (r >= 300) and (r < 400):
                rrup_range.append(int(350))
                rrup_pgv.append(int(350))
            elif (r >= 400) and (r < 500):
                rrup_range.append(int(450))
                rrup_pgv.append(int(450))
            elif (r >= 500) and (r < 600):
                rrup_range.append(int(550))
                rrup_pgv.append(int(550))
            elif (r >= 600) and (r < 700):
                rrup_range.append(int(650))
                rrup_pgv.append(int(650))
            elif (r >= 700) and (r < 800):
                rrup_range.append(int(750))
                rrup_pgv.append(int(750))
            elif (r >= 800) and (r < 900):
                rrup_range.append(int(850))
                rrup_pgv.append(int(850))
        
        # BCHydro
        pga_res = np.append(pga_res, np.array(df['lnPGA_res_BCHydro']))
        gmm = np.append(gmm, ['BCHydro']*len(df))
        
        rrup = np.array(df['Rrup(km)'])
        for r in rrup:
            if r < 100:
                rrup_range.append(int(50))
            elif (r >= 100) and (r < 200):
                rrup_range.append(int(150))
            elif (r >= 200) and (r < 300):
                rrup_range.append(int(250))
            elif (r >= 300) and (r < 400):
                rrup_range.append(int(350))
            elif (r >= 400) and (r < 500):
                rrup_range.append(int(450))
            elif (r >= 500) and (r < 600):
                rrup_range.append(int(550))
            elif (r >= 600) and (r < 700):
                rrup_range.append(int(650))
            elif (r >= 700) and (r < 800):
                rrup_range.append(int(750))
            elif (r >= 800) and (r < 900):
                rrup_range.append(int(850))
    except:
        continue


# Plot rrup residuals
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Helvetica'
fig, axs = plt.subplots(1,2,figsize=(10,4))
graph = sns.boxplot(x=rrup_range, y=pga_res, hue=gmm, ax=axs[0])
graph.axhline(0, ls='--', color='k', lw=1)
axs[0].set_xlabel(r'$R_{rup}$ (km)')
axs[0].set_ylabel('ln PGA residual (m/s/s)')
graph1 = sns.boxplot(x=rrup_pgv, y=pgv_res, hue=gmm_pgv, ax=axs[1])
graph1.axhline(0, ls='--', color='k', lw=1)
axs[1].set_xlabel(r'$R_{rup}$ (km)')
axs[1].set_ylabel('ln PGV residual (m/s)')
plt.subplots_adjust(wspace=0.25,hspace=0.2,right=0.97,bottom=0.12,left=0.07,top=0.94)
plt.savefig('/Users/tnye/tsuquakes/plots/q_param_test/Qmodel_vs_GMM/newQ_m7.87_HF_rrup.png',dpi=300)
