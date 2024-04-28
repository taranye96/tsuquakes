#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:07:33 2023

@author: tnye
"""

import seaborn as sns
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from functools import reduce


#%%

sd_files_sm = sorted(glob('/Users/tnye/FakeQuakes/final_runs/no_soft_layer/stress_drop/*/flatfiles/residuals/*_sm.csv'))
sd_files_gnss = sorted(glob('/Users/tnye/FakeQuakes/final_runs/no_soft_layer/stress_drop/*/flatfiles/residuals/*_gnss.csv'))

IM_res_stress = np.array([])
stress = np.array([])
avg_res_stress = np.array([])
hf_res_stress = np.array([])

for i in range(len(sd_files_sm)):
    
    df_sm = pd.read_csv(sd_files_sm[i])
    df_gnss = pd.read_csv(sd_files_gnss[i])
    
    avg_res = []
    avg_hf_res = []
    for j in range(16):
        
        pga_res = df_sm.pgv_res.values[j*9:(j*9)+9]
        pgv_res = df_sm.pga_res.values[j*9:(j*9)+9]
        pgd_res = df_gnss.pgd_res.values[j*6:(j*6)+6]
        tpgd_res_ln = df_gnss.tPGD_res_ln.values[j*6:(j*6)+6]
        
        all_res = np.concatenate([pga_res,pgv_res,pgd_res,tpgd_res_ln])
        hf_res = np.concatenate([pga_res,pgv_res])
        
        avg_res.append(np.mean(all_res))
        avg_hf_res.append(np.mean(hf_res))
        
    avg_res_stress = np.append(avg_res_stress, avg_res)
    hf_res_stress = np.append(hf_res_stress, avg_hf_res)
    stress = np.append(stress, np.array([float(sd_files_sm[i].split('/')[-1].split('_')[0][2:])]*len(avg_res)))
    

rt_files_sm = sorted(glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/new_runs/risetime/*/flatfiles/residuals/*_sm.csv'))
rt_files_gnss = sorted(glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/new_runs/risetime/*/flatfiles/residuals/*_gnss.csv'))

rt = np.array([])
avg_res_rt = np.array([])
tpgd_res_rt = np.array([])

for i in range(len(rt_files_sm)):
    
    df_sm = pd.read_csv(rt_files_sm[i])
    df_gnss = pd.read_csv(rt_files_gnss[i])
    
    avg_res = []
    avg_tpgd_res = []
    for j in range(16):
        
        pga_res = df_sm.pgv_res.values[j*9:(j*9)+9]
        pgv_res = df_sm.pga_res.values[j*9:(j*9)+9]
        pgd_res = df_gnss.pgd_res.values[j*6:(j*6)+6]
        tpgd_res_ln = df_gnss.tPGD_res_ln.values[j*6:(j*6)+6]
        tpgd_res = df_gnss.tPGD_res_linear.values[j*6:(j*6)+6]
        
        all_res = np.concatenate([pga_res,pgv_res,pgd_res,tpgd_res_ln])
        
        avg_res.append(np.mean(all_res))
        avg_tpgd_res.append(np.mean(tpgd_res))
        
    avg_res_rt = np.append(avg_res_rt, avg_res)
    tpgd_res_rt = np.append(tpgd_res_rt, avg_tpgd_res)
    rt = np.append(rt, np.array([float(rt_files_sm[i].split('/')[-1].split('_')[0][2:].strip('x'))]*len(avg_res)))
    
vrupt_files_sm = sorted(glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/new_runs/vrupt/*/flatfiles/residuals/*_sm.csv'))
vrupt_files_gnss = sorted(glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/new_runs/vrupt/*/flatfiles/residuals/*_gnss.csv'))

vrupt = np.array([])
avg_res_vrupt = np.array([])
tpgd_res_vrupt = np.array([])

for i in range(len(vrupt_files_sm)):
    
    df_sm = pd.read_csv(vrupt_files_sm[i])
    df_gnss = pd.read_csv(vrupt_files_gnss[i])
    
    avg_res = []
    avg_tpgd_res = []
    for j in range(16):
        
        pga_res = df_sm.pgv_res.values[j*9:(j*9)+9]
        pgv_res = df_sm.pga_res.values[j*9:(j*9)+9]
        pgd_res = df_gnss.pgd_res.values[j*6:(j*6)+6]
        tpgd_res_ln = df_gnss.tPGD_res_ln.values[j*6:(j*6)+6]
        tpgd_res = df_gnss.tPGD_res_linear.values[j*6:(j*6)+6]
        
        all_res = np.concatenate([pga_res,pgv_res,pgd_res,tpgd_res_ln])
        
        avg_res.append(np.mean(all_res))
        avg_tpgd_res.append(np.mean(tpgd_res))
        
    avg_res_vrupt = np.append(avg_res_vrupt, avg_res)
    tpgd_res_vrupt = np.append(tpgd_res_vrupt, avg_tpgd_res)
    vrupt = np.append(vrupt, np.array([float(vrupt_files_sm[i].split('/')[-1].split('_')[0][2:])]*len(avg_res)))

sd_vrupt_files_sm = np.array(sorted([file for file in glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/new_runs/multi_param/*/flatfiles/residuals/*_sm.csv') if 'rt' not in file]))
sd_vrupt_files_gnss = np.array(sorted([file for file in glob('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/new_runs/multi_param/*/flatfiles/residuals/*_gnss.csv') if 'rt' not in file]))
sd_vrupt_files_sm = np.append(sd_vrupt_files_sm, vrupt_files_sm)
sd_vrupt_files_gnss = np.append(sd_vrupt_files_gnss, vrupt_files_gnss)

multi_sd = np.array([])
multi_vrupt = np.array([])
avg_res_sd_vrupt = np.array([])

for i in range(len(sd_vrupt_files_sm)):
    
    df_sm = pd.read_csv(sd_vrupt_files_sm[i])
    df_gnss = pd.read_csv(sd_vrupt_files_gnss[i])
    
    avg_res = []
    for j in range(2):
            pga_res = df_sm.pga_res.values[j*9:(j*9)+9]
            pgv_res = df_sm.pgv_res.values[j*9:(j*9)+9]
            pgd_res = df_gnss.pgd_res.values[j*6:(j*6)+6]
            tpgd_res_ln = df_gnss.tPGD_res_ln.values[j*6:(j*6)+6]
            
            all_res = np.concatenate([pga_res,pgv_res,pgd_res,tpgd_res_ln])
            avg_res.append(np.mean(all_res))
        
    avg_res_sd_vrupt = np.append(avg_res_sd_vrupt, avg_res)
    multi_vrupt = np.append(multi_vrupt, np.array([float(sd_vrupt_files_sm[i].split('/')[-1].split('_')[0][2:])]*len(avg_res)))
    multi_sd = np.append(multi_sd, np.array([float(sd_vrupt_files_sm[i].split('/')[-1].split('_')[0][2:])]*len(avg_res)))


#%%  stress drop residuals plot

x = stress
y = hf_res_stress

# Compute the conditional probability density of y given x
g = sns.jointplot(x=x, y=y, kind='kde', bw_adjust=0.5)

# Extract the density estimate (KDE plot) from the jointplot
kde = g.ax_joint.collections[0]

# Get the x and y values of the density estimate
x_density, y_density = kde.get_paths()[0].vertices.T


# Create an interpolation function from the density estimate data
interp_func = interp1d(y_density, x_density)
x_new = np.linspace(np.min(x),np.max(x),1000)

# Find the x value with the highest probability density of y=0
x_zero_prob = np.interp(0, y, x)

# kde = gaussian_kde(y)
# def weighted_mean(x, kde):
#     weights = kde(x)
#     return np.sum(weights * x) / np.sum(weights)

# # Calculate the weighted mean of x where y is closest to zero
# x_weighted_mean = weighted_mean(x, lambda y: kde(abs(y)))

# print("X value with the highest probability of Y=0: ", x_weighted_mean)

# Plot a vertical line at x_max_prob
# g.ax_joint.axvline(x=x_zero_prob, color='red')

# Set the x and y labels
g.set_axis_labels('Stress Drop', r'Average $\delta$',fontsize=12)
plt.subplots_adjust(left=0.15, bottom=0.1)
plt.savefig('/Users/tnye/tsuquakes/plots/gridsearch/stress_drop.png',dpi=300)


#%%  risetime residuals plot

x = rt
y = tpgd_res_rt
# y = avg_res_rt

# Compute the conditional probability density of y given x
g = sns.jointplot(x=x, y=y, kind='kde', bw_adjust=0.5)

# Extract the density estimate (KDE plot) from the jointplot
kde = g.ax_joint.collections[0]

# Get the x and y values of the density estimate
x_density, y_density = kde.get_paths()[0].vertices.T


# Create an interpolation function from the density estimate data
interp_func = interp1d(y_density, x_density)
x_new = np.linspace(np.min(x),np.max(x),1000)

# Find the x value with the highest probability density of y=0
x_zero_prob = np.interp(0, y, x)

# Set the x and y labels
g.set_axis_labels('Risetime Multiplication Factor', r'Average $\delta$',fontsize=12)
plt.subplots_adjust(left=0.15, bottom=0.1)
plt.savefig('/Users/tnye/tsuquakes/plots/gridsearch/rise_time.png',dpi=300)


#%%  vrupt residuals plot

x = vrupt
# y = avg_res_vrupt
y = tpgd_res_vrupt

# Compute the conditional probability density of y given x
g = sns.jointplot(x=x, y=y, kind='kde', bw_adjust=0.5)

# Extract the density estimate (KDE plot) from the jointplot
kde = g.ax_joint.collections[0]

# Get the x and y values of the density estimate
x_density, y_density = kde.get_paths()[0].vertices.T


# Create an interpolation function from the density estimate data
interp_func = interp1d(y_density, x_density)
x_new = np.linspace(np.min(x),np.max(x),1000)

# Find the x value with the highest probability density of y=0
x_zero_prob = np.interp(0, y, x)

# Set the x and y labels
g.set_axis_labels('Shallow Shear Wave Fraction', r'Average $\delta$',fontsize=12)
plt.subplots_adjust(left=0.15, bottom=0.1)
plt.savefig('/Users/tnye/tsuquakes/plots/gridsearch/rupture_velocity.png',dpi=300)


#%%  stress drop vrupt residuals plot

x = sd_vrupt
y = avg_res_sd_vrupt

# Compute the conditional probability density of y given x
g = sns.jointplot(x=x, y=y, kind='kde', bw_adjust=0.5)

# Extract the density estimate (KDE plot) from the jointplot
kde = g.ax_joint.collections[0] 

# Get the x and y values of the density estimate
x_density, y_density = kde.get_paths()[0].vertices.T


# Create an interpolation function from the density estimate data
interp_func = interp1d(y_density, x_density)
x_new = np.linspace(np.min(x),np.max(x),1000)

# Find the x value with the highest probability density of y=0
x_zero_prob = np.interp(0, y, x)

# Set the x and y labels
g.set_axis_labels('Shallow Shear Wave Fraction', r'Average $\delta$; $\sigma$=1MPa',fontsize=12)
plt.subplots_adjust(left=0.15, bottom=0.1)

# Show the plot
plt.show()


#%%

import matplotlib.colors as colors
import matplotlib.cm as cmx

# Set up colormap
viridis = plt.get_cmap('viridis_r') 
cNorm  = colors.Normalize(vmin=-0.6, vmax=0.6)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=viridis)

scatter_colors = []
for val in avg_res_sd_vrupt:
    scatter_colors.append(scalarMap.to_rgba(val))

fig, ax = plt.subplots(1,1)
plt.scatter(multi_sd, multi_vrupt, c=scatter_colors)
ax.set_xlabel('Stress Drop')
ax.set_ylabel('Shallow Shear Wave Fraction')


