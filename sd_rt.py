#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:29:40 2020

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Reaad in dataframes
rt_df = pd.read_csv('/Users/tnye/tsuquakes/data/misc/melgar_hayes2017.csv')
sd_df = pd.read_csv('/Users/tnye/tsuquakes/data/misc/ye2016.csv')

# Obtain origin times form the dfs
rt_datetimes = np.array(rt_df['origin time'])
rt_USGSID = np.array(rt_df['#USGS ID'])
rt_types = np.array(rt_df['type'])
rt_mag = np.array(rt_df['Mw'])
rt_depths = np.array(rt_df['depth(km)'])
sd_dates = np.array(sd_df['Date'])
sd_times = np.array(sd_df['Time'])

# Obtain rise time and stress drops
all_rise_times = np.array(rt_df['rise time(s)'])
all_apparent_stress = np.array(sd_df['σa(MPa)'])
all_energy_stress2 = np.array(sd_df['ΔσE2.0(MPa)'])
all_energy_stress25 = np.array(sd_df['ΔσE2.0(MPa)'])
all_energy_stress3 = np.array(sd_df['ΔσE3.0(MPa)'])

# Initialize origin lists
rt_origins = np.array([])
sd_origins = np.array([])

# Loop through rise time df
for origin in rt_datetimes:
    short_orig = origin.split('.')[0]
    new_orig = short_orig.split(':')[0] + ':' + short_orig.split(':')[1]
    rt_origins = np.append(rt_origins, new_orig)

# Loop through stress drop df
for i, date in enumerate(sd_dates):
    yyyy = date.split('-')[0]
    mth = date.split('-')[1]
    dd = date.split('-')[2]
    hr = sd_times[i].split(':')[0]
    mm = sd_times[i].split(':')[1]
    
    origin = yyyy + '-' + mth + '-' + dd + 'T' + hr + ':' + mm
    sd_origins = np.append(sd_origins, origin)

# Find common events between both datasets
rise_times = []
apparent_stress = [] 
energy_stress2 = []
energy_stress25 = []
energy_stress3 = []
common_events = []
common_depths = []
common_mag = []
common_IDs = []
for i, element in enumerate(rt_origins):
    
    # Only select megathrust events
    if rt_types[i] == "i":
        
        if element in sd_origins:
            
            common_events.append(element.split('T')[0])
            common_depths.append(rt_depths[i])
            common_mag.append(rt_mag[i])
            common_IDs.append(rt_USGSID[i])
            
            # Find indexes of rise times and stress drops for common events
            rt_ind = i
            sd_ind = np.where(sd_origins == element)[0][0]
            
            # Find rise times and stress drops for common events
            rise_times.append(all_rise_times[rt_ind])
            apparent_stress.append(all_apparent_stress[sd_ind])
            energy_stress2.append(all_energy_stress2[sd_ind])
            energy_stress25.append(all_energy_stress25[sd_ind])
            energy_stress3.append(all_energy_stress3[sd_ind])


###################### Plot stress drop vs rise time ##########################

stress_types = [apparent_stress, energy_stress2, energy_stress25, energy_stress3]

for stress in stress_types:
    
    ########################### Find line of best fit #########################
    coefficients = np.polyfit(np.log10(rise_times), np.log10(stress), 1)
    polynomial = np.poly1d(coefficients)
    log10_y_fit = polynomial(np.log10(rise_times))
    
    # Calc R^2
    correlation_matrix = np.corrcoef(np.log10(rise_times), np.log10(stress))
    correlation_xy = correlation_matrix[0,1]
    r2 = correlation_xy**2
    
    r2 = r2_score(np.log10(stress), log10_y_fit)
    
    ############################### Make Plot #################################
    if stress == apparent_stress:
        ylabel = 'Apparent Stress(MPa)'
        figname = 'RTvsAS.png'
    elif stress == energy_stress2:
        ylabel = 'Energy-Based Stress Drop 2.0(MPa)'
        figname = 'RTvsES2.png'
    elif stress == energy_stress25:
        ylabel == 'Energy-Based Stress Drop 2.5(MPa)'
        figname = 'RTvsES2_5.png'
    elif stress == energy_stress3:
        ylabel == 'Energy-Based Stress Drop 3.0(MPa)'
        figname = 'RTvsES3.ong'
        
    x = np.linspace(4,20)
    fig = plt.figure(figsize=(10,15))
    ax = plt.gca()
    color_map = plt.cm.get_cmap('plasma').reversed()
    im = ax.scatter(rise_times, stress, c=common_depths, cmap=color_map)
    for i, event in enumerate(common_events):
        ax.annotate(f'{common_IDs[i]}', (rise_times[i], stress[i]), size=6)
    plt.plot(rise_times, 10**log10_y_fit)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rise Time (s)')
    ax.set_ylabel(ylabel)
    
    # Set up text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        f'log10(sd) = log10(rt) * {coefficients[0]} + {coefficients[1]}',
        r'R2 = %.2f' % (r2, )))
    plt.text(9, 1.7, textstr, fontsize=8, bbox=props)
    
    # Set up colorbar
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='2.5%', pad=0.8, pack_start=True)
    fig.add_axes(cax)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Depth(km)')
    cbar.ax.invert_yaxis()
    
    # Save fig
    plt.savefig(f'/Users/tnye/tsuquakes/plots/misc/{figname}.png', dpi=300)
    plt.close()
