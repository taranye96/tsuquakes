#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:24:10 2021

@author: tnye
"""

###############################################################################
# Script that performs the stress drop vs rupture velocity regression to obtain
# a mathematical way to co-vary the two parameters.
###############################################################################

# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Reaad in dataframes
vrupt_df = pd.read_csv('/Users/tnye/tsuquakes/data/misc/melgar_hayes2017.csv')
sd_df = pd.read_csv('/Users/tnye/tsuquakes/data/misc/ye2016.csv')

# Obtain event info from dataframes
vrupt_datetimes = np.array(vrupt_df['origin time'])
vrupt_USGSID = np.array(vrupt_df['#USGS ID'])
vrupt_types = np.array(vrupt_df['type'])
vrupt_mag = np.array(vrupt_df['Mw'])
vrupt_depths = np.array(vrupt_df['depth(km)'])
sd_dates = np.array(sd_df['Date'])
sd_times = np.array(sd_df['Time'])

# Obtain risetime and stress drops
all_vrupt = np.array(vrupt_df['rupture vel(km/s)'])
all_apparent_stress = np.array(sd_df['σa(MPa)'])

# Initialize origin lists
vrupt_origins = np.array([])
sd_origins = np.array([])

# Loop through rupture velocity df
for origin in vrupt_datetimes:
    short_orig = origin.split('.')[0]
    new_orig = short_orig.split(':')[0] + ':' + short_orig.split(':')[1]
    vrupt_origins = np.append(vrupt_origins, new_orig)

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
vrupt = []
apparent_stress = [] 
common_events = []
common_depths = []
common_mag = []
common_IDs = []
color = []

# tsq_ID = ['p000jqvm', 'p000fn2b', 'p000a45f', 'p000ah4q', 'p000hnj4', 'c000f1s0', 'p000fjta']
tsq_ID = ['p000ensm', 'p000hnj4', 'p0007dmb', 'p0006djk']
for i, element in enumerate(vrupt_origins):
    
    # Only select megathrust events
    if vrupt_types[i] == "i":
        
        if element in sd_origins:
            
            common_events.append(element.split('T')[0])
            common_depths.append(vrupt_depths[i])
            common_mag.append(vrupt_mag[i])
            common_IDs.append(vrupt_USGSID[i])
            if vrupt_USGSID[i] in tsq_ID:
                color.append(1)
            else:
                color.append(0)
            
            # Find indexes of rise times and stress drops for common events
            vrupt_ind = i
            sd_ind = np.where(sd_origins == element)[0][0]
            
            # Find rise times and stress drops for common events
            vrupt.append(all_vrupt[vrupt_ind])
            apparent_stress.append(all_apparent_stress[sd_ind])

# Convert form apparent stress to stress drop
stress_drop = []
for stress in apparent_stress:
    sd = 4.3*stress
    stress_drop.append(sd)


########################### Perform regression  #########################

# Line of best fit
coefficients = np.polyfit(np.log10(stress_drop), np.log10(vrupt), 1)
polynomial = np.poly1d(coefficients)
log10_y_fit = polynomial(np.log10(stress_drop))

# Calc R^2
correlation_matrix = np.corrcoef(np.log10(stress_drop), np.log10(vrupt))
correlation_xy = correlation_matrix[0,1]
r2 = correlation_xy**2

print(f'log10(Rupture Velocity) = log10(stress drop) * {round(coefficients[0],2)} + {round(coefficients[1],2)}')
print(r'R2 = %.2f' % (r2, ))


############################### Make Plot #################################

fig = plt.figure(figsize=(7,5))
ax = plt.gca()
color_map = plt.cm.get_cmap('winter').reversed()
im = ax.scatter(stress_drop, vrupt, c=color, cmap=color_map)
plt.plot(stress_drop, 10**log10_y_fit)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Stress Drop (MPa)')
ax.set_ylabel('Vrupt (km/s)')

# Save fig
plt.savefig(f'/Users/tnye/tsuquakes/plots/parameter_correlations/stressdrop_vrupt.png', dpi=300)


