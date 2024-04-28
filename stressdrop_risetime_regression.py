#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:29:40 2020

@author: tnye
"""

###############################################################################
# Script that performs the stress drop vs risetime regression to obtain
# a mathematical way to co-vary the two parameters.
###############################################################################

# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Reaad in dataframes
rt_df = pd.read_csv('/Users/tnye/tsuquakes/data/misc/melgar_hayes2017.csv')
sd_df = pd.read_csv('/Users/tnye/tsuquakes/data/misc/ye2016.csv')

# Obtain event info from the dataframes
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
common_events = []
common_depths = []
common_mag = []
common_IDs = []
color = []

# tsq_ID = ['p000jqvm', 'p000fn2b', 'p000a45f', 'p000ah4q', 'p000hnj4', 'c000f1s0', 'p000fjta']
tsq_ID = ['p000ensm', 'p000hnj4', 'p0007dmb', 'p0006djk']
for i, element in enumerate(rt_origins):
    
    # Only select megathrust events
    if rt_types[i] == "i":
        
        if element in sd_origins:
            
            common_events.append(element.split('T')[0])
            common_depths.append(rt_depths[i])
            common_mag.append(rt_mag[i])
            common_IDs.append(rt_USGSID[i])
            if rt_USGSID[i] in tsq_ID:
                color.append(1)
            else:
                color.append(0)
            
            # Find indexes of rise times and stress drops for common events
            rt_ind = i
            sd_ind = np.where(sd_origins == element)[0][0]
            
            # Find rise times and stress drops for common events
            rise_times.append(all_rise_times[rt_ind])
            apparent_stress.append(all_apparent_stress[sd_ind])

# Convert form apparent stress to stress drop
stress_drop = []
for stress in apparent_stress:
    sd = 4.3*stress
    stress_drop.append(sd)


###################### Plot stress drop vs rise time ##########################

    
########################### Find line of best fit #########################

coefficients = np.polyfit(np.log10(stress_drop), rise_times, 1)
polynomial = np.poly1d(coefficients)
y_fit = polynomial(np.log10(stress_drop))

# Calc R^2
correlation_matrix = np.corrcoef(np.log10(stress_drop), rise_times)
correlation_xy = correlation_matrix[0,1]
r2 = correlation_xy**2

r2 = r2_score(rise_times, y_fit)

fig = plt.figure(figsize=(7,5))
ax = plt.gca()
color_map = plt.cm.get_cmap('winter').reversed()
im = ax.scatter(stress_drop, rise_times, c=color, cmap=color_map)
plt.plot(stress_drop, y_fit)
ax.set_xscale('log')
ax.set_xlabel('Stress Drop (MPa)')
ax.set_ylabel('Risetime (s)')

# Save fig
plt.savefig(f'/Users/tnye/tsuquakes/plots/parameter_correlations/stressdrop_risetime.png', dpi=300)


# Set up text box
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# textstr = '\n'.join((
#     f'log10(sd) = log10(rt) * {round(coefficients[0],2)} + {round(coefficients[1],2)}',
#     r'R2 = %.2f' % (r2, )))
# plt.text(9, 1.7, textstr, fontsize=8, bbox=props)
print(f'Rise Time = log10(Stress Drop) * {round(coefficients[0],2)} + {round(coefficients[1],2)}')
print(r'R2 = %.2f' % (r2, ))


############################### Make Plot #################################

# fig = plt.figure(figsize=(7,5))
# ax = plt.gca()
# color_map = plt.cm.get_cmap('winter').reversed()
# im = ax.scatter(stress_drop, rise_times, c=color, cmap=color_map)
# plt.plot(stress_drop, 10**log10_y_fit)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlabel('Stress Drop (MPa)')
# ax.set_ylabel('Risetime (s)')

# # Set up text box
# # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# # textstr = '\n'.join((
# #     f'log10(sd) = log10(rt) * {round(coefficients[0],2)} + {round(coefficients[1],2)}',
# #     r'R2 = %.2f' % (r2, )))
# # plt.text(9, 1.7, textstr, fontsize=8, bbox=props)
# print(f'log10(Rise Time) = log10(Stress Drop) * {round(coefficients[0],2)} + {round(coefficients[1],2)}')
# print(r'R2 = %.2f' % (r2, ))

# # Save fig
# # plt.savefig(f'/Users/tnye/tsuquakes/plots/parameter_correlations/SDvRT.png', dpi=300)
# # plt.close()
