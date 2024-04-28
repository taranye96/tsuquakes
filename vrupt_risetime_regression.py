#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:10:39 2021

@author: tnye
"""

###############################################################################
# Script that performs the risetime vs rupture velocity regression to obtain
# a mathematical way to co-vary the two parameters.
###############################################################################

# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


########################### Assemble dataset #########################

# Reaad in dataframe with values from real events
df = pd.read_csv('/Users/tnye/tsuquakes/data/misc/melgar_hayes2017.csv')

# Obtain event IDs
USGSID = np.array(df['#USGS ID'])

# Obtain event types
types = np.array(df['type'])

# Obtain rise time and rupture velocities
all_rise_times = np.array(df['rise time(s)'])
all_vrupt = np.array(df['rupture vel(km/s)'])

# Separate tsunami earthquakes from dataset and give them a different color
color = []
# tsq_ID = ['p000jqvm', 'p000fn2b', 'p000a45f', 'p000ah4q', 'p000hnj4', 'c000f1s0', 'p000fjta']
tsq_ID = ['p000ensm', 'p000hnj4', 'p0007dmb', 'p0006djk']
rise_times = []
vrupt = []
for i in range(len(df)):
    
    # Only select megathrust events
    if types[i] == "i":
            
        if USGSID[i] in tsq_ID:
            color.append(1)
        else:
            color.append(0)
        
        # Append only thrust events
        rise_times.append(all_rise_times[i])
        vrupt.append(all_vrupt[i])

    
########################### Perform regression  #########################

# Line of best fit
coefficients = np.polyfit(np.log10(vrupt), np.log10(rise_times), 1)
polynomial = np.poly1d(coefficients)
log10_y_fit = polynomial(np.log10(vrupt))

# Calc R^2
correlation_matrix = np.corrcoef(np.log10(vrupt), np.log10(rise_times))
correlation_xy = correlation_matrix[0,1]
r2 = correlation_xy**2


############################### Make Plot #################################

fig = plt.figure(figsize=(7,5))
ax = plt.gca()
color_map = plt.cm.get_cmap('winter').reversed()
im = ax.scatter(vrupt, rise_times, c=color, cmap=color_map)
plt.plot(vrupt, 10**log10_y_fit)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Rupture Velocity (km/s)')
ax.set_ylabel('Risetime (s)')

print(f'log10(Risetime) = log10(Vrupt) * {round(coefficients[0],2)} + {round(coefficients[1],2)}')
print(r'R2 = %.2f' % (r2, ))

# Save figure
plt.savefig('/Users/tnye/tsuquakes/plots/parameter_correlations/vrupt_risetime.png', dpi=300)


