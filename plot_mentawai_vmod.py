#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:51:11 2020

@author: tnye
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in velocity model
vmod = pd.read_csv('/Users/tnye/tsuquakes/data/misc/vel_mod.csv')

# Get arrays for model thicknesses and Vs values
mod_thick = np.array(vmod['Thickness(km)'])
mod_Vs = np.array(vmod['Vs(km/s)'])

# Initialize lists for depths and Vs values to plot
depths = []
Vs = []

# Start initial depth at 0 km
depth = 0

# Loop through layers
for i, lthick in enumerate(mod_thick):
    
    # Set up array of depths in the thickness range
    ldepths = np.linspace(depth, depth+lthick, lthick*10)
    # Append to full depths list
    depths.extend(ldepths)
    
    # Set up array of Vs values in the thickness range
    lVs = [mod_Vs[i]]*len(ldepths)
    # Append to full Vs list
    Vs.extend(lVs)
    
    # Set new depth to be old depth plus current layer thickness
    depth = depth+lthick


# Plot Velocity Model
plt.figure(figsize=(5,10))
plt.plot(Vs, depths)
plt.gca().invert_yaxis()
plt.xlabel('Vs (km/s)')
plt.ylabel('Depth (km)')
plt.title('Mentawai Velocity Model (Yue et al., 2014)')
plt.savefig('/Users/tnye/tsuquakes/plots/misc/Yue2014_vmod.png', dpi=300)