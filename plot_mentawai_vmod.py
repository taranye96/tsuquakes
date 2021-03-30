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
# vmod = pd.read_csv('/Users/tnye/tsuquakes/data/misc/vel_mod.csv')
vmod = np.genfromtxt('/Users/tnye/FakeQuakes/files/mentawai.mod')

# Get arrays for model thicknesses and Vs values
# mod_thick = np.array(vmod['Thickness(km)'])
# mod_Vs = np.array(vmod['Vs(km/s)'])
# mod_Vp = np.array(vmod['Vp(km/s)'])
mod_thick = vmod[:,0]
mod_Vs = vmod[:,1]
mod_Vp = vmod[:,2]

# Initialize lists for depths and Vs values to plot
depths = []
Vp = []
Vs = []

# Start initial depth at 0 km
depth = 0

# Loop through layers
for i, lthick in enumerate(mod_thick):
    
    # Set up array of depths in the thickness range
    ldepths = np.linspace(depth, int(depth+lthick), int(lthick)*10)
    # Append to full depths list
    depths.extend(ldepths)
    
    # Set up array of V values in the thickness range
    lVs = [mod_Vs[i]]*len(ldepths)
    lVp = [mod_Vp[i]]*len(ldepths)
    
    # Append to full V list
    Vp.extend(lVp)
    Vs.extend(lVs)
    
    # Set new depth to be old depth plus current layer thickness
    depth = depth+lthick


# Plot Velocity Model
plt.figure(figsize=(5,10))
plt.plot(Vp, depths, label='Vp')
plt.plot(Vs, depths, label='Vs')
plt.gca().invert_yaxis()
plt.legend()
plt.xlabel('Velocity (km/s)')
plt.ylabel('Depth (km)')
plt.title('Mentawai Velocity Model (Yue et al., 2014)')
plt.savefig('/Users/tnye/tsuquakes/plots/misc/Yue2014_vmod.png', dpi=300)