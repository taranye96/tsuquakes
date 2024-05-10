#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:58:11 2019

@author: tnye
"""

###############################################################################
# This script creates a more finely discretized .fault file with dimensions
# based on the Han Yue et al. (2014) finite fault model for the 2010 Mentawai
# event. This script also calls resample_to_finer_rupt.sh to interpolate the
# slip pattern on the finer fault model.
###############################################################################

# Imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import genfromtxt
from mudpy import forward
import pandas as pd
import numpy as np
import geopy.distance
import subprocess

# File path to save new .fault file
fout = '/Users/tnye/FakeQuakes/files/fault_info/mentawai_fine2.fault'

# Read in Han Yue et al. (2014) fault model
han_yue = pd.read_csv('/Users/tnye/FakeQuakes/files/model_info/han_yue.rupt', delimiter='\t')

# Get the coordinates of each subfault
han_lon = np.array(han_yue['lon'])
han_lat = np.array(han_yue['lat'])
han_depth = np.array(han_yue['z(km)']) - 3  # subtract 3 km because the Han Yue model assumes 3km of water at the top

# Get the center of the Han Yue fault model
center = sum(han_lon)/len(han_lon), sum(han_lat)/len(han_lat), sum(han_depth)/len(han_depth)

# Get min depth from the Han Yue fault model
l1_ind = np.where(han_depth==np.min(han_depth))[0]

# Get the upper corners of the fault model
c1 = han_lat[np.where(han_lon==np.min(han_lon[l1_ind]))][0], np.min(han_lon[l1_ind])
c2 = han_lat[np.where(han_lon==np.max(han_lon[l1_ind]))][0], np.max(han_lon[l1_ind])

# Get the fault strke length in km
strike_dist = geopy.distance.distance(c1, c2).km

# Set certain parameters for the fault model, including the strike, dip,
    # rise time (doesn't do anything), fault center, along dip and along strike
    # length (km) of the fault, number of subfaults along strike, and the number
    # of subfaults updip and downdip from the fault center
strike = 324
dip = 7.5
rise_time = 8
epicenter = center
dx_dip = 2.5
dx_strike = 2.5
nstrike = round(strike_dist/dx_strike)
num_updip = round(np.abs((center[2]-np.min(han_depth))/np.cos(np.deg2rad(90-dip)))/dx_dip)
num_downdip = round(np.abs((np.max(han_depth)-center[2])/np.cos(np.deg2rad(90-dip)))/dx_dip)

# Make new fault model
forward.makefault(fout,strike,dip,nstrike,dx_dip,dx_strike,epicenter,num_updip,num_downdip,rise_time)


#%%
# Plot to coarsely view fault models
o = genfromtxt('/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/han_yue_depth_adjusted.rupt')
f = genfromtxt(fout)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(f[:,1], f[:,2], -f[:,3], c='blue')
ax.scatter(o[:,1], o[:,2], -o[:,3], c='red')


#%%

frupt = fout.replace('fault_info','model_info').replace('.fault','.rupt')
subprocess.run(['./resample_to_finer_rupt.sh',fout,frupt])
