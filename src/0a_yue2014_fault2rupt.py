#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:33:55 2022

@author: tnye
"""

###############################################################################
# This script makes a .rupt file from the Yue et al. (2014) fault file for the
# 2010 Mentawai event.
###############################################################################

# Imports
import numpy as np
import pandas as pd

# Read in fault file
yue_fault = pd.read_csv('/Users/tnye/tsuquakes/files/han_yue_fault.txt', delimiter=',')

# Set up columns of .rupt file
nsub = np.arange(1,len(yue_fault)+1,1)
lon = np.array(yue_fault['lon'])
lat = np.array(yue_fault['lat'])
depth = np.array(yue_fault['depth'])
strike = np.array(yue_fault['strike'])
dip = np.array(yue_fault['dip'])
rise = np.full_like(dip, 0)                 # null column
dura = np.full_like(dip, 0)                 # null column (will be filled when making stochastic ruptures)
ss_slip = np.full_like(dip, 0)              # no strike-slip, assume pure thrust
ds_slip = np.array(yue_fault['slip(m)'])
width = np.full_like(dip, 14250)
length = np.full_like(dip, 15000)
rupt_time = np.full_like(dip, 0)            # null column (will be filled when making stochastic ruptures)
rigidity = np.full_like(dip, 0)             # null column (will be filled when making stochastic ruptures)

# Save to .rupt file
outfile = open('/Users/tnye/tsuquakes/data/fault_info/han_yue.rupt', 'w')
file_data = np.array([nsub,lon,lat,depth,strike,dip,rise,dura,ss_slip,ds_slip,width,length,rupt_time,rigidity],dtype=object)
file_data = file_data.T
np.savetxt(outfile, file_data, fmt='%E', delimiter='\t')
outfile.close()
