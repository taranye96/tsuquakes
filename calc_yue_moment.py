#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:37:43 2021

@author: tnye
"""

###############################################################################
# Script used to calculate the seismic moment from the Han Yue fault model.
###############################################################################

# Imports
import numpy as np

# Calculate seismic moment for the Han Yue fault model
fault = np.genfromtxt('/Users/tnye/tsuquakes/files/han_yue_fault.txt',delimiter=',')
velmod = np.genfromtxt('/Users/tnye/tsuquakes/files/mentawai.mod')
# fault = np.genfromtxt('/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/han_yue_depth_adjusted.txt',delimiter=',')
# velmod = np.genfromtxt('/Users/tnye/FakeQuakes/files/velmods/mentawai_v1.mod')

depth2bottom = []
for i in range(len(velmod)):
    if i == 0:
        depth2bottom.append(velmod[i,0]+3)
    else:
        depth2bottom.append(velmod[i,0]+depth2bottom[-1])

M0 = 0

for i in range(len(fault)):
    A = 14.25*15*1000000 #Area in m^2
    slip = fault[i,9] #slip in m
    depth = fault[i,4] #depth in km
    
    # Get shear modulus using density and Vs from velocity model
    if depth < depth2bottom[0]:
        rho = velmod[0,3]*1000 #density in kg/m^3
        Vs = velmod[0,1]*1000 #Vs in m/s
    else:
        for j in range(len(depth2bottom)):
            if j > 0:
                if depth < depth2bottom[j] and depth > depth2bottom[j-1]:
                    rho = velmod[j,3]*1000 #density in kg/m^3
                    Vs = velmod[j,1]*1000 #Vs in m/s
    mu = rho*(Vs**2)
    
    subM0 = mu*A*slip
    # print(f'subfault M0 = {subM0}')
    
    M0 += subM0

Mw = (2/3) * (np.log10(M0) - 9.1)
print(f'(N*m) Mw = {Mw}')

M0 = M0*10**7
Mw = (2/3) * (np.log10(M0) - 16.1)
print(f'(dyn-cm) Mw = {Mw}')
