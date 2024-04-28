#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:37:43 2021

@author: tnye
"""

###############################################################################
# Script used to calculate the seismic moment from stochastic rupture models. 
###############################################################################

# Imports
import numpy as np
import pandas as pd

# Calculate seismic moment for the Han Yue fault model
fault = np.genfromtxt('/Users/tnye/tsuquakes/files/mentawai_fine2.rupt')
# fault = np.genfromtxt('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.85/new_q_params/standard_parameters/output/ruptures/mentawai.000000.rupt')
velmod = np.genfromtxt('/Users/tnye/tsuquakes/files/mentawai_newQ.mod')

depth2bottom = []
for i in range(len(velmod)):
    if i == 0:
        depth2bottom.append(velmod[i,0])
    else:
        depth2bottom.append(velmod[i,0]+depth2bottom[-1])

M0 = 0

for i in range(len(fault)):
    A = 2000*2000 #Area in m^2
    ss_slip = fault[i,8]
    ds_slip = fault[i,9] 
    slip = np.sqrt(ds_slip**2 + ss_slip**2)
    depth = fault[i,3] #depth in km
    
    # Get shear modulus using density and Vs from velocity model
    if depth < depth2bottom[0]:
        rho = velmod[0,3]*1000 #density in kg/m^3
        Vs = velmod[0,1]*1000 #Vs in m/s
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
