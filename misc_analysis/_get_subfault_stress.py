#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:56:46 2023

@author: tnye
"""

# Imports
import numpy as np
from numpy import pi, mean
from mudpy.forward import get_mu
from mudpy import hfsims

stress = 50 #bars
fault = np.genfromtxt('/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/mentawai_fine2.rupt')
structure = np.genfromtxt('/Users/tnye/FakeQuakes/files/velmods/mentawai_v1.mod')

# Get subfault M0
slip=(fault[:,8]**2+fault[:,9]**2)**0.5
subfault_M0=slip*fault[:,10]*fault[:,11]*fault[:,13]
M0=subfault_M0.sum()
subfault_Mw = np.array([])
for i in range(len(subfault_M0)):
    if i != 2803:
        subfault_Mw = np.append(subfault_Mw, (2/3) * (np.log10(subfault_M0[i]) - 9.1))
    else:
        subfault_Mw = np.append(subfault_Mw, 0)

ind = np.where(subfault_M0 > 0)[0]
N=len(ind)

dl=mean((fault[:,10]+fault[:,11])/2) #predominant length scale
dl=dl/1000 # to km
    
# Frankel 95 scaling of corner frequency #verified this looks the same in GP
fc_scale=(M0)/(N*stress*dl**3*1e21) #Frankel scaling
small_event_M0 = stress*dl**3*1e21

subfault_fc = []
subfault_beta = []

#Get rho, alpha, beta at subfault depth
for kfault in ind:
    zs=fault[kfault,3]
    mu,alpha,beta=get_mu(structure,zs,return_speeds=True)
    rho=mu/beta**2
    
    #Get radiation scale factor
    Spartition=1/2**0.5
    
    rho=rho/1000 #to g/cm**3
    beta=(beta/1000)*1e5 #to cm/s
    alpha=(alpha/1000)*1e5
    
    #Verified this produces same value as in GP
    CS=(2*Spartition)/(4*pi*(rho)*(beta**3))
    CP=2/(4*pi*(rho)*(alpha**3))
    
    #Get local subfault rupture speed
    beta=beta/100 #to m/s
    vr=hfsims.get_local_rupture_speed(zs,beta,[10,15])
    vr=vr/1000 #to km/s
    dip_factor=hfsims.get_dip_factor(fault[kfault,5],fault[kfault,8],fault[kfault,9])
    
    #Subfault corner frequency
    c0=2.0 #GP2015 value
    subfault_fc.append((c0*vr)/(dip_factor))
    subfault_beta.append(beta)

subfault_stress = []
for i in range(len(ind)):
    
    # Rearranged eq. for theoreticla fc from kappa paper
    # subfault_stress.append((subfault_fc[i]/subfault_beta[i])**3 * 8.47 * subfault_M0[i])

    # Brune 1970 stress
    subfault_stress.append(8.5 * subfault_M0[i] * (subfault_fc[i]/subfault_beta[i])**3)

stress_total = np.array(subfault_stress).sum()/1e6
