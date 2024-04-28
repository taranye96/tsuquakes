#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:10:53 2022

@author: dmelgarm
"""

###############################################################################
# Script that compares ground motion from simulations using various Q
# parameters with GMM predictions. 
###############################################################################

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from glob import glob

orig_wfs = np.sort(glob('/Users/tnye/FakeQuakes/simulations/FQ_status/new_Q_model/new_fault_model/m7.85/old_q_params/standard_parameters/output/waveforms/*'))
shallowest_150_wfs = np.sort(glob('/Users/tnye/FakeQuakes/simulations/FQ_status/new_Q_model/new_fault_model/m7.85/old_q_params/q_param_test/all_stns/shallowest_150/output/waveforms/*'))
no_moho_150_wfs = np.sort(glob('/Users/tnye/FakeQuakes/simulations/FQ_status/new_Q_model/new_fault_model/m7.85/old_q_params/q_param_test/all_stns/no_moho_150/output/waveforms/*'))  
fastest_150_wfs = np.sort(glob('/Users/tnye/FakeQuakes/simulations/FQ_status/new_Q_model/new_fault_model/m7.85/old_q_params/q_param_test/all_stns/fastest_150/output/waveforms/*'))  
direct_150_wfs = np.sort(glob('/Users/tnye/FakeQuakes/simulations/FQ_status/new_Q_model/new_fault_model/m7.85/old_q_params/q_param_test/all_stns/direct_150/output/waveforms/*'))  
deeper_fault_wfs = np.sort(glob('/Users/tnye/FakeQuakes/simulations/FQ_status/new_Q_model/new_fault_model/m7.85/old_q_params/q_param_test/all_stns/no_moho_150_deeper_fault/output/waveforms/*'))  

gmm = np.genfromtxt('/Users/tnye/tsuquakes/files/GMM_pga/NGA-sub_PGA_m7.85.txt')
gmm2= np.genfromtxt('/Users/tnye/tsuquakes/files/GMM_pga/BCHydro_PGA_m7.85.txt')
# gmm3= np.genfromtxt('/Users/dmelgarm/ONC/GMM/Zhao2016_PGA.txt')
# gmm4= np.genfromtxt('/Users/dmelgarm/ONC/GMM/Montalva2017_PGA.txt')

sns.set_style('darkgrid')
plt.figure()
plt.loglog(gmm[:,0],np.exp(gmm[:,1])*9.81,color='blue',ls='-',label='NGA-Sub')
plt.loglog(gmm[:,0],np.exp(gmm[:,1]+gmm[:,2])*9.81,color='blue',ls='--')
plt.loglog(gmm[:,0],np.exp(gmm[:,1]-gmm[:,2])*9.81,color='blue',ls='--')
plt.loglog(gmm2[:,0],np.exp(gmm2[:,1])*9.81,label='BC Hydro',color='orange',ls='-')
plt.loglog(gmm2[:,0],np.exp(gmm2[:,1]+gmm2[:,2])*9.81,color='orange',ls='--')
plt.loglog(gmm2[:,0],np.exp(gmm2[:,1]-gmm2[:,2])*9.81,color='orange',ls='--')


N=2
for k in range(N):
    
    pga0 = np.genfromtxt(orig_wfs[k]+'/_ground_motions.txt',delimiter=',')
    pga1 = np.genfromtxt(shallowest_150_wfs[k]+'/_ground_motions.txt',delimiter=',')
    pga2 = np.genfromtxt(no_moho_150_wfs[k]+'/_ground_motions.txt',delimiter=',')
    pga3 = np.genfromtxt(fastest_150_wfs[k]+'/_ground_motions.txt',delimiter=',')
    pga4 = np.genfromtxt(direct_150_wfs[k]+'/_ground_motions.txt',delimiter=',')
    pga5 = np.genfromtxt(deeper_fault_wfs[k]+'/_ground_motions.txt',delimiter=',')
    
    plt.scatter(pga0[:,2],pga0[:,3],facecolor='k',label='Original FQ')
    plt.scatter(pga1[:,2],pga1[:,3],facecolor='r',label='shallowest, baselineQ=150')
    plt.scatter(pga2[:,2],pga2[:,3],facecolor='green',label='no moho, baselineQ=150')
    plt.scatter(pga3[:,2],pga3[:,3],facecolor='b',label='fastest, baselineQ=150')
    plt.scatter(pga4[:,2],pga4[:,3],facecolor='purple',label='direct, baselineQ=150')
    plt.scatter(pga5[:,2],pga4[:,3],facecolor='orange',label='deeper fault, no_moho, baselineQ=150')
    
    if k==0:
        plt.legend(loc='lower left')
    
plt.xlabel('Rrup (km)')
plt.ylabel('PGA HF (m/s/s/)')
plt.title('M7.85 (N=%d)' % N)

# plt.figure()
# plt.loglog(gmm[:,0],gmm[:,1])

# for k in range(N):
    
#     pga = np.genfromtxt(waveforms_folders[k]+'/_ground_motions.txt',delimiter=',')
#     plt.scatter(pga[:,2],pga[:,4])
    
# plt.xlabel('Rrup (km)')
# plt.ylabel('PGA BB (m/s/s/)')
# plt.title('M8.3 (N=16)')