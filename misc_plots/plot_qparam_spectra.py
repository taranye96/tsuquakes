#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:30:31 2022

@author: tnye
"""

###############################################################################
# Script used to make plots of spectra using different values for Q-attenuation
# parameters.
###############################################################################

# Imports
import numpy as np
import pandas as pd
from glob import glob
from mtspec import mtspec
from obspy import read
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import IM_fns

stns = pd.read_csv('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/baseline_test/sd1.0_base100/data/station_info/sm_close_stns.txt').Station.values
obs_df = pd.read_csv('/Users/tnye/tsuquakes/data/obs_spectra/acc_binned_spec.csv')

for stn_ind, stn in enumerate(stns):
    
    obs_amps = np.array(obs_df.iloc[:,int(obs_df.shape[1]/2):])[stn_ind]
    
    base_wfs_E = sorted(glob(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/baseline_test/*/processed_wfs/acc/mentawai.000000/{stn}.HNE*'))
    qexp_wfs_E = sorted(glob(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/qexp_test/*/processed_wfs/acc/mentawai.000000/{stn}.HNE*'))
    qcexp_wfs_E = sorted(glob(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/qcexp_test/*/processed_wfs/acc/mentawai.000000/{stn}.HNE*'))
    scat_wfs_E = sorted(glob(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/scattering_test/*/processed_wfs/acc/mentawai.000000/{stn}.HNE*'))
    base_wfs_N = sorted(glob(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/baseline_test/*/processed_wfs/acc/mentawai.000000/{stn}.HNN*'))
    qexp_wfs_N = sorted(glob(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/qexp_test/*/processed_wfs/acc/mentawai.000000/{stn}.HNN*'))
    qcexp_wfs_N = sorted(glob(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/qcexp_test/*/processed_wfs/acc/mentawai.000000/{stn}.HNN*'))
    scat_wfs_N = sorted(glob(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/scattering_test/*/processed_wfs/acc/mentawai.000000/{stn}.HNN*'))
    
    base_ampsE = []
    base_ampsN = []
    base_ampsNE = []
    base_freqs = []
    base_runs = []
    qexp_ampsE = []
    qexp_ampsN = []
    qexp_ampsNE = []
    qexp_freqs = []
    qexp_runs = []
    qcexp_ampsE = []
    qcexp_ampsN = []
    qcexp_ampsNE = []
    qcexp_freqs = []
    qcexp_runs = []
    scat_ampsE = []
    scat_ampsN = []
    scat_ampsNE = []
    scat_freqs = []
    scat_runs = []
    
    for i in range(len(base_wfs_E)):

        # Read in file 
        streamE = read(base_wfs_E[i])
        streamN = read(base_wfs_N[i])
        freq, ampE = IM_fns.calc_spectra(streamE, 'sm')
        freq, ampN = IM_fns.calc_spectra(streamN, 'sm')
        
        base_ampsE.append(ampE)
        base_ampsN.append(ampN)
        base_ampsNE.append(np.sqrt(ampE**2 + ampN**2))
        base_freqs.append(freq)
        
        base_runs.append(base_wfs_E[i].split('/')[-5])
    
    for i in range(len(qexp_wfs_E)):
        
        # Read in file 
        streamE = read(qexp_wfs_E[i])
        streamN = read(qexp_wfs_N[i])
        freq, ampE = IM_fns.calc_spectra(streamE, 'sm')
        freq, ampN = IM_fns.calc_spectra(streamN, 'sm')
        
        qexp_ampsE.append(ampE)
        qexp_ampsN.append(ampN)
        qexp_ampsNE.append(np.sqrt(ampE**2 + ampN**2))
        qexp_freqs.append(freq)
        
        qexp_runs.append(qexp_wfs_E[i].split('/')[-5])
    
    for i in range(len(qcexp_wfs_E)):
        
        # Read in file 
        streamE = read(qcexp_wfs_E[i])
        streamN = read(qcexp_wfs_N[i])
        freq, ampE = IM_fns.calc_spectra(streamE, 'sm')
        freq, ampN = IM_fns.calc_spectra(streamN, 'sm')
        
        qcexp_ampsE.append(ampE)
        qcexp_ampsN.append(ampN)
        qcexp_ampsNE.append(np.sqrt(ampE**2 + ampN**2))
        qcexp_freqs.append(freq)
        
        qcexp_runs.append(qcexp_wfs_E[i].split('/')[-5])
    
    for i in range(len(scat_wfs_E)):
        
        # Read in file 
        streamE = read(scat_wfs_E[i])
        streamN = read(scat_wfs_N[i])
        freq, ampE = IM_fns.calc_spectra(streamE, 'sm')
        freq, ampN = IM_fns.calc_spectra(streamN, 'sm')
        
        scat_ampsE.append(ampE)
        scat_ampsN.append(ampN)
        scat_ampsNE.append(np.sqrt(ampE**2 + ampN**2))
        scat_freqs.append(freq)
        
        scat_runs.append(scat_wfs_E[i].split('/')[-5])
    
    # East component
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    for j in range(len(base_ampsNE)):
        color = cm.Reds_r(np.linspace(0, 1, 6))
        ax.loglog(base_freqs[j],base_ampsNE[j],c=color[j],lw=0.8,alpha=0.5,label=base_runs[j])
    for j in range(len(qexp_ampsNE)):
        color = cm.Blues_r(np.linspace(0, 1, 6))
        ax.loglog(qexp_freqs[j],qexp_ampsNE[j],c=color[j],lw=0.8,alpha=0.5,label=qexp_runs[j])
    for j in range(len(qcexp_ampsNE)):
        color = cm.Greens_r(np.linspace(0, 1, 6))
        ax.loglog(qcexp_freqs[j],qcexp_ampsNE[j],c=color[j],lw=0.8,alpha=0.5,label=qcexp_runs[j])
    for j in range(len(scat_ampsNE)): 
        color = cm.Purples_r(np.linspace(0, 1, 6))
        ax.loglog(scat_freqs[j],scat_ampsNE[j],c=color[j],lw=0.8,alpha=0.5,label=scat_runs[j])
    ax.loglog(scat_freqs[j],obs_amps,c='k',lw=2,alpha=0.8,label='observed')
    plt.legend()
    ax.set_xlim(0.05,9)
    ax.set_ylim(ymin=10**-7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(stn)
    plt.subplots_adjust(top=0.96,bottom=0.075,right=0.975,left=0.095)
    plt.savefig(f'/Users/tnye/tsuquakes/plots/q_param_test/{stn}.png')
    
    # East component
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    for j in range(len(base_ampsNE)):
        color = cm.Reds_r(np.linspace(0, 1, 6))
        ax.loglog(base_freqs[j],base_ampsNE[j],c=color[j],lw=0.8,alpha=0.5,label=base_runs[j])
    ax.loglog(base_freqs[j],obs_amps,c='k',lw=2,alpha=0.8,label='observed')
    plt.legend()
    ax.set_xlim(0.05,9)
    ax.set_ylim(ymin=10**-7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(stn)
    plt.subplots_adjust(top=0.96,bottom=0.075,right=0.975,left=0.095)
    plt.savefig(f'/Users/tnye/tsuquakes/plots/q_param_test/{stn}_baseline.png')
    plt.close()
    
    # East component
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    for j in range(len(qexp_ampsNE)):
        color = cm.Blues_r(np.linspace(0, 1, 6))
        ax.loglog(qexp_freqs[j],qexp_ampsNE[j],c=color[j],lw=0.8,alpha=0.5,label=qexp_runs[j])
    ax.loglog(qexp_freqs[j],obs_amps,c='k',lw=2,alpha=0.8,label='observed')
    plt.legend()
    ax.set_xlim(0.05,9)
    ax.set_ylim(ymin=10**-7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(stn)
    plt.subplots_adjust(top=0.96,bottom=0.075,right=0.975,left=0.095)
    plt.savefig(f'/Users/tnye/tsuquakes/plots/q_param_test/{stn}_qexp.png')
    plt.close()
    
    # East component
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    for j in range(len(qcexp_ampsNE)):
        color = cm.Greens_r(np.linspace(0, 1, 6))
        ax.loglog(qcexp_freqs[j],qcexp_ampsNE[j],c=color[j],lw=0.8,alpha=0.5,label=qcexp_runs[j])
    ax.loglog(qcexp_freqs[j],obs_amps,c='k',lw=2,alpha=0.8,label='observed')
    plt.legend()
    ax.set_xlim(0.05,9)
    ax.set_ylim(ymin=10**-7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(stn)
    plt.subplots_adjust(top=0.96,bottom=0.075,right=0.975,left=0.095)
    plt.savefig(f'/Users/tnye/tsuquakes/plots/q_param_test/{stn}_qcexp.png')
    plt.close()
    
    # East component
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    for j in range(len(scat_ampsNE)): 
        color = cm.Purples_r(np.linspace(0, 1, 6))
        ax.loglog(scat_freqs[j],scat_ampsNE[j],c=color[j],lw=0.8,alpha=0.5,label=scat_runs[j])
    ax.loglog(scat_freqs[j],obs_amps,c='k',lw=2,alpha=0.8,label='observed')
    plt.legend()
    ax.set_xlim(0.05,9)
    ax.set_ylim(ymin=10**-7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(stn)
    plt.subplots_adjust(top=0.96,bottom=0.075,right=0.975,left=0.095)
    plt.savefig(f'/Users/tnye/tsuquakes/plots/q_param_test/{stn}_scattering.png')
    plt.close()


    
    