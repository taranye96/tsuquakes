#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 08:43:01 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from glob import glob
from obspy import read
from mtspec import mtspec
import tsueqs_main_fns as tmf
import matplotlib.pyplot as plt

#%%

################### Observed GNSS disp -- Synthetic SM disp ###################

syn_dir = '/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/qexp_test/sd1.0_qexp0.8'
syn_acc_files = [f'{syn_dir}/output/waveforms/mentawai.000000/BSAT.bb.HNE.sac',
            f'{syn_dir}/output/waveforms/mentawai.000000/SLBU.bb.HNE.sac',
            f'{syn_dir}/output/waveforms/mentawai.000000/SMGY.bb.HNE.sac']

obs_files = ['/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/BSAT.LXE.mseed',
             '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/SLBU.LXE.mseed',
             '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/SMGY.LXE.mseed']

fcorner_high = 1/15.
order=2

syn_freqs = []
syn_spec = []
obs_freqs = []
obs_spec = []

for i, file in enumerate(syn_acc_files):
    
    stn = file.split('/')[-1].split('.')[0]
    
    # synthetic
    st_acc = read(file)
    samprate = st_acc[0].stats.sampling_rate
    
    st_disp = tmf.accel_to_veloc(tmf.highpass(tmf.accel_to_veloc(st_acc),fcorner_high,samprate,order,zerophase=True))
    
    # if stn == 'BSAT':
    #     data = st_disp[0].data[:5701+12000]
    # elif stn == 'SLBU':
    #     data = st_disp[0].data[:6101+12000]
    # elif stn == 'SMGY':
    #     data = st_disp[0].data[:6701+12000]
        
    data = st_disp[0].data
    
    syn_amp_squared, syn_freq =  mtspec(data, delta=st_disp[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=len(data), quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    syn_amp = np.sqrt(syn_amp_squared)
    
    syn_freqs.append(syn_freq)
    syn_spec.append(syn_amp)
    
    
    # observed
    st_obs = read(obs_files[i])
    samprate = st_obs[0].stats.sampling_rate
    
    # if stn == 'BSAT':
    #     data = st_obs[0].data[:58+120]
    # elif stn == 'SLBU':
    #     data = st_obs[0].data[:64+120]
    # elif stn == 'SMGY':
    #     data = st_obs[0].data[:68+120]
        
    data = st_obs[0].data
    
    obs_amp_squared, obs_freq =  mtspec(data, delta=st_obs[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=len(data), quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    obs_amp = np.sqrt(obs_amp_squared)
    
    obs_freqs.append(obs_freq)
    obs_spec.append(obs_amp)
    
    
    
    
fig, axs = plt.subplots(1,3,figsize=(12,4))
axs[0].loglog(obs_freqs[0],obs_spec[0],label='Observed')
axs[0].loglog(syn_freqs[0],syn_spec[0],label='Synthetic')
axs[1].loglog(obs_freqs[1],obs_spec[1],label='Observed')
axs[1].loglog(syn_freqs[1],syn_spec[1],label='Synthetic')
axs[2].loglog(obs_freqs[2],obs_spec[2],label='Observed')
axs[2].loglog(syn_freqs[2],syn_spec[2],label='Synthetic')
axs[0].set_title('BSAT')
axs[1].set_title('SLBU')
axs[2].set_title('SMGY')
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[0].set_xlim(xmax=1)
axs[1].set_xlim(xmax=1)
axs[2].set_xlim(xmax=1)
axs[0].set_ylim(ymin=10**-5)
axs[1].set_ylim(ymin=10**-5)
axs[2].set_ylim(ymin=10**-5)
fig.supxlabel('Frequency (Hz)')
fig.supylabel('Displacement Amplitude')
plt.subplots_adjust(left=0.08,bottom=0.125,right=0.98)
plt.savefig('/Users/tnye/tsuquakes/plots/misc/low-freq_acc_analysis/GNSSdisp_SMdisp.png',dpi=300)


#%%

#################### Observed GNSS acc -- Synthetic SM acc ####################

syn_dir = '/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/qexp_test/sd1.0_qexp0.8'
syn_acc_files = [f'{syn_dir}/output/waveforms/mentawai.000000/BSAT.bb.HNE.sac',
            f'{syn_dir}/output/waveforms/mentawai.000000/SLBU.bb.HNE.sac',
            f'{syn_dir}/output/waveforms/mentawai.000000/SMGY.bb.HNE.sac']

obs_files = ['/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/BSAT.LXE.mseed',
             '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/SLBU.LXE.mseed',
             '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/SMGY.LXE.mseed']

fcorner_high = 1/15.
order=2

syn_freqs = []
syn_spec = []
obs_freqs = []
obs_spec = []

for i, file in enumerate(obs_files):
    
    stn = file.split('/')[-1].split('.')[0]
    
    # observed
    st_disp = read(file)
    samprate = st_disp[0].stats.sampling_rate
    # st_disp = tmf.lowpass(st_disp, 0.1, samprate, order)
    acc_data = np.diff(np.diff(st_disp[0].data))
    
    # if stn == 'BSAT':
    #     data = acc_data[:58+120]
    # elif stn == 'SLBU':
    #     data = acc_data[:64+120]
    # elif stn == 'SMGY':
    #     data = acc_data[:68+120]

    data = acc_data
    
    obs_amp_squared, obs_freq =  mtspec(data, delta=st_disp[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=len(data), quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    obs_amp = np.sqrt(obs_amp_squared)
    
    obs_freqs.append(obs_freq)
    obs_spec.append(obs_amp)
    
    
    # synthetic
    st_syn= read(syn_acc_files[i])
    samprate = st_syn[0].stats.sampling_rate
    # st_syn = tmf.highpass(st_syn,fcorner_high,samprate,order,zerophase=True)
    
    # if stn == 'BSAT':
    #     data = st_syn[0].data[:5701+12000]
    # elif stn == 'SLBU':
    #     data = st_syn[0].data[:6101+12000]
    # elif stn == 'SMGY':
    #     data = st_syn[0].data[:6701+12000]
        
    data = st_syn[0].data
    
    syn_amp_squared, syn_freq =  mtspec(data, delta=st_syn[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=len(data), quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    syn_amp = np.sqrt(syn_amp_squared)
    
    syn_freqs.append(syn_freq)
    syn_spec.append(syn_amp)
    
    
    
    
fig, axs = plt.subplots(1,3,figsize=(14,4))
axs[0].loglog(obs_freqs[0],obs_spec[0],label='Observed')
axs[0].loglog(syn_freqs[0],syn_spec[0],label='Synthetic')
axs[1].loglog(obs_freqs[1],obs_spec[1],label='Observed')
axs[1].loglog(syn_freqs[1],syn_spec[1],label='Synthetic')
axs[2].loglog(obs_freqs[2],obs_spec[2],label='Observed')
axs[2].loglog(syn_freqs[2],syn_spec[2],label='Synthetic')
axs[0].set_title('BSAT')
axs[1].set_title('SLBU')
axs[2].set_title('SMGY')
# axs[0].set_xlim(xmax=1)
# axs[1].set_xlim(xmax=1)
# axs[2].set_xlim(xmax=1)
axs[0].set_ylim(ymin=10**-7)
axs[1].set_ylim(ymin=10**-7)
axs[2].set_ylim(ymin=10**-7)
fig.supxlabel('Frequency (Hz)')
fig.supylabel('Displacement Amplitude')
plt.subplots_adjust(left=0.08,bottom=0.125,right=0.98)
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.savefig('/Users/tnye/tsuquakes/plots/misc/low-freq_acc_analysis/GNSSacc_SMacc.png',dpi=300)


#%%

#################### Observed GNSS vel -- Synthetic SM vel ####################

syn_dir = '/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/qexp_test/sd1.0_qexp0.8'
syn_acc_files = [f'{syn_dir}/output/waveforms/mentawai.000000/BSAT.bb.HNE.sac',
            f'{syn_dir}/output/waveforms/mentawai.000000/SLBU.bb.HNE.sac',
            f'{syn_dir}/output/waveforms/mentawai.000000/SMGY.bb.HNE.sac']

obs_files = ['/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/BSAT.LXE.mseed',
             '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/SLBU.LXE.mseed',
             '/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/SMGY.LXE.mseed']

fcorner_high = 1/15.
order=2

syn_freqs = []
syn_spec = []
obs_freqs = []
obs_spec = []

for i, file in enumerate(obs_files):
    
    stn = file.split('/')[-1].split('.')[0]
    
    # observed
    st_disp = read(file)
    samprate = st_disp[0].stats.sampling_rate
    gnss_vel = np.diff(st_disp[0].data)
    
    if stn == 'BSAT':
        data = gnss_vel[:58+120]
    elif stn == 'SLBU':
        data = gnss_vel[:64+120]
    elif stn == 'SMGY':
        data = gnss_vel[:68+120]
    
    obs_amp_squared, obs_freq =  mtspec(gnss_vel, delta=st_disp[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=len(gnss_vel), quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    obs_amp = np.sqrt(obs_amp_squared)
    
    obs_freqs.append(obs_freq)
    obs_spec.append(obs_amp)
    
    
    # synthetic
    st_syn = read(syn_acc_files[i])
    samprate = st_syn[0].stats.sampling_rate
    # syn_vel = tmf.highpass(tmf.accel_to_veloc(st_syn),fcorner_high,samprate,order,zerophase=True)
    syn_vel = tmf.accel_to_veloc(st_syn)
    
    if stn == 'BSAT':
        data = st_syn[0].data[:5701+12000]
    elif stn == 'SLBU':
        data = st_syn[0].data[:6101+12000]
    elif stn == 'SMGY':
        data = st_syn[0].data[:6701+12000]
        
    data = syn_vel[0].data
    
    syn_amp_squared, syn_freq =  mtspec(data, delta=st_syn[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=len(data), quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    syn_amp = np.sqrt(syn_amp_squared)
    
    syn_freqs.append(syn_freq)
    syn_spec.append(syn_amp)
    
    
    
    
fig, axs = plt.subplots(1,3,figsize=(14,4))
axs[0].loglog(obs_freqs[0],obs_spec[0],label='Observed')
axs[0].loglog(syn_freqs[0],syn_spec[0],label='Synthetic')
axs[1].loglog(obs_freqs[1],obs_spec[1],label='Observed')
axs[1].loglog(syn_freqs[1],syn_spec[1],label='Synthetic')
axs[2].loglog(obs_freqs[2],obs_spec[2],label='Observed')
axs[2].loglog(syn_freqs[2],syn_spec[2],label='Synthetic')
axs[0].set_title('BSAT')
axs[1].set_title('SLBU')
axs[2].set_title('SMGY')
# axs[0].set_xlim(xmax=1)
# axs[1].set_xlim(xmax=1)
# axs[2].set_xlim(xmax=1)
axs[0].set_ylim(ymin=10**-7)
axs[1].set_ylim(ymin=10**-7)
axs[2].set_ylim(ymin=10**-7)
fig.supxlabel('Frequency (Hz)')
fig.supylabel('Velocity Amplitude')
plt.subplots_adjust(left=0.08,bottom=0.125,right=0.98)
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.savefig('/Users/tnye/tsuquakes/plots/misc/low-freq_acc_analysis/GNSSvel_SMvel_120.png',dpi=300)


