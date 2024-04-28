#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 22:35:52 2022

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from obspy import read
from mtspec import mtspec
import matplotlib.pyplot as plt

# parameters
stress = 5e6
Vs = 3500 

stn = 'UCLU'
ruptures = ['eews.000000', 'eews.000034', 'eews.000078', 'eews.000109']
colors = ['orange', 'blue', 'green', 'purple']
home_dir = '/Users/tnye/ONC/simulations/cascadia'
obs_df = pd.read_csv('/Users/tnye/tsuquakes/data/obs_spectra/acc_binned_spec.csv')

fig, ax = plt.subplots(1,1,figsize=(8,8))
for i, rupture in enumerate(ruptures):
    
    # Get magnitude
    log = open(f'{home_dir}/ruptures/{rupture}.log')
    lines = log.readlines()
    M = float(lines[-7].split(' ')[3].split('\n')[0])

    # Estimate fc
    M0 = 10.**((3./2.)*M + 9.1)
    # fc = Vs*(stress/(8.47*M0))**(1./3.)
    fc = ((stress*(Vs**3))/(47.24*M0))**(1./3.)
    
    # Read in waveforms
    lf_stream = read(f'{home_dir}/raw_waveforms/ONC-GNSS/{rupture}/{stn}.LYE.sac')
    bb_stream = read(f'{home_dir}/raw_waveforms/ONC-strong-motion/{rupture}/{stn}.bb.HNE.sac')
    
    # Get stats
    lf_data = lf_stream[0].data
    lf_dt = lf_stream[0].stats.delta
    lf_nyquist = 0.5 * lf_stream[0].stats.sampling_rate
    lf_npts = lf_stream[0].stats.npts
    bb_data = bb_stream[0].data
    bb_dt = bb_stream[0].stats.delta
    bb_nyquist = 0.5 * bb_stream[0].stats.sampling_rate
    bb_npts = bb_stream[0].stats.npts   

    # Calc spectra amplitudes and frequencies 
    lf_amp_squared, lf_freq =  mtspec(lf_data, delta=lf_dt, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_npts, quadratic=True)
    bb_amp_squared, bb_freq =  mtspec(bb_data, delta=bb_dt, time_bandwidth=4, 
                              number_of_tapers=5, nfft=bb_npts, quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    lf_amp = np.sqrt(lf_amp_squared)
    bb_amp = np.sqrt(bb_amp_squared)
    
    # Plot figure
    ax.loglog(lf_freq,lf_amp,lw=0.8,color=colors[i],alpha=0.4,label=f'M{M}')
    ax.loglog(bb_freq,bb_amp,lw=0.8,color=colors[i],alpha=0.4,ls='--')
    # plt.vlines(fc,np.min(np.concatenate((lf_amp,bb_amp))),np.max(np.concatenate((lf_amp,bb_amp))),ls='--',lw=0.8,color='k')
    plt.vlines(fc,10**-8,2*10**2,ls='-',lw=0.8,color=colors[i])
    plt.text(fc,.0001,f'M{M}',rotation=90)
    plt.legend()
    ax.set_xlim(xmax=10)
    ax.set_ylim(10**-8,2*10**2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(stn)
    plt.subplots_adjust(top=0.925,bottom=0.095,right=0.975,left=0.12)
    # plt.savefig(f'/Users/tnye/tsuquakes/plots/q_param_test/spectra_fc/cascadia_{stn}.png')
    # plt.close()