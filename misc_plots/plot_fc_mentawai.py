#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 00:48:51 2022

@author: tnye
"""

###############################################################################
# Script used to make plots of matched filter components with a theoretical fc. 
###############################################################################

# Imports
import numpy as np
import pandas as pd
from obspy import read
from mtspec import mtspec
import matplotlib.pyplot as plt
import IM_fns

# parameters
stress = 1e6
Vs = 2500 

qexp=0.8

stns = pd.read_csv('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/baseline_test/sd1.0_base100/data/station_info/sm_close_stns.txt').Station.values
# stn = 'MNSI'
# stn_ind = 6

home_dir = f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.84/q_param_test/qexp_test/sd1.0_qexp{qexp}'

obs_df = pd.read_csv('/Users/tnye/tsuquakes/data/obs_spectra/acc_binned_spec.csv')

for stn_ind, stn in enumerate(stns):
    
    obs_amps = np.array(obs_df.iloc[:,int(obs_df.shape[1]/2):])[stn_ind]

    # Get magnitude
    log = open(f'{home_dir}/output/ruptures/mentawai.000000.log')
    lines = log.readlines()
    M = float(lines[-10].split(' ')[3].split('\n')[0])
    
    # Estimate fc
    M0 = 10.**((3./2.)*M + 9.1)
    fc = Vs*(stress/(8.47*M0))**(1./3.)
    
    # Read in waveforms
    lf_stream = read(f'{home_dir}/output/waveforms/mentawai.000000/{stn}.LYE.sac')
    hf_stream = read(f'{home_dir}/output/waveforms/mentawai.000000/{stn}.HNE.mpi.sac')
    bb_stream = read(f'{home_dir}/output/waveforms/mentawai.000000/{stn}.bb.HNE.sac')
    
    lf_freq, lf_amp = IM_fns.calc_spectra(lf_stream, 'sm')
    hf_freq, hf_amp = IM_fns.calc_spectra(hf_stream, 'sm')
    bb_freq, bb_amp = IM_fns.calc_spectra(bb_stream, 'sm')
    
    # Plot figure
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.loglog(hf_freq,obs_amps,c='k',lw=2,alpha=0.8,label='observed')
    ax.loglog(lf_freq,lf_amp,lw=0.8,alpha=0.4,label='LF')
    ax.loglog(hf_freq,hf_amp,lw=0.8,alpha=0.4,label='HF')
    ax.loglog(bb_freq,bb_amp,lw=0.8,alpha=0.4,label='BB')
    plt.vlines(fc,10**-8,2*10**-2,ls='--',lw=0.8,color='k')
    plt.text(fc,.00001,' fc',fontsize=11)
    plt.legend()
    # ax.set_xlim(10**-2,10)
    # ax.set_ylim(10**-8,2*10**-2)
    ax.set_xlim(10**-2,9)
    ax.set_ylim(ymin=10**-7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{stn}, Qexp={qexp}')
    plt.subplots_adjust(top=0.96,bottom=0.075,right=0.975,left=0.095)
    plt.savefig(f'/Users/tnye/tsuquakes/plots/spectra_fc/{stn}_qexp{qexp}.png')
    plt.close()
    
    