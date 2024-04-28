#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 18:02:46 2020

@author: tnye
"""

###############################################################################
# Script used to make displacement and acceleration comparison waveform figures
# for a sample of stations using the varied rise times.  This was to help
# analyze why varying rise time seems to have little to no effect on the IMs.  
###############################################################################

# Imports
from obspy import read
from os import path, makedirs
import matplotlib.pyplot as plt

# Run names
runs = ['000000', '000001']

home_dir = '/Users/tnye/FakeQuakes/simulations/FQ_status/new_Q_model/new_fault_model/m7.84/new_runs'
out_folder_name = 'new_runs'

# Loop over runs
for run in runs:
    
    # Create folder to store plots
    if not path.exists(f'/Users/tnye/tsuquakes/plots/risetime_comparison/{out_folder_name}'):
        makedirs(f'/Users/tnye/tsuquakes/plots/risetime_comparison/{out_folder_name}')
    
    # Read in displacement waveforms 
    disp_1x_st = read(f'{home_dir}/standard/output/waveforms/mentawai.{run}/BSAT.LYE.sac')
    disp_2x_st = read(f'{home_dir}/risetime/rt2x/output/waveforms/mentawai.{run}/BSAT.LYE.sac')
    disp_obs_st = read('/Users/tnye/tsuquakes/data/waveforms/individual/disp/BSAT.LXE.mseed')
    
    # Read in acceleration waveforms
    acc_1x_st = read(f'{home_dir}/risetime/rt2x/output/waveforms/mentawai.{run}/KASI.bb.HNE.sac')
    acc_2x_st = read(f'{home_dir}/risetime/rt2x/output/waveforms/mentawai.{run}/KASI.bb.HNE.sac')
    acc_obs_st = read('/Users/tnye/tsuquakes/data/waveforms/individual/acc/KASI.HNE.mseed')
    
    # Make displacement figure
    plt.figure()
    plt.plot(disp_obs_st[0].times(), disp_obs_st[0].data, linewidth=.8, label='obs')
    plt.plot(disp_1x_st[0].times(), disp_1x_st[0].data, linewidth=.8, label='1x')
    plt.plot(disp_2x_st[0].times(), disp_2x_st[0].data, linewidth=.8, label='2x')
    plt.legend()
    plt.title('Rise Time Disp Waveforms: BSAT')
    plt.xlabel('time (s)')
    plt.ylabel('amplitude (m)')
    plt.savefig(f'/Users/tnye/tsuquakes/plots/risetime_comparison/{out_folder_name}/BSAT_{run}.png', dpi=300)
    plt.close()
    
    # Make acceleration figure
    plt.figure()
    plt.plot(acc_1x_st[0].times(), acc_1x_st[0].data, linewidth=.8, label='1x')
    plt.plot(acc_2x_st[0].times(), acc_2x_st[0].data, linewidth=.8, label='2x')
    plt.plot(acc_obs_st[0].times(), acc_obs_st[0].data, linewidth=.8, label='obs')
    plt.legend()
    plt.title('Rise Time Acc Waveforms: KASI')
    plt.xlabel('time (s)')
    plt.ylabel('amplitude (m)')
    plt.savefig(f'/Users/tnye/tsuquakes/plots/risetime_comparison/{out_folder_name}/KASI_{run}.png', dpi=300)
    plt.close()
    
    
    