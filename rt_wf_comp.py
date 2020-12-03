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

from obspy import read
import matplotlib.pyplot as plt

runs = ['000000', '000001', '000002', '000003', '000004']

for run in runs:
    # Read in waveforms 
    disp_1x_st = read(f'/Users/tnye/FakeQuakes/parameters/standard/std/disp/output/waveforms/mentawai.{run}/BSAT.LYE.sac')
    disp_2x_st = read(f'/Users/tnye/FakeQuakes/parameters/rise_time/rt2x/disp/output/waveforms/mentawai.{run}/BSAT.LYE.sac')
    disp_3x_st = read(f'/Users/tnye/FakeQuakes/parameters/rise_time/rt3x/disp/output/waveforms/mentawai.{run}/BSAT.LYE.sac')
    disp_4x_st = read(f'/Users/tnye/FakeQuakes/parameters/rise_time/rt4x/disp/output/waveforms/mentawai.{run}/BSAT.LYE.sac')
    disp_obs_st = read('/Users/tnye/tsuquakes/data/Mentawai2010/disp_corr/BSAT.LXE.corr.mseed')
    
    acc_1x_st = read(f'/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/mentawai.{run}/CGJI.bb.HNE.sac')
    acc_2x_st = read(f'/Users/tnye/FakeQuakes/parameters/rise_time/rt2x/sm/output/waveforms/mentawai.{run}/CGJI.bb.HNE.sac')
    acc_3x_st = read(f'/Users/tnye/FakeQuakes/parameters/rise_time/rt3x/sm/output/waveforms/mentawai.{run}/CGJI.bb.HNE.sac')
    acc_4x_st = read(f'/Users/tnye/FakeQuakes/parameters/rise_time/rt4x/sm/output/waveforms/mentawai.{run}/CGJI.bb.HNE.sac')
    acc_obs_st = read('/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr/CGJI.HNE.corr.mseed')
    
    # Make displacement figure
    plt.plot(disp_obs_st[0].times(), disp_obs_st[0].data, linewidth=.8, color='black', label='obs')
    plt.plot(disp_1x_st[0].times(), disp_1x_st[0].data, linewidth=.8, color='lightcoral', label='1x')
    plt.plot(disp_2x_st[0].times(), disp_2x_st[0].data, linewidth=.8, color='orange', label='2x')
    plt.plot(disp_3x_st[0].times(), disp_3x_st[0].data, linewidth=.8, color='green', label='3x')
    plt.plot(disp_4x_st[0].times(), disp_4x_st[0].data, linewidth=.8, color='blue', label='4x')
    plt.legend()
    plt.title('Rise Time Disp Waveforms: BSAT')
    plt.xlabel('time (s)')
    plt.ylabel('amplitude (m)')
    plt.savefig(f'/Users/tnye/tsuquakes/plots/rt_comp/rt_disp_{run}.png', dpi=300)
    plt.close()
    
    # Make acceleration figure
    plt.plot(acc_1x_st[0].times(), acc_1x_st[0].data, linewidth=.8, color='lightcoral', label='1x')
    plt.plot(acc_2x_st[0].times(), acc_2x_st[0].data, linewidth=.8, color='orange', label='2x')
    plt.plot(acc_3x_st[0].times(), acc_3x_st[0].data, linewidth=.8, color='green', label='3x')
    plt.plot(acc_4x_st[0].times(), acc_4x_st[0].data, linewidth=.8, color='blue', label='4x')
    plt.plot(acc_obs_st[0].times(), acc_obs_st[0].data, linewidth=.8, color='black', label='obs')
    plt.legend()
    plt.title('Rise Time Acc Waveforms: CGJI')
    plt.xlabel('time (s)')
    plt.ylabel('amplitude (m)')
    plt.savefig(f'/Users/tnye/tsuquakes/plots/rt_comp/rt_acc_{run}.png', dpi=300)
    plt.close()