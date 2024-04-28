#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:26:23 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd 
from glob import glob
from obspy import read
from mtspec import mtspec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, ScalarFormatter

obs_acc = sorted(glob('/Users/tnye/tsuquakes/data/waveforms/average/rotd50/acc/*'))
obs_vel = sorted(glob('/Users/tnye/tsuquakes/data/waveforms/average/rotd50/vel/*'))
obs_disp = sorted(glob('/Users/tnye/tsuquakes/data/waveforms/average/eucnorm_3comp/disp/*'))

### Acceleration
acc_times = []
acc_amps = []
acc_freqs = []
acc_spec = []
sm_names = []

# Loop through files
for file in obs_acc:
    
    st = read(file)
    dt = st[0].stats.delta
    npts = st[0].stats.npts
    sm_names.append(file.split('/')[-1].split('.')[0])
    
    # Append time series data
    acc_times.append(st[0].times('matplotlib'))
    acc_amps.append(st[0].data)
    
    # Calculate spectra
    amp_squared, freq =  mtspec(st[0].data, delta=dt, time_bandwidth=4, 
                              number_of_tapers=5, nfft=npts, quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    amp = np.sqrt(amp_squared)
    
    acc_freqs.append(freq)
    acc_spec.append(amp)


### Velocity
vel_times = []
vel_amps = []
vel_freqs = []
vel_spec = []

# Loop through files
for file in obs_vel:
    
    st = read(file)
    dt = st[0].stats.delta
    npts = st[0].stats.npts
    
    # Append time series data
    vel_times.append(st[0].times('matplotlib'))
    vel_amps.append(st[0].data)
    
    # Calculate spectra
    amp_squared, freq =  mtspec(st[0].data, delta=dt, time_bandwidth=4, 
                              number_of_tapers=5, nfft=npts, quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    amp = np.sqrt(amp_squared)
    
    vel_freqs.append(freq)
    vel_spec.append(amp)


### Displacement
disp_times = []
disp_amps = []
disp_freqs = []
disp_spec = []
gnss_names = []

# Loop through files
for file in obs_disp:
    
    st = read(file)
    dt = st[0].stats.delta
    npts = st[0].stats.npts
    gnss_names.append(file.split('/')[-1].split('.')[0])
    
    # Append time series data
    disp_times.append(st[0].times('matplotlib'))
    disp_amps.append(st[0].data)
    
    # Calculate spectra
    amp_squared, freq =  mtspec(st[0].data, delta=dt, time_bandwidth=4, 
                              number_of_tapers=5, nfft=npts, quadratic=True)
     
    # Convert from power spectra to amplitude spectra
    amp = np.sqrt(amp_squared)
    
    disp_freqs.append(freq)
    disp_spec.append(amp)
    

#%%

# Acceleration spectra figure 
units = 'm/s'
ylim = 7*10**-15, 6*10**-1
xlim = .002, 10
fig, axs = plt.subplots(3,3,figsize=(10,8))
k = 0
for i in range(3):
    for j in range(3):
        if k+1 <= len(obs_acc):
            axs[i][j].loglog(acc_freqs[k],acc_spec[k],lw=1,ls='-')
            axs[i][j].grid(linestyle='--')
            axs[i][j].text(0.8,5E-2,'rotd50',transform=axs[i][j].transAxes,size=10)
            axs[i][j].tick_params(axis='both', which='major', labelsize=10)
            axs[i][j].set_xlim(xlim)
            axs[i][j].set_ylim(ylim)
            axs[i][j].set_title(sm_names[k],fontsize=10)
            if i < 3 -1:
                axs[i][j].set_xticklabels([])
            if i == 3 -2 and j == 0:
                axs[i][j].set_xticklabels([])
            if j > 0:
                axs[i][j].set_yticklabels([])
            k += 1
handles, labels = axs[0][0].get_legend_handles_labels()
fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
fig.supylabel(f'Amplitude ({units})',fontsize=12)
fig.supxlabel('Frequency (Hz)',fontsize=12)
fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
plt.subplots_adjust(left=0.11, bottom=0.075, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
plt.savefig('/Users/tnye/tsuquakes/plots/misc/observed/acc_FAS.png', dpi=300)
plt.close()


# Acceleration waveforms figure 
units = 'm/s/s'
fig, axs = plt.subplots(3,3,figsize=(10,8))
k = 0
for i in range(3):
    for j in range(3):
        if k+1 <= len(obs_acc):
            axs[i][j].plot(acc_times[k],acc_amps[k],lw=1,ls='-')
            axs[i][j].grid(linestyle='--')
            axs[i][j].text(0.8,5E-2,'rotd50',transform=axs[i][j].transAxes,size=10)
            axs[i][j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            axs[i][j].tick_params(axis='both', which='major', labelsize=10)
            axs[i][j].yaxis.offsetText.set_fontsize(10)
            axs[i][j].set_title(sm_names[k],fontsize=10)
            axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            if i < 3 -1:
                axs[i][j].set_xticklabels([])
            if i == 3 -2 and j == 0:
                axs[i][j].set_xticklabels([])
            k += 1
handles, labels = axs[0][0].get_legend_handles_labels()
fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
fig.supylabel(f'Amplitude ({units})',fontsize=12)
fig.supxlabel('UTC Time(hr:min:sec)',fontsize=12)
fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
plt.subplots_adjust(left=0.11, bottom=0.075, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
plt.savefig('/Users/tnye/tsuquakes/plots/misc/observed/acc_wf.png', dpi=300)
plt.close()


#%%

# Velocity spectra figure 
units = 'm'
ylim = 6*10**-15, 8*10**-2
xlim = .002, 10
fig, axs = plt.subplots(3,3,figsize=(10,8))
k = 0
for i in range(3):
    for j in range(3):
        if k+1 <= len(obs_vel):
            axs[i][j].loglog(vel_freqs[k],vel_spec[k],lw=1,ls='-')
            axs[i][j].grid(linestyle='--')
            axs[i][j].text(0.8,5E-2,'rotd50',transform=axs[i][j].transAxes,size=10)
            axs[i][j].tick_params(axis='both', which='major', labelsize=10)
            axs[i][j].set_xlim(xlim)
            axs[i][j].set_ylim(ylim)
            axs[i][j].set_title(sm_names[k],fontsize=10)
            if i < 3 -1:
                axs[i][j].set_xticklabels([])
            if i == 3 -2 and j == 0:
                axs[i][j].set_xticklabels([])
            if j > 0:
                axs[i][j].set_yticklabels([])
            k += 1
handles, labels = axs[0][0].get_legend_handles_labels()
fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
fig.supylabel(f'Amplitude ({units})',fontsize=12)
fig.supxlabel('Frequency (Hz)',fontsize=12)
fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
plt.subplots_adjust(left=0.11, bottom=0.075, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
plt.savefig('/Users/tnye/tsuquakes/plots/misc/observed/vel_FAS.png', dpi=300)
plt.close()


# Velocity waveforms figure 
units = 'm/s'
fig, axs = plt.subplots(3,3,figsize=(10,8))
k = 0
for i in range(3):
    for j in range(3):
        if k+1 <= len(obs_vel):
            axs[i][j].plot(vel_times[k],vel_amps[k],lw=1,ls='-')
            axs[i][j].grid(linestyle='--')
            axs[i][j].text(0.8,5E-2,'rotd50',transform=axs[i][j].transAxes,size=10)
            axs[i][j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            axs[i][j].tick_params(axis='both', which='major', labelsize=10)
            axs[i][j].yaxis.offsetText.set_fontsize(10)
            axs[i][j].set_title(sm_names[k],fontsize=10)
            axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            if i < 3 -1:
                axs[i][j].set_xticklabels([])
            if i == 3 -2 and j == 0:
                axs[i][j].set_xticklabels([])
            k += 1
handles, labels = axs[0][0].get_legend_handles_labels()
fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
fig.supylabel(f'Amplitude ({units})',fontsize=12)
fig.supxlabel('UTC Time(hr:min:sec)',fontsize=12)
fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
plt.subplots_adjust(left=0.11, bottom=0.075, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
plt.savefig('/Users/tnye/tsuquakes/plots/misc/observed/vel_wf.png', dpi=300)
plt.close()
 

#%%

# Displacement spectra figure 
units = 'm*s'
ylim = 10**-5, 9*10**-1
xlim = 0.004, 5*10**-1
fig, axs = plt.subplots(3,3,figsize=(10,8))
k = 0
for i in range(3):
    for j in range(3):
        if k+1 <= len(obs_disp):
            axs[i][j].loglog(disp_freqs[k],disp_spec[k],lw=1,ls='-')
            axs[i][j].grid(linestyle='--')
            axs[i][j].text(0.5,5E-2,'eucnorm-3comp',transform=axs[i][j].transAxes,size=10)
            axs[i][j].tick_params(axis='both', which='major', labelsize=10)
            axs[i][j].set_xlim(xlim)
            axs[i][j].set_ylim(ylim)
            axs[i][j].set_title(sm_names[k],fontsize=10)
            if i < 1:
                axs[i][j].set_xticklabels([])
            if i == 1 and j == 0:
                axs[i][j].set_xticklabels([])
            if j > 0:
                axs[i][j].set_yticklabels([])
            k += 1
handles, labels = axs[0][0].get_legend_handles_labels()
fig.delaxes(axs[2][1])
fig.delaxes(axs[2][2])
fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
fig.supylabel(f'Amplitude ({units})',fontsize=12)
fig.supxlabel('Frequency (Hz)',fontsize=12)
fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
plt.subplots_adjust(left=0.11, bottom=0.075, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
plt.savefig('/Users/tnye/tsuquakes/plots/misc/observed/disp_FAS.png', dpi=300)
plt.close()


# Displacement waveforms figure 
units = 'm'
fig, axs = plt.subplots(3,3,figsize=(10,8))
k = 0
for i in range(3):
    for j in range(3):
        if k+1 <= len(obs_disp):
            axs[i][j].plot(disp_times[k],disp_amps[k],lw=1,ls='-')
            axs[i][j].grid(linestyle='--')
            axs[i][j].text(0.5,5E-2,'eucnorm-3comp',transform=axs[i][j].transAxes,size=10)
            axs[i][j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            axs[i][j].tick_params(axis='both', which='major', labelsize=10)
            axs[i][j].yaxis.offsetText.set_fontsize(10)
            axs[i][j].set_title(sm_names[k],fontsize=10)
            axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            if i < 1:
                axs[i][j].set_xticklabels([])
            if i == 1 and j == 0:
                axs[i][j].set_xticklabels([])
            k += 1
handles, labels = axs[0][0].get_legend_handles_labels()
fig.delaxes(axs[2][1])
fig.delaxes(axs[2][2])
fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
fig.supylabel(f'Amplitude ({units})',fontsize=12)
fig.supxlabel('UTC Time(hr:min:sec)',fontsize=12)
fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
plt.subplots_adjust(left=0.11, bottom=0.075, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
plt.savefig('/Users/tnye/tsuquakes/plots/misc/observed/disp_wf.png', dpi=300)
plt.close()
