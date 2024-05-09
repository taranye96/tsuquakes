#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:40:52 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from obspy import read
from glob import glob
from os import path, makedirs
from mtspec import mtspec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import LogLocator, MultipleLocator, ScalarFormatter, FormatStrFormatter
import matplotlib as mpl
from matplotlib import ticker
# sys.path.insert(0, '/Users/tnye/tsuquakes/code/processing/') # location of src 
import signal_average_fns as avg
from rotd50 import compute_rotd50
import latex

hf_stn_names = ['PPSI','LHSI','SBSI']
lf_stn_names = ['BSAT','SLBU','PKRT']
hf_hypdist = [82,381,545]
lf_hypdist = [52,82,161]

################################ Observed data ################################

all_disp_obs_files = sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/average/eucnorm_3comp/disp/*'))
all_acc_obs_files = sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/average/rotd50/acc/*'))
all_vel_obs_files = sorted(glob(f'/Users/tnye/tsuquakes/data/waveforms/average/rotd50/vel/*'))

disp_obs_files = np.array([all_disp_obs_files[0],all_disp_obs_files[5],all_disp_obs_files[2]])
acc_obs_files = np.array([all_acc_obs_files[7],all_acc_obs_files[2],all_acc_obs_files[8]])
vel_obs_files = np.array([all_vel_obs_files[7],all_vel_obs_files[2],all_vel_obs_files[8]])

obs_disp_amps = []
obs_acc_amps = []
obs_vel_amps = []
obs_hf_times = []
obs_lf_times = []

obs_disp_spec = []
obs_acc_spec = []
obs_vel_spec = []
obs_hf_freq = []
obs_lf_freq = []

for i in range(len(disp_obs_files)):
    obs_disp_amps.append(read(disp_obs_files[i])[0].data.tolist())
    obs_lf_times.append(read(disp_obs_files[i])[0].times('matplotlib').tolist())
    obs_acc_amps.append(read(acc_obs_files[i])[0].data.tolist())
    obs_hf_times.append(read(acc_obs_files[i])[0].times('matplotlib').tolist())
    obs_vel_amps.append(read(vel_obs_files[i])[0].data.tolist())
    
    hf_delta = read(acc_obs_files[i])[0].stats.delta
    hf_samprate = read(acc_obs_files[i])[0].stats.sampling_rate
    hf_nyquist = 0.5 * hf_samprate
    hf_npts = read(acc_obs_files[i])[0].stats.npts
    lf_delta = read(disp_obs_files[i])[0].stats.delta
    lf_samprate = read(disp_obs_files[i])[0].stats.sampling_rate
    lf_nyquist = 0.5 * lf_samprate
    lf_npts = read(disp_obs_files[i])[0].stats.npts
    
    acc_amp2, hf_freq =  mtspec(read(acc_obs_files[i])[0].data, delta=hf_delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=hf_npts, quadratic=True)
    acc_amp = np.sqrt(acc_amp2)
    vel_amp2, hf_freq =  mtspec(read(vel_obs_files[i])[0].data, delta=hf_delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=hf_npts, quadratic=True)
    vel_amp = np.sqrt(vel_amp2)
    disp_amp2, lf_freq =  mtspec(read(disp_obs_files[i])[0].data, delta=lf_delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_npts, quadratic=True)
    disp_amp = np.sqrt(disp_amp2)
    
    obs_disp_spec.append(disp_amp)
    obs_lf_freq.append(lf_freq)
    obs_acc_spec.append(acc_amp)
    obs_hf_freq.append(hf_freq)
    obs_vel_spec.append(vel_amp)

################################ Synthetic data ###############################

all_disp_syn_files = sorted(glob('/Users/tnye/FakeQuakes/simulations/test_runs_m7.8/standard/processed_wfs/disp_noise/mentawai.000000/*'))
all_acc_syn_files = sorted(glob('/Users/tnye/FakeQuakes/simulations/test_runs_m7.8/standard/processed_wfs/acc/mentawai.000000/*'))
all_vel_syn_files = sorted(glob('/Users/tnye/FakeQuakes/simulations/test_runs_m7.8/standard/processed_wfs/vel/mentawai.000000/*'))

# Group all files by station
N = 3
all_disp_syn_grouped = [all_disp_syn_files[n:n+N] for n in range(0, len(all_disp_syn_files), N)]
all_acc_syn_grouped = [all_acc_syn_files[n:n+N] for n in range(0, len(all_acc_syn_files), N)]
all_vel_syn_grouped = [all_vel_syn_files[n:n+N] for n in range(0, len(all_vel_syn_files), N)]

disp_syn_files = np.array([all_disp_syn_grouped[0],all_disp_syn_grouped[5],all_disp_syn_grouped[2]])
acc_syn_files = np.array([all_acc_syn_grouped[7],all_acc_syn_grouped[2],all_acc_syn_grouped[8]])
vel_syn_files = np.array([all_vel_syn_grouped[7],all_vel_syn_grouped[2],all_vel_syn_grouped[8]])

syn_disp_amps = []
syn_acc_amps = []
syn_vel_amps = []
syn_hf_times = []
syn_lf_times = []

syn_disp_spec = []
syn_acc_spec = []
syn_vel_spec = []
syn_hf_freq = []
syn_lf_freq = []

for i in range(len(disp_syn_files)):
    
    stE_disp = read(disp_syn_files[i][0])
    stN_disp = read(disp_syn_files[i][1])
    stZ_disp = read(disp_syn_files[i][2])
    stE_acc = read(acc_syn_files[i][0])
    stN_acc = read(acc_syn_files[i][1])
    stE_vel = read(vel_syn_files[i][0])
    stN_vel = read(vel_syn_files[i][1])
    
    avg_disp = avg.get_eucl_norm_3comp(stE_disp[0].data, stN_disp[0].data, stZ_disp[0].data)
    avg_acc = compute_rotd50(stE_acc[0].data,stN_acc[0].data)
    avg_vel = compute_rotd50(stE_vel[0].data,stN_vel[0].data)
    syn_disp_amps.append(avg_disp)
    syn_lf_times.append(stE_disp[0].times('matplotlib'))
    syn_acc_amps.append(avg_acc)
    syn_vel_amps.append(avg_vel)
    syn_hf_times.append(stE_acc[0].times('matplotlib'))
    
    hf_delta = 0.01
    hf_samprate = 100
    hf_nyquist = 0.5 * hf_samprate
    hf_npts = read(acc_syn_files[i][0])[0].stats.npts
    lf_delta = 0.5
    lf_samprate = 2
    lf_nyquist = 0.5 * lf_samprate
    lf_npts = read(disp_syn_files[i][0])[0].stats.npts
    
    acc_amp2, hf_freq =  mtspec(avg_acc, delta=hf_delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=hf_npts, quadratic=True)
    acc_amp = np.sqrt(acc_amp2)
    vel_amp2, hf_freq =  mtspec(avg_vel, delta=hf_delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=hf_npts, quadratic=True)
    vel_amp = np.sqrt(vel_amp2)
    disp_amp2, lf_freq =  mtspec(avg_disp, delta=lf_delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_npts, quadratic=True)
    disp_amp = np.sqrt(disp_amp2)
    
    syn_disp_spec.append(disp_amp)
    syn_lf_freq.append(lf_freq)
    syn_acc_spec.append(acc_amp)
    syn_hf_freq.append(hf_freq)
    syn_vel_spec.append(vel_amp)


#%% 
#### Waveforms and spectra in one figure ####

labels=[r'$\bf{(a)}$',r'$\bf{(b)}$',r'$\bf{(c)}$',r'$\bf{(d)}$',r'$\bf{(e)}$',r'$\bf{(f)}$',
        r'$\bf{(g)}$',r'$\bf{(h)}$',r'$\bf{(i)}$',r'$\bf{(j)}$',r'$\bf{(k)}$',r'$\bf{(l)}$']


# Waveform figure
# fig, axs = plt.subplots(4,3,figsize=(6.5,7)) 
layout = [["a", "b", "c"],["d", "e", "f"],["null","null","null"],["g", "h", "i"],["j", "k", "l"]]
fig, axs = plt.subplot_mosaic(layout, figsize=(6.5,7), gridspec_kw={'height_ratios':[1,1,0.1,1,1]})

j=0

# Disp waveforms
for i in range(3):
    axs[layout[0][i]].plot(syn_lf_times[i],syn_disp_amps[i],color='C1',alpha=0.7,lw=0.75,label='synthetic')
    axs[layout[0][i]].plot(obs_lf_times[i],obs_disp_amps[i],'steelblue',alpha=0.7,lw=0.75,label='observed')
    axs[layout[0][i]].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[layout[0][i]].text(0.98,5E-2,r'$R_{hyp}$'+f'={int(lf_hypdist[i])}km',horizontalalignment='right',transform=axs[layout[0][i]].transAxes,size=10)
    axs[layout[0][i]].text(0.98,0.85,lf_stn_names[i],transform=axs[layout[0][i]].transAxes,size=10,horizontalalignment='right')
    axs[layout[0][i]].grid(alpha=0.25)
    axs[layout[0][i]].text(-0.05,1.1,labels[j],transform=axs[layout[0][i]].transAxes,fontsize=10,va='top',ha='right')
    # axs[layout[0][i]].tick_params('x', labelbottom=False)
    if np.max(np.abs(syn_disp_amps[i].tolist()+obs_disp_amps[i])) < 0.2:
       axs[layout[0][i]].yaxis.set_major_locator(MultipleLocator(0.05))
    else:
       axs[layout[0][i]].yaxis.set_major_locator(MultipleLocator(0.1))
    if i == 1:
        axs[layout[0][i]].set_title(r'$\bf{Waveforms}$',pad=10,fontsize=11)
    j+=1
# axs['a'].set_ylabel('Amplitude (m)')
    
# Acc waveforms
for i in range(3):
    # axs[layout[1][i]].plot(syn_hf_times[i],syn_acc_amps[i],color='C1',alpha=0.7,lw=0.75,label='synthetic')
    axs[layout[1][i]].plot(obs_hf_times[i],obs_acc_amps[i],'steelblue',alpha=0.7,lw=0.5,label='observed')
    axs[layout[1][i]].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[layout[1][i]].tick_params(axis='both', which='major', labelsize=10)
    axs[layout[1][i]].text(0.98,5E-2,r'$R_{hyp}$'+f'={int(hf_hypdist[i])}km',horizontalalignment='right',transform=axs[layout[1][i]].transAxes,size=10)
    axs[layout[1][i]].text(0.98,0.85,hf_stn_names[i],transform=axs[layout[1][i]].transAxes,size=10,horizontalalignment='right')
    axs[layout[1][i]].grid(alpha=0.25)   
    axs[layout[1][i]].text(-0.05,1.1,labels[j],transform=axs[layout[1][i]].transAxes,fontsize=10,va='top',ha='right')
    if np.max(np.abs(syn_acc_amps[i])) > 0.025 and np.max(np.abs(syn_acc_amps[i])) < 0.1:
        axs[layout[1][i]].yaxis.set_major_locator(MultipleLocator(0.05))
    elif np.max(np.abs(syn_acc_amps[i])) < 0.025:
        axs[layout[1][i]].yaxis.set_major_locator(MultipleLocator(0.01))
    else:
        axs[layout[1][i]].yaxis.set_major_locator(MultipleLocator(0.5))
    j+=1
axs["e"].set_xlabel('October 25, 2010 UTC Time(hr:min)',fontsize=10)
# axs['d'].set_ylabel(r'Amplitude (m/s$^{2}$)')

# Disp spectra
for i in range(3):
    axs[layout[3][i]].loglog(syn_lf_freq[i],syn_disp_spec[i],color='C1',alpha=0.7,lw=0.75,label='synthetic')
    axs[layout[3][i]].loglog(obs_lf_freq[i],obs_disp_spec[i],'steelblue',alpha=0.7,lw=0.75,label='observed')
    axs[layout[3][i]].tick_params(axis='both', which='major', labelsize=10)
    axs[layout[3][i]].text(0.98,5E-2,r'$R_{hyp}$'+f'={int(lf_hypdist[i])}km',horizontalalignment='right',transform=axs[layout[3][i]].transAxes,size=10)
    axs[layout[3][i]].text(0.98,0.85,lf_stn_names[i],transform=axs[layout[3][i]].transAxes,size=10,horizontalalignment='right')
    axs[layout[3][i]].grid(alpha=0.25)
    axs[layout[3][i]].set_xlim(xmax=0.5)
    axs[layout[3][i]].text(-0.05,1.1,labels[j],transform=axs[layout[3][i]].transAxes,fontsize=10,va='top',ha='right')
    axs[layout[3][i]].set_ylim(2*10**-4,5)
    axs[layout[3][i]].set_xlim(0.004,1)
    axs[layout[3][i]].tick_params(which='minor',bottom=False,left=False)
    # if i !=0:
    #     axs[layout[3][i]].yaxis.set_tick_params(labelleft=False)
    if i == 1:
        axs[layout[3][i]].set_title(r'$\bf{Fourier}$ $\bf{Amplitude}$ $\bf{Spectra}$',pad=10,fontsize=11)
    j+=1
# axs['g'].set_ylabel('Amplitude (m*s)')

# Acc spectra
for i in range(3):
    axs[layout[4][i]].loglog(syn_hf_freq[i],syn_acc_spec[i],color='C1',alpha=0.7,lw=0.75,label='synthetic')
    axs[layout[4][i]].loglog(obs_hf_freq[i],obs_acc_spec[i],'steelblue',alpha=0.7,lw=0.75,label='observed')
    axs[layout[4][i]].tick_params(axis='both', which='major', labelsize=10)
    axs[layout[4][i]].text(0.98,5E-2,r'$R_{hyp}$'+f'={int(hf_hypdist[i])}km',horizontalalignment='right',transform=axs[layout[4][i]].transAxes,size=10)
    axs[layout[4][i]].text(0.98,0.85,hf_stn_names[i],transform=axs[layout[4][i]].transAxes,size=10,horizontalalignment='right')
    axs[layout[4][i]].grid(alpha=0.25)
    axs[layout[4][i]].set_xlim(0.01,10)
    axs[layout[4][i]].set_ylim(2*10**-12,8*10**-1)
    axs[layout[4][i]].text(-0.05,1.1,labels[j],transform=axs[layout[4][i]].transAxes,fontsize=10,va='top',ha='right')
    # if i == 1:
        # axs[layout[4][i]].set_title(r'$\bf{Acceleration}$',pad=10)   
    # if i !=0:
    #     axs[layout[4][i]].yaxis.set_tick_params(labelleft=False)
    j+=1
axs["k"].set_xlabel('Fequency (Hz)',fontsize=10)
# axs['j'].set_ylabel('Amplitude (m/s)')

axs["null"].remove()
axs["k"].legend(loc='upper center', bbox_to_anchor=(0.5, -0.6),fancybox=False, shadow=False, ncol=2)
fig.supylabel(f'Amplitude',fontsize=10,x=0.01,y=0.575)
# fig.supxlabel('October 25, 2010 UTC Time(hr:min)',fontsize=11,x=0.545,y=0.115,va='bottom')
plt.subplots_adjust(left=0.115, bottom=0.15, right=0.975, top=0.95, wspace=0.4, hspace=0.55)
# plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/wf-spectra_comparison_standard.png',dpi=300)
    
    
#%%

labels=[r'$\bf{(a)}$',r'$\bf{(b)}$',r'$\bf{(c)}$',r'$\bf{(d)}$',r'$\bf{(e)}$',r'$\bf{(f)}$']


# Waveform figure
fig, axs = plt.subplots(2,3,figsize=(6.5,5)) 

j=0

# Disp
for i in range(3):
    axs[0][i].plot(syn_lf_times[i],syn_disp_amps[i],color='C1',alpha=0.7,lw=0.75,label='synthetic')
    axs[0][i].plot(obs_lf_times[i],obs_disp_amps[i],'steelblue',alpha=0.7,lw=0.75,label='observed')
    axs[0][i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[0][i].text(0.98,5E-2,f'Hypdist={int(lf_hypdist[i])}km',horizontalalignment='right',transform=axs[0][i].transAxes,size=10)
    axs[0][i].text(0.98,0.9,lf_stn_names[i],transform=axs[0][i].transAxes,size=10,horizontalalignment='right')
    axs[0][i].grid(alpha=0.25)
    axs[0][i].text(-0.05,1.0,labels[j],transform=axs[0][i].transAxes,fontsize=10,va='top',ha='right')
    if np.max(np.abs(syn_disp_amps[i].tolist()+obs_disp_amps[i])) < 0.2:
       axs[0][i].yaxis.set_major_locator(MultipleLocator(0.05))
    else:
       axs[0][i].yaxis.set_major_locator(MultipleLocator(0.1))
    if i == 1:
        axs[0][i].set_title(r'$\bf{Displacement}$',pad=10)
    j+=1
    
    
# Acc
for i in range(3):
    axs[1][i].plot(syn_hf_times[i],syn_acc_amps[i],color='C1',alpha=0.7,lw=0.75,label='synthetic')
    axs[1][i].plot(obs_hf_times[i],obs_acc_amps[i],'steelblue',alpha=0.7,lw=0.75,label='observed')
    axs[1][i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[1][i].tick_params(axis='both', which='major', labelsize=10)
    axs[1][i].text(0.98,5E-2,f'Hypdist={int(hf_hypdist[i])}km',horizontalalignment='right',transform=axs[1][i].transAxes,size=10)
    axs[1][i].text(0.98,0.9,hf_stn_names[i],transform=axs[1][i].transAxes,size=10,horizontalalignment='right')
    axs[1][i].grid(alpha=0.25)   
    axs[1][i].text(-0.05,1.0,labels[j],transform=axs[1][i].transAxes,fontsize=10,va='top',ha='right')
    if np.max(np.abs(syn_acc_amps[i])) > 0.025 and np.max(np.abs(syn_acc_amps[i])) < 0.1:
        axs[1][i].yaxis.set_major_locator(MultipleLocator(0.05))
    elif np.max(np.abs(syn_acc_amps[i])) < 0.025:
        axs[1][i].yaxis.set_major_locator(MultipleLocator(0.01))
    else:
        axs[1][i].yaxis.set_major_locator(MultipleLocator(0.5))
    if i == 1:
        axs[1][i].set_title(r'$\bf{Acceleration}$',pad=10)
    j+=1


        
handles, labels = axs[0][0].get_legend_handles_labels()
axs[1][1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),fancybox=False, shadow=False, ncol=2)
fig.supylabel(f'Amplitude',fontsize=11,x=0.01,y=0.575)
fig.supxlabel('October 25, 2010 UTC Time(hr:min)',fontsize=11,x=0.545,y=0.115,va='bottom')
plt.subplots_adjust(left=0.115, bottom=0.215, right=0.975, top=0.925, wspace=0.3, hspace=0.5)
plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/wf_comparison_standard.png',dpi=300)

#%%

fig, axs = plt.subplots(2,3,figsize=(6.5,5))
labels=[r'$\bf{(a)}$',r'$\bf{(b)}$',r'$\bf{(c)}$',r'$\bf{(d)}$',r'$\bf{(e)}$',r'$\bf{(f)}$']

j = 0

# Disp
for i in range(3):
    axs[0][i].loglog(syn_lf_freq[i],syn_disp_spec[i],color='C1',alpha=0.7,lw=0.75,label='synthetic')
    axs[0][i].loglog(obs_lf_freq[i],obs_disp_spec[i],'steelblue',alpha=0.7,lw=0.75,label='observed')
    axs[0][i].tick_params(axis='both', which='major', labelsize=10)
    axs[0][i].text(0.98,5E-2,f'Hypdist={int(lf_hypdist[i])}km',horizontalalignment='right',transform=axs[0][i].transAxes,size=10)
    axs[0][i].text(0.98,0.9,lf_stn_names[i],transform=axs[0][i].transAxes,size=10,horizontalalignment='right')
    axs[0][i].grid(alpha=0.25)
    axs[0][i].set_xlim(xmax=0.5)
    axs[0][i].text(-0.05,1.0,labels[j],transform=axs[0][i].transAxes,fontsize=10,va='top',ha='right')
    axs[0][i].set_ylim(2*10**-4,5)
    axs[0][i].tick_params(which='minor',bottom=False,left=False)
    if i !=0:
        axs[0][i].yaxis.set_tick_params(labelleft=False)
    if i == 1:
        axs[0][i].set_title(r'$\bf{Displacement}$',pad=10)
    j+=1

# Acc
for i in range(3):
    axs[1][i].loglog(syn_hf_freq[i],syn_acc_spec[i],color='C1',alpha=0.7,lw=0.75,label='synthetic')
    axs[1][i].loglog(obs_hf_freq[i],obs_acc_spec[i],'steelblue',alpha=0.7,lw=0.75,label='observed')
    axs[1][i].tick_params(axis='both', which='major', labelsize=10)
    axs[1][i].text(0.98,5E-2,f'Hypdist={int(hf_hypdist[i])}km',horizontalalignment='right',transform=axs[1][i].transAxes,size=10)
    axs[1][i].text(0.98,0.9,hf_stn_names[i],transform=axs[1][i].transAxes,size=10,horizontalalignment='right')
    axs[1][i].grid(alpha=0.25)
    axs[1][i].set_xlim(xmax=10)
    axs[1][i].set_ylim(2*10**-12,8*10**-1)
    axs[1][i].text(-0.05,1.0,labels[j],transform=axs[1][i].transAxes,fontsize=10,va='top',ha='right')
    if i == 1:
        axs[1][i].set_title(r'$\bf{Acceleration}$',pad=10)   
    if i !=0:
        axs[1][i].yaxis.set_tick_params(labelleft=False)
    j+=1

        
handles, labels = axs[0][0].get_legend_handles_labels()
axs[1][1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),fancybox=False, shadow=False, ncol=2)
fig.supylabel(f'Amplitude',fontsize=11,x=0.01,y=0.575)
fig.supxlabel('Frequency (Hz)',fontsize=11,x=0.545,y=0.115,va='bottom')
plt.subplots_adjust(left=0.115, bottom=0.215, right=0.975, top=0.925, wspace=0.3, hspace=0.5)
plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/fas_comparison_standard.png',dpi=300)
