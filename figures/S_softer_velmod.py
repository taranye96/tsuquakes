#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:35:25 2023

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from obspy import read
from glob import glob
from mtspec import mtspec
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator, ScalarFormatter, FormatStrFormatter
import signal_average_fns as avg
import IM_fns


#%% 

velmod_0 = np.genfromtxt('/Users/tnye/FakeQuakes/files/velmods/mentawai_v0.mod')
velmod_1 = np.genfromtxt('/Users/tnye/FakeQuakes/files/velmods/mentawai_soft1.mod')
velmod_2 = np.genfromtxt('/Users/tnye/FakeQuakes/files/velmods/mentawai_soft2.mod')
velmod_3 = np.genfromtxt('/Users/tnye/FakeQuakes/files/velmods/mentawai_soft3.mod')


d_0 = []
d_1 = []
d_2 = []
d_3 = []

for i in range(len(velmod_0)):
    if i == 0:
        d_0.append(0)
    else:
        d_0.append(np.sum(velmod_0[:i,0]))
for i in range(len(velmod_1)):
    if i == 0:
        d_1.append(0)
    else:
        d_1.append(np.sum(velmod_1[:i,0]))
for i in range(len(velmod_2)):
    if i == 0:
        d_2.append(0)
    else:
        d_2.append(np.sum(velmod_2[:i,0]))    
for i in range(len(velmod_3)):
    if i == 0:
        d_3.append(0)
    else:
        d_3.append(np.sum(velmod_3[:i,0]))     


#%%

def calc_spectra(stream):

    tr = stream[0]
    data = tr.data
    delta = tr.stats.delta
    samprate = tr.stats.sampling_rate
    npts = tr.stats.npts
    nyquist = 0.5 * samprate

    amp_squared, freq =  mtspec(data, delta=delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=npts, quadratic=True)

    amp = np.sqrt(amp_squared)
    
    return(freq, amp)


lf_stn_names = ['PKRT','SLBU','SMGY']
lf_hypdist = [161,82,98]

N = 3

################################ Observed data ################################
obs_dir = '/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/events'
obs_files = [f'{obs_dir}/PKRT.LXE.mseed',f'{obs_dir}/PKRT.LXN.mseed',f'{obs_dir}/PKRT.LXZ.mseed',
             f'{obs_dir}/SLBU.LXE.mseed',f'{obs_dir}/SLBU.LXN.mseed',f'{obs_dir}/SLBU.LXZ.mseed',
             f'{obs_dir}/SMGY.LXE.mseed',f'{obs_dir}/SMGY.LXN.mseed',f'{obs_dir}/SMGY.LXZ.mseed']
obs_grouped = [obs_files[n:n+N] for n in range(0, len(obs_files), N)]
obs_spec = []
obs_freq = []

for i in range(len(obs_grouped)):
    
    stE_obs = read(obs_grouped[i][0])
    stN_obs = read(obs_grouped[i][1])
    stZ_obs = read(obs_grouped[i][2])
    
    avg_obs = avg.get_eucl_norm_3comp(stE_obs[0].data, stN_obs[0].data, stZ_obs[0].data)
    
    freq, E_amp = calc_spectra(stE_obs)
    freq, N_amp = calc_spectra(stN_obs)
    NE_data = np.sqrt(E_amp**2 + N_amp**2)
    obs_spec.append(NE_data)
    obs_freq.append(freq)

################################ Synthetic data ###############################

disp_syn_soft0 = sorted(glob('/Users/tnye/FakeQuakes/simulations/test_runs_m7.8/standard/output/waveforms/mentawai.000000/*.LY*'))
disp_syn_soft1 = sorted(glob('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft1/output/waveforms/mentawai.000000/*.LY*'))
disp_syn_soft2 = sorted(glob('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft2/output/waveforms/mentawai.000000/*.LY*'))
disp_syn_soft3 = sorted(glob('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft3/output/waveforms/mentawai.000000/*.LY*'))
disp_syn_soft4 = sorted(glob('/Users/tnye/FakeQuakes/simulations/soft_velmod_test/soft4/output/waveforms/mentawai.000000/*.LY*'))

# Group all files by station
disp_syn_soft0_grouped = [disp_syn_soft0[n:n+N] for n in range(0, len(disp_syn_soft0), N)]
disp_syn_soft1_grouped = [disp_syn_soft1[n:n+N] for n in range(0, len(disp_syn_soft1), N)]
disp_syn_soft2_grouped = [disp_syn_soft2[n:n+N] for n in range(0, len(disp_syn_soft2), N)]
disp_syn_soft3_grouped = [disp_syn_soft3[n:n+N] for n in range(0, len(disp_syn_soft3), N)]

disp_syn_files_soft0 = np.array([disp_syn_soft0_grouped[10],disp_syn_soft0_grouped[14],disp_syn_soft0_grouped[15]])
disp_syn_files_soft1 = np.array([disp_syn_soft1_grouped[3],disp_syn_soft1_grouped[5],disp_syn_soft1_grouped[6]])
disp_syn_files_soft2 = np.array([disp_syn_soft2_grouped[3],disp_syn_soft2_grouped[5],disp_syn_soft2_grouped[6]])
disp_syn_files_soft3 = np.array([disp_syn_soft3_grouped[3],disp_syn_soft3_grouped[5],disp_syn_soft3_grouped[6]])

syn_disp_soft0_spec = []
syn_disp_soft1_spec = []
syn_disp_soft2_spec = []
syn_disp_soft3_spec = []

syn_lf_freq = []

  
for i in range(len(disp_syn_files_soft1)):
    
    stE_disp_soft0 = read(disp_syn_files_soft0[i][0])
    stN_disp_soft0 = read(disp_syn_files_soft0[i][1])
    stZ_disp_soft0 = read(disp_syn_files_soft0[i][2])
    stE_disp_soft1 = read(disp_syn_files_soft1[i][0])
    stN_disp_soft1 = read(disp_syn_files_soft1[i][1])
    stZ_disp_soft1 = read(disp_syn_files_soft1[i][2])
    stE_disp_soft2 = read(disp_syn_files_soft2[i][0])
    stN_disp_soft2 = read(disp_syn_files_soft2[i][1])
    stZ_disp_soft2 = read(disp_syn_files_soft2[i][2])
    stE_disp_soft3 = read(disp_syn_files_soft3[i][0])
    stN_disp_soft3 = read(disp_syn_files_soft3[i][1])
    stZ_disp_soft3 = read(disp_syn_files_soft3[i][2])
    
    avg_disp_soft0 = avg.get_eucl_norm_3comp(stE_disp_soft0[0].data, stN_disp_soft0[0].data, stZ_disp_soft0[0].data)
    avg_disp_soft1 = avg.get_eucl_norm_3comp(stE_disp_soft1[0].data, stN_disp_soft1[0].data, stZ_disp_soft1[0].data)
    avg_disp_soft2 = avg.get_eucl_norm_3comp(stE_disp_soft2[0].data, stN_disp_soft2[0].data, stZ_disp_soft2[0].data)
    avg_disp_soft3 = avg.get_eucl_norm_3comp(stE_disp_soft3[0].data, stN_disp_soft3[0].data, stZ_disp_soft3[0].data)
    
    freq, E_amp = calc_spectra(stE_disp_soft0)
    freq, N_amp = calc_spectra(stN_disp_soft0)
    NE_data = np.sqrt(E_amp**2 + N_amp**2)
    syn_disp_soft0_spec.append(NE_data)
    
    freq, E_amp = calc_spectra(stE_disp_soft1)
    freq, N_amp = calc_spectra(stN_disp_soft1)
    NE_data = np.sqrt(E_amp**2 + N_amp**2)
    syn_disp_soft1_spec.append(NE_data)
    syn_lf_freq.append(freq)

    freq, E_amp = calc_spectra(stE_disp_soft2)
    freq, N_amp = calc_spectra(stN_disp_soft2)
    NE_data = np.sqrt(E_amp**2 + N_amp**2)
    syn_disp_soft2_spec.append(NE_data)

    freq, E_amp = calc_spectra(stE_disp_soft3)
    freq, N_amp = calc_spectra(stN_disp_soft3)
    NE_data = np.sqrt(E_amp**2 + N_amp**2)
    syn_disp_soft3_spec.append(NE_data)
    

#%%

layout = [
    ["A", "B"],
    ["A", "C"],
    ["A", "D"]]

fig, axs = plt.subplot_mosaic(layout, figsize=(6,6))

axs['A'].plot(velmod_0[:,2],d_0,lw=1,label='Orig velmod')
axs['A'].plot(velmod_1[:,2],d_1,lw=1,label='Softer velmod 1')
axs['A'].plot(velmod_2[:,2],d_2,lw=1,label='Softer velmod 2')
axs['A'].plot(velmod_3[:,2],d_3,lw=1,label='Softer velmod 3')
# axs['A'].plot(velmod_4[:,2],d_4,lw=1,label='Softer velmod 4')
axs['A'].grid(alpha=0.25)
axs['A'].invert_yaxis()
axs['A'].set_xlim(xmax=3.5)
axs['A'].set_ylim(1,0)
axs['A'].set_xlabel('Vp (km/s)')
axs['A'].set_ylabel('Depth (m)')

axs['B'].loglog(syn_lf_freq[0],syn_disp_soft0_spec[0],alpha=0.7,lw=1)
axs['B'].loglog(syn_lf_freq[0],syn_disp_soft1_spec[0],alpha=0.7,lw=1)
axs['B'].loglog(syn_lf_freq[0],syn_disp_soft2_spec[0],alpha=0.7,lw=1)
axs['B'].loglog(syn_lf_freq[0],syn_disp_soft3_spec[0],alpha=0.7,lw=1)
axs['B'].loglog(obs_freq[0],obs_spec[0],c='k',alpha=0.7,lw=1)
axs['B'].tick_params(axis='both', which='major', labelsize=10)
axs['B'].text(0.98,5E-2,f'Hypdist={int(lf_hypdist[0])}km',horizontalalignment='right',transform=axs['B'].transAxes,size=10)
axs['B'].text(0.98,0.9,lf_stn_names[0],transform=axs['B'].transAxes,size=10,horizontalalignment='right')
axs['B'].grid(alpha=0.25)
axs['B'].set_xlim(0.004,0.5)
axs['B'].tick_params(which='minor',bottom=False,left=False)

axs['C'].loglog(syn_lf_freq[1],syn_disp_soft0_spec[1],alpha=0.7,lw=1)
axs['C'].loglog(syn_lf_freq[1],syn_disp_soft1_spec[1],alpha=0.7,lw=1)
axs['C'].loglog(syn_lf_freq[1],syn_disp_soft2_spec[1],alpha=0.7,lw=1)
axs['C'].loglog(syn_lf_freq[1],syn_disp_soft3_spec[1],alpha=0.7,lw=1)
axs['C'].loglog(obs_freq[1],obs_spec[1],c='k',alpha=0.7,lw=1)
axs['C'].tick_params(axis='both', which='major', labelsize=10)
axs['C'].text(0.98,5E-2,f'Hypdist={int(lf_hypdist[1])}km',horizontalalignment='right',transform=axs['C'].transAxes,size=10)
axs['C'].text(0.98,0.9,lf_stn_names[1],transform=axs['C'].transAxes,size=10,horizontalalignment='right')
axs['C'].grid(alpha=0.25)
axs['C'].set_xlim(0.004,0.5)
axs['C'].set_ylabel('Amplitude (m*s)')
axs['C'].tick_params(which='minor',bottom=False,left=False)

axs['D'].loglog(syn_lf_freq[2],syn_disp_soft0_spec[2],alpha=0.7,lw=1)
axs['D'].loglog(syn_lf_freq[2],syn_disp_soft1_spec[2],alpha=0.7,lw=1)
axs['D'].loglog(syn_lf_freq[2],syn_disp_soft2_spec[2],alpha=0.7,lw=1)
axs['D'].loglog(syn_lf_freq[2],syn_disp_soft3_spec[2],alpha=0.7,lw=1)
# axs['D'].loglog(syn_lf_freq[2],syn_disp_soft4_spec[2],alpha=0.7,lw=1)
axs['D'].loglog(obs_freq[2],obs_spec[2],c='k',alpha=0.7,lw=1)
axs['D'].tick_params(axis='both', which='major', labelsize=10)
axs['D'].text(0.98,5E-2,f'Hypdist={int(lf_hypdist[2])}km',horizontalalignment='right',transform=axs['D'].transAxes,size=10)
axs['D'].text(0.98,0.9,lf_stn_names[2],transform=axs['D'].transAxes,size=10,horizontalalignment='right')
axs['D'].grid(alpha=0.25)
axs['D'].set_xlim(0.004,0.5)
axs['D'].tick_params(which='minor',bottom=False,left=False)
axs['D'].set_xlabel('Frequency (Hz)')

handles, labels = axs['A'].get_legend_handles_labels()
axs['A'].legend(handles,labels,loc='upper left',bbox_to_anchor=(0.325,-0.125),facecolor='white',
                frameon=True,fontsize=10,title_fontsize=10,ncol=2,markerscale=2)
plt.subplots_adjust(top=0.98, bottom=0.2, right=0.98, left=0.1, wspace=0.35, hspace=0.3)

plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/S_softer_velmod.png',dpi=300)



#%%


velmod_0 = np.genfromtxt('/Users/tnye/FakeQuakes/files/velmods/mentawai_v0.mod')
d_0 = []

for i in range(len(velmod_0)):
    if i == 0:
        d_0.append(0)
    else:
        d_0.append(np.sum(velmod_0[:i,0]))

d_0.append(80)

fig, ax = plt.subplots(1,1, figsize=(3,4))
vp = np.append(velmod_0[:,2],velmod_0[-1,2])
vs = np.append(velmod_0[:,3],velmod_0[-1,3])

ax.plot(vp,d_0,lw=1,label='Vp')
ax.plot(vs,d_0,lw=1,label='Vs')
ax.set_xlim(xmax=10)
ax.set_xlabel('V (km/s)')
ax.set_ylabel('Depth (km)')
ax.invert_yaxis()
ax.legend()
ax.xaxis.set_major_locator(MultipleLocator(1))

plt.subplots_adjust(top=0.95,left=0.3)

