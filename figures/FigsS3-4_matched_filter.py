#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:43:19 2023

@author: tnye
"""

# Imports
import numpy as np
import scipy.constants
from glob import glob
from obspy import read
from numpy import genfromtxt,where,r_,diff,interp
from mudpy import forward
import functions as filt 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
from rotd50 import compute_rotd50
import tsueqs_main_fns as tmf
import matched_filter_fns as filt

# Set parameters
order = 4
fc_low = 0.998
fc_high = np.flip(np.linspace(0.1,1,10))
fc_high = fc_high [1:]
g = scipy.constants.g

rupture = 'mentawai.000001'
stn = 'LHSI'
# stn = 'KASI'

home = '/Users/tnye/FakeQuakes/simulations/final_runs_m7.8/'
project_name = 'standard'

lf_wfs = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/final_runs_m7.8/standard/output/waveforms/{rupture}/{stn}.LY*'))
hf_wfs = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/final_runs_m7.8/standard/output/waveforms/{rupture}/{stn}.HN*mpi*'))

############################## Read in waveforms ##############################

lf_E = read(lf_wfs[0])
hf_E = read(hf_wfs[0])


########################## Differentiate disp to acc ##########################

lf_E_acc = lf_E.copy()

lf_E_acc[0].data=r_[0,diff(lf_E_acc[0].data)/lf_E_acc[0].stats.delta]
lf_E_acc[0].data=r_[0,diff(lf_E_acc[0].data)/lf_E_acc[0].stats.delta]


################################# Filter data #################################

fsample_low=1./lf_E_acc[0].stats.delta
fsample_hf=1./hf_E[0].stats.delta

lf_E_filt_1 = lf_E_acc.copy()
hf_E_filt_1 = hf_E.copy()

lf_E_filt_01 = lf_E_acc.copy()
hf_E_filt_01 = hf_E.copy()

lf_E_filt_1[0].data=filt.lowpass(lf_E_filt_1[0].data,0.998,fsample_low,4,zerophase=True)
hf_E_filt_1[0].data=filt.highpass(hf_E_filt_1[0].data,0.998,fsample_hf,4,zerophase=True)

lf_E_filt_01[0].data=filt.lowpass(lf_E_filt_01[0].data,0.1,fsample_low,4,zerophase=True)
hf_E_filt_01[0].data=filt.highpass(hf_E_filt_01[0].data,0.1,fsample_hf,4,zerophase=True)


################################ Matched Filter ###############################

bb_1 = filt.matched_filter(home, project_name, rupture, lf_E, hf_E, 0.998, 0.998, 4, True)
bb_01 = filt.matched_filter(home, project_name, rupture, lf_E, hf_E, 0.998, 0.1, 4, True)


############################## Calculate spectra ##############################

lf_freq, lf_amp_E = filt.calc_spec(lf_E[0].data, lf_E[0].stats.delta, lf_E[0].stats.npts)
lf_freq, lf_amp_E_acc = filt.calc_spec(lf_E_acc[0].data, lf_E_acc[0].stats.delta, lf_E_acc[0].stats.npts)

hf_freq, hf_amp_E = filt.calc_spec(hf_E[0].data, hf_E[0].stats.delta, hf_E[0].stats.npts)

lf_freq, lf_amp_E_filt_1 = filt.calc_spec(lf_E_filt_1[0].data, lf_E_filt_1[0].stats.delta, lf_E_filt_1[0].stats.npts)
hf_freq, hf_amp_E_filt_1 = filt.calc_spec(hf_E_filt_1[0].data, hf_E_filt_1[0].stats.delta, hf_E_filt_1[0].stats.npts)

lf_freq, lf_amp_E_filt_01 = filt.calc_spec(lf_E_filt_01[0].data, lf_E_filt_01[0].stats.delta, lf_E_filt_01[0].stats.npts)
hf_freq, hf_amp_E_filt_01 = filt.calc_spec(hf_E_filt_01[0].data, hf_E_filt_01[0].stats.delta, hf_E_filt_01[0].stats.npts)

bb_freq, bb_amp_1 = filt.calc_spec(bb_1[0].data, bb_1[0].stats.delta, bb_1[0].stats.npts)
bb_freq, bb_amp_01 = filt.calc_spec(bb_01[0].data, bb_01[0].stats.delta, bb_01[0].stats.npts)


############################## Get normalized amp #############################

lf_amp_norm = lf_E_acc[0].data/np.max(lf_E_acc[0].data)
hf_amp_norm = hf_E[0].data/np.max(hf_E[0].data)

lf_amp_filt_norm_1 = lf_E_filt_1[0].data/np.max(lf_E_filt_1[0].data)
hf_amp_filt_norm_1 = hf_E_filt_1[0].data/np.max(hf_E_filt_1[0].data)

lf_amp_filt_norm_01 = lf_E_filt_01[0].data/np.max(lf_E_filt_01[0].data)
hf_amp_filt_norm_01 = hf_E_filt_01[0].data/np.max(hf_E_filt_01[0].data)

bb_amp_norm_1 = bb_1[0].data/np.max(bb_1[0].data)
bb_amp_norm_01 = bb_01[0].data/np.max(bb_01[0].data)


#%%
################################## Figure 1fc #################################

# BB waveform and spectrum (filtered)
fig, axs = plt.subplots (3,2, figsize=(9,7))
axs[0,0].plot(hf_E[0].times(),hf_amp_norm,lw=1,alpha=0.7,c='purple',label='High-frequency')
axs[0,0].plot(lf_E_acc[0].times(),lf_amp_norm,lw=1,alpha=0.7,c='gold',label='Low-frequency')
axs[0,0].set_xlim(25,300)
axs[0,0].grid(alpha=0.5)
axs[0,0].legend(loc='lower right')
axs[0,0].text(0.025, 0.875, 'Unfiltered', transform=axs[0,0].transAxes, ha='left')

axs[0,1].loglog(hf_freq,hf_amp_E,lw=1,alpha=0.7,c='purple',label='High-Frequency')
axs[0,1].loglog(lf_freq,lf_amp_E_acc,lw=1,alpha=0.7,c='gold',label='Low-Frequency')
axs[0,1].set_xlim(1/(hf_E[0].stats.npts*hf_E[0].stats.delta),50)
axs[0,1].grid(alpha=0.5)
axs[0,1].legend(loc='lower right')
axs[0,1].text(0.025, 0.875, 'Unfiltered', transform=axs[0,1].transAxes, ha='left')

axs[1,0].plot(hf_E_filt_1[0].times(),hf_amp_filt_norm_1,lw=1,alpha=0.7,c='purple',label='High-frequency')
axs[1,0].plot(lf_E_filt_1[0].times(),lf_amp_filt_norm_1,lw=1,alpha=0.7,c='gold',label='Low-frequency')
axs[1,0].set_xlim(25,300)
axs[1,0].set_ylabel(r'Normalized Amplitude')
axs[1,0].grid(alpha=0.5)
axs[1,0].legend(loc='lower right')
axs[1,0].text(0.025, 0.875, 'Filtered', transform=axs[1,0].transAxes, ha='left')

axs[1,1].loglog(hf_freq,hf_amp_E_filt_1,lw=1,alpha=0.7,c='purple',label='High-Frequency')
axs[1,1].loglog(lf_freq,lf_amp_E_filt_1,lw=1,alpha=0.7,c='gold',label='Low-Frequency')
axs[1,1].set_ylabel('Amplitude (m/s)')
axs[1,1].set_xlim(1/(hf_E[0].stats.npts*hf_E[0].stats.delta),50)
axs[1,1].axvline(x=0.998, lw=1, ls='--', c='k', label='fc')
axs[1,1].grid(alpha=0.5)
axs[1,1].legend(loc='lower right')
axs[1,1].text(0.025, 0.875, 'Filtered', transform=axs[1,1].transAxes, ha='left')

axs[2,0].plot(bb_1[0].times(),bb_amp_norm_1,lw=1,c='mediumseagreen',label='Broadband')
axs[2,0].set_xlabel('Time (s)')
axs[2,0].set_xlim(25,300)
axs[2,0].grid(alpha=0.5)
axs[2,0].legend(loc='lower right')

axs[2,1].loglog(bb_freq,bb_amp_1,lw=1,c='mediumseagreen',label='Broadband')
axs[2,1].set_xlabel('Frequency (Hz)')
axs[2,1].set_xlim(1/(hf_E[0].stats.npts*hf_E[0].stats.delta),50)
axs[2,1].grid(alpha=0.5)
axs[2,1].legend(loc='lower right')
plt.subplots_adjust(left=0.075,right=0.98,bottom=0.1,top=0.98,hspace=0.225)
plt.savefig('/Users/tnye/tsuquakes/manuscript/figures/unannotated/FigS3_matchedfilt_1fc.png',dpi=300)




#%%
################################## Figure 2fc #################################

# BB waveform and spectrum (filtered)
fig, axs = plt.subplots (3,2, figsize=(9,7))
axs[0,0].plot(hf_E[0].times(),hf_amp_norm,lw=1,alpha=0.7,c='purple',label='High-frequency')
axs[0,0].plot(lf_E_acc[0].times(),lf_amp_norm,lw=1,alpha=0.7,c='gold',label='Low-frequency')
axs[0,0].set_xlim(25,300)
axs[0,0].grid(alpha=0.5)
axs[0,0].legend(loc='lower right')
axs[0,0].text(0.025, 0.875, 'Unfiltered', transform=axs[0,0].transAxes, ha='left')

axs[0,1].loglog(hf_freq,hf_amp_E,lw=1,alpha=0.7,c='purple',label='High-Frequency')
axs[0,1].loglog(lf_freq,lf_amp_E_acc,lw=1,alpha=0.7,c='gold',label='Low-Frequency')
axs[0,1].set_xlim(1/(hf_E[0].stats.npts*hf_E[0].stats.delta),50)
axs[0,1].grid(alpha=0.5)
axs[0,1].legend(loc='lower right')
axs[0,1].text(0.025, 0.875, 'Unfiltered', transform=axs[0,1].transAxes, ha='left')

axs[1,0].plot(hf_E_filt_01[0].times(),hf_amp_filt_norm_01,lw=1,alpha=0.7,c='purple',label='High-frequency')
axs[1,0].plot(lf_E_filt_1[0].times(),lf_amp_filt_norm_1,lw=1,alpha=0.7,c='gold',label='Low-frequency')
axs[1,0].set_xlim(25,300)
axs[1,0].set_ylabel(r'Normalized Amplitude')
axs[1,0].grid(alpha=0.5)
axs[1,0].legend(loc='lower right')
axs[1,0].text(0.025, 0.875, 'Filtered', transform=axs[1,0].transAxes, ha='left')

axs[1,1].loglog(hf_freq,hf_amp_E_filt_01,lw=1,alpha=0.7,c='purple',label='High-Frequency')
axs[1,1].loglog(lf_freq,lf_amp_E_filt_1,lw=1,alpha=0.7,c='gold',label='Low-Frequency')
axs[1,1].set_ylabel('Amplitude (m/s)')
axs[1,1].set_xlim(1/(hf_E[0].stats.npts*hf_E[0].stats.delta),50)
axs[1,1].axvline(x=0.998, lw=1, ls='--', c='k', label=r'$f_{c,low}$')
axs[1,1].axvline(x=0.1, lw=1, ls=':', c='k', label=r'$f_{c,high}$')
axs[1,1].grid(alpha=0.5)
axs[1,1].legend(loc='lower right')
axs[1,1].text(0.025, 0.875, 'Filtered', transform=axs[1,1].transAxes, ha='left')

axs[2,0].plot(bb_01[0].times(),bb_amp_norm_01,lw=1,c='mediumseagreen',label='Broadband')
axs[2,0].set_xlabel('Time (s)')
axs[2,0].set_xlim(25,300)
axs[2,0].grid(alpha=0.5)
axs[2,0].legend(loc='lower right')

axs[2,1].loglog(bb_freq,bb_amp_01,lw=1,c='mediumseagreen',label='Broadband')
axs[2,1].set_xlabel('Frequency (Hz)')
axs[2,1].set_xlim(1/(hf_E[0].stats.npts*hf_E[0].stats.delta),50)
axs[2,1].grid(alpha=0.5)
axs[2,1].legend(loc='lower right')
plt.subplots_adjust(left=0.075,right=0.98,bottom=0.1,top=0.98,hspace=0.225)
plt.savefig(f'/Users/tnye/tsuquakes/manuscript/figures/unannotated/FigS4_matchedfilt_2fc.png',dpi=300)


