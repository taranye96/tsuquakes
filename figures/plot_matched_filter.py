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

lf_wfs_2fc = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/final_suite/standard/output/waveforms/{rupture}/{stn}.LY*'))
hf_wfs_2fc = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/final_suite/standard/output/waveforms/{rupture}/{stn}.HN*mpi*'))
bb_wfs_2fc = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/final_suite/standard/output/waveforms/{rupture}/{stn}.bb.HN*'))

lf_wfs_1fc = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/final_suite/standard/output/waveforms/{rupture}/{stn}.LY*'))
hf_wfs_1fc = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/final_suite/standard/output/waveforms/{rupture}/{stn}.HN*.mpi*'))
bb_wfs_1fc = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/final_suite/standard/matched_filter/{rupture}/{stn}.bb.HN*'))


############################## Read in waveforms ##############################

lf_E_2fc = read(lf_wfs_2fc[0])
lf_N_2fc = read(lf_wfs_2fc[1])
hf_E_2fc = read(hf_wfs_2fc[0])
hf_N_2fc = read(hf_wfs_2fc[1])
bb_E_2fc = read(bb_wfs_2fc[0])
bb_N_2fc = read(bb_wfs_2fc[1])

lf_E_1fc = read(lf_wfs_1fc[0])
lf_N_1fc = read(lf_wfs_1fc[1])
hf_E_1fc = read(hf_wfs_1fc[0])
hf_N_1fc = read(hf_wfs_1fc[1])
bb_E_1fc = read(bb_wfs_1fc[0])
bb_N_1fc = read(bb_wfs_1fc[1])


########################## Calc rotd50 ground motion ##########################

rot50_lf_2fc = lf_E_2fc.copy()
rot50_hf_2fc = hf_E_2fc.copy()
rot50_bb_2fc = hf_E_2fc.copy()

rot50_lf_2fc[0].data = compute_rotd50(lf_E_2fc[0].data,lf_N_2fc[0].data)
rot50_hf_2fc[0].data = compute_rotd50(hf_E_2fc[0].data,hf_N_2fc[0].data)
rot50_bb_2fc[0].data = compute_rotd50(bb_E_2fc[0].data,bb_N_2fc[0].data)

rot50_lf_1fc = lf_E_1fc.copy()
rot50_hf_1fc = hf_E_1fc.copy()
rot50_bb_1fc = hf_E_1fc.copy()

rot50_lf_1fc[0].data = compute_rotd50(lf_E_1fc[0].data,lf_N_1fc[0].data)
rot50_hf_1fc[0].data = compute_rotd50(hf_E_1fc[0].data,hf_N_1fc[0].data)
rot50_bb_1fc[0].data = compute_rotd50(bb_E_1fc[0].data,bb_N_1fc[0].data)


########################## Differentiate disp to acc ##########################

lf_E_acc_2fc = lf_E_2fc.copy()
lf_N_acc_2fc = lf_N_2fc.copy()
rot50_lf_acc_2fc = rot50_lf_2fc.copy()

lf_E_acc_2fc[0].data=r_[0,diff(lf_E_acc_2fc[0].data)/lf_E_acc_2fc[0].stats.delta]
lf_E_acc_2fc[0].data=r_[0,diff(lf_E_acc_2fc[0].data)/lf_E_acc_2fc[0].stats.delta]

lf_N_acc_2fc[0].data=r_[0,diff(lf_N_acc_2fc[0].data)/lf_N_acc_2fc[0].stats.delta]
lf_N_acc_2fc[0].data=r_[0,diff(lf_N_acc_2fc[0].data)/lf_N_acc_2fc[0].stats.delta]

rot50_lf_acc_2fc[0].data=r_[0,diff(rot50_lf_acc_2fc[0].data)/rot50_lf_acc_2fc[0].stats.delta]
rot50_lf_acc_2fc[0].data=r_[0,diff(rot50_lf_acc_2fc[0].data)/rot50_lf_acc_2fc[0].stats.delta]

lf_E_acc_1fc = lf_E_1fc.copy()
lf_N_acc_1fc = lf_E_1fc.copy()
rot50_lf_acc_1fc = rot50_lf_1fc.copy()

lf_E_acc_1fc[0].data=r_[0,diff(lf_E_acc_1fc[0].data)/lf_E_acc_1fc[0].stats.delta]
lf_E_acc_1fc[0].data=r_[0,diff(lf_E_acc_1fc[0].data)/lf_E_acc_1fc[0].stats.delta]

lf_N_acc_1fc[0].data=r_[0,diff(lf_N_acc_1fc[0].data)/lf_N_acc_1fc[0].stats.delta]
lf_N_acc_1fc[0].data=r_[0,diff(lf_N_acc_1fc[0].data)/lf_N_acc_1fc[0].stats.delta]

rot50_lf_acc_1fc[0].data=r_[0,diff(rot50_lf_acc_1fc[0].data)/rot50_lf_acc_1fc[0].stats.delta]
rot50_lf_acc_1fc[0].data=r_[0,diff(rot50_lf_acc_1fc[0].data)/rot50_lf_acc_1fc[0].stats.delta]


################################# Filter data #################################

fsample_low=1./lf_E_2fc[0].stats.delta
fsample_hf=1./hf_E_2fc[0].stats.delta

lf_E_filt_2fc = lf_E_2fc.copy()
lf_N_filt_2fc = lf_N_2fc.copy()
rot50_lf_filt_2fc = rot50_lf_acc_2fc.copy()

hf_E_filt_2fc = hf_E_2fc.copy()
hf_N_filt_2fc = hf_N_2fc.copy()
rot50_hf_filt_2fc = rot50_hf_2fc.copy()

lf_E_filt_2fc[0].data=filt.lowpass(lf_E_filt_2fc[0].data,0.998,fsample_low,4,zerophase=True)
lf_N_filt_2fc[0].data=filt.lowpass(lf_N_filt_2fc[0].data,0.998,fsample_low,4,zerophase=True)
rot50_lf_filt_2fc[0].data=filt.lowpass(rot50_lf_filt_2fc[0].data,0.998,fsample_low,4,zerophase=True)

hf_E_filt_2fc[0].data=filt.highpass(hf_E_2fc[0].data,0.1,fsample_hf,4,zerophase=True)
hf_N_filt_2fc[0].data=filt.highpass(hf_N_2fc[0].data,0.1,fsample_hf,4,zerophase=True)
rot50_hf_filt_2fc[0].data=filt.highpass(rot50_hf_filt_2fc[0].data,0.1,fsample_hf,4,zerophase=True)


lf_E_filt_1fc = lf_E_1fc.copy()
lf_N_filt_1fc = lf_N_1fc.copy()
rot50_lf_filt_1fc = rot50_lf_acc_1fc.copy()

hf_E_filt_1fc = hf_E_1fc.copy()
hf_N_filt_1fc = hf_N_1fc.copy()
rot50_hf_filt_1fc = rot50_hf_1fc.copy()

lf_E_filt_1fc[0].data=filt.lowpass(lf_E_filt_1fc[0].data,0.998,fsample_low,4,zerophase=True)
lf_N_filt_1fc[0].data=filt.lowpass(lf_N_filt_1fc[0].data,0.998,fsample_low,4,zerophase=True)
rot50_lf_filt_1fc[0].data=filt.lowpass(rot50_lf_filt_1fc[0].data,0.998,fsample_low,4,zerophase=True)

hf_E_filt_1fc[0].data=filt.highpass(hf_E_1fc[0].data,0.998,fsample_hf,4,zerophase=True)
hf_N_filt_1fc[0].data=filt.highpass(hf_N_1fc[0].data,0.998,fsample_hf,4,zerophase=True)
rot50_hf_filt_1fc[0].data=filt.highpass(rot50_hf_filt_1fc[0].data,0.998,fsample_hf,4,zerophase=True)


############################## Calculate spectra ##############################

lf_freq, lf_amp_E_2fc = filt.calc_spec(lf_E_2fc[0].data, lf_E_2fc[0].stats.delta, lf_E_2fc[0].stats.npts)
lf_freq, lf_amp_N_2fc = filt.calc_spec(lf_N_2fc[0].data, lf_N_2fc[0].stats.delta, lf_N_2fc[0].stats.npts)
lf_freq, lf_amp_rot50_2fc = filt.calc_spec(rot50_lf_2fc[0].data, rot50_lf_2fc[0].stats.delta, rot50_lf_2fc[0].stats.npts)

lf_freq, lf_amp_E_acc_2fc = filt.calc_spec(lf_E_acc_2fc[0].data, lf_E_acc_2fc[0].stats.delta, lf_E_acc_2fc[0].stats.npts)
lf_freq, lf_amp_N_acc_2fc = filt.calc_spec(lf_N_acc_2fc[0].data, lf_N_acc_2fc[0].stats.delta, lf_N_acc_2fc[0].stats.npts)
lf_freq, lf_amp_rot50_acc_2fc = filt.calc_spec(rot50_lf_acc_2fc[0].data, rot50_lf_acc_2fc[0].stats.delta, rot50_lf_acc_2fc[0].stats.npts)

hf_freq, hf_amp_E_2fc = filt.calc_spec(hf_E_2fc[0].data, hf_E_2fc[0].stats.delta, hf_E_2fc[0].stats.npts)
hf_freq, hf_amp_N_2fc = filt.calc_spec(hf_N_2fc[0].data, hf_N_2fc[0].stats.delta, hf_N_2fc[0].stats.npts)
hf_freq, hf_amp_rot50_2fc = filt.calc_spec(rot50_hf_2fc[0].data, rot50_hf_2fc[0].stats.delta, rot50_hf_2fc[0].stats.npts)

lf_freq, lf_amp_E_filt_2fc = filt.calc_spec(lf_E_filt_2fc[0].data, lf_E_filt_2fc[0].stats.delta, lf_E_filt_2fc[0].stats.npts)
lf_freq, lf_amp_N_filt_2fc = filt.calc_spec(lf_N_filt_2fc[0].data, lf_N_filt_2fc[0].stats.delta, lf_N_filt_2fc[0].stats.npts)
lf_freq, lf_amp_rot50_filt_2fc = filt.calc_spec(rot50_lf_filt_2fc[0].data, rot50_lf_filt_2fc[0].stats.delta, rot50_lf_filt_2fc[0].stats.npts)

hf_freq, hf_amp_E_filt_2fc = filt.calc_spec(hf_E_filt_2fc[0].data, hf_E_filt_2fc[0].stats.delta, hf_E_filt_2fc[0].stats.npts)
hf_freq, hf_amp_N_filt_2fc = filt.calc_spec(hf_N_filt_2fc[0].data, hf_N_filt_2fc[0].stats.delta, hf_N_filt_2fc[0].stats.npts)
hf_freq, hf_amp_rot50_filt_2fc = filt.calc_spec(rot50_hf_filt_2fc[0].data, rot50_hf_filt_2fc[0].stats.delta, rot50_hf_filt_2fc[0].stats.npts)

bb_freq, bb_amp_E_2fc = filt.calc_spec(bb_E_2fc[0].data, bb_E_2fc[0].stats.delta, bb_E_2fc[0].stats.npts)
bb_freq, bb_amp_N_2fc = filt.calc_spec(bb_N_2fc[0].data, bb_N_2fc[0].stats.delta, bb_N_2fc[0].stats.npts)
bb_freq, bb_amp_rot50_2fc = filt.calc_spec(rot50_bb_2fc[0].data, rot50_bb_2fc[0].stats.delta, rot50_bb_2fc[0].stats.npts)

lf_amp_norm_2fc = rot50_lf_acc_2fc[0].data/np.max(rot50_lf_acc_2fc[0].data)
lf_amp_filt_norm_2fc = rot50_lf_filt_2fc[0].data/np.max(rot50_lf_filt_2fc[0].data)
hf_amp_norm_2fc = rot50_hf_2fc[0].data/np.max(rot50_hf_2fc[0].data)
hf_amp_filt_norm_2fc = rot50_hf_filt_2fc[0].data/np.max(rot50_hf_filt_2fc[0].data)
bb_amp_filt_norm_2fc = rot50_bb_2fc[0].data/np.max(rot50_bb_2fc[0].data)


lf_freq, lf_amp_E_2fc = filt.calc_spec(lf_E_1fc[0].data, lf_E_1fc[0].stats.delta, lf_E_1fc[0].stats.npts)
lf_freq, lf_amp_N_1fc = filt.calc_spec(lf_N_1fc[0].data, lf_N_1fc[0].stats.delta, lf_N_1fc[0].stats.npts)
lf_freq, lf_amp_rot50_1fc = filt.calc_spec(rot50_lf_1fc[0].data, rot50_lf_1fc[0].stats.delta, rot50_lf_1fc[0].stats.npts)

lf_freq, lf_amp_E_acc_1fc = filt.calc_spec(lf_E_acc_1fc[0].data, lf_E_acc_1fc[0].stats.delta, lf_E_acc_1fc[0].stats.npts)
lf_freq, lf_amp_N_acc_1fc = filt.calc_spec(lf_N_acc_1fc[0].data, lf_N_acc_1fc[0].stats.delta, lf_N_acc_1fc[0].stats.npts)
lf_freq, lf_amp_rot50_acc_1fc = filt.calc_spec(rot50_lf_acc_1fc[0].data, rot50_lf_acc_1fc[0].stats.delta, rot50_lf_acc_1fc[0].stats.npts)

hf_freq, hf_amp_E_1fc = filt.calc_spec(hf_E_1fc[0].data, hf_E_1fc[0].stats.delta, hf_E_1fc[0].stats.npts)
hf_freq, hf_amp_N_1fc = filt.calc_spec(hf_N_1fc[0].data, hf_N_1fc[0].stats.delta, hf_N_1fc[0].stats.npts)
hf_freq, hf_amp_rot50_1fc = filt.calc_spec(rot50_hf_1fc[0].data, rot50_hf_1fc[0].stats.delta, rot50_hf_1fc[0].stats.npts)

lf_freq, lf_amp_E_filt_1fc = filt.calc_spec(lf_E_filt_1fc[0].data, lf_E_filt_1fc[0].stats.delta, lf_E_filt_1fc[0].stats.npts)
lf_freq, lf_amp_N_filt_1fc = filt.calc_spec(lf_N_filt_1fc[0].data, lf_N_filt_1fc[0].stats.delta, lf_N_filt_1fc[0].stats.npts)
lf_freq, lf_amp_rot50_filt_1fc = filt.calc_spec(rot50_lf_filt_1fc[0].data, rot50_lf_filt_1fc[0].stats.delta, rot50_lf_filt_1fc[0].stats.npts)

hf_freq, hf_amp_E_filt_1fc = filt.calc_spec(hf_E_filt_1fc[0].data, hf_E_filt_1fc[0].stats.delta, hf_E_filt_1fc[0].stats.npts)
hf_freq, hf_amp_N_filt_1fc = filt.calc_spec(hf_N_filt_1fc[0].data, hf_N_filt_1fc[0].stats.delta, hf_N_filt_1fc[0].stats.npts)
hf_freq, hf_amp_rot50_filt_1fc = filt.calc_spec(rot50_hf_filt_1fc[0].data, rot50_hf_filt_1fc[0].stats.delta, rot50_hf_filt_1fc[0].stats.npts)

bb_freq, bb_amp_E_1fc = filt.calc_spec(bb_E_1fc[0].data, bb_E_1fc[0].stats.delta, bb_E_1fc[0].stats.npts)
bb_freq, bb_amp_N_1fc = filt.calc_spec(bb_N_1fc[0].data, bb_N_1fc[0].stats.delta, bb_N_1fc[0].stats.npts)
bb_freq, bb_amp_rot50_1fc = filt.calc_spec(rot50_bb_1fc[0].data, rot50_bb_1fc[0].stats.delta, rot50_bb_1fc[0].stats.npts)

lf_amp_norm_1fc = rot50_lf_acc_1fc[0].data/np.max(rot50_lf_acc_1fc[0].data)
lf_amp_filt_norm_1fc = rot50_lf_filt_1fc[0].data/np.max(rot50_lf_filt_1fc[0].data)
hf_amp_norm_1fc = rot50_hf_1fc[0].data/np.max(rot50_hf_1fc[0].data)
hf_amp_filt_norm_1fc = rot50_hf_filt_1fc[0].data/np.max(rot50_hf_filt_1fc[0].data)
bb_amp_filt_norm_1fc = rot50_bb_1fc[0].data/np.max(rot50_bb_1fc[0].data)

#%%
################################## Figure 2fc #################################

# BB waveform and spectrum (filtered)
fig, axs = plt.subplots (3,2, figsize=(9,7))
# fig, axs = plt.subplots(3,2)
axs[0,0].plot(rot50_hf_2fc[0].times(),hf_amp_norm_2fc,lw=1,alpha=0.7,c='purple',label='High-frequency')
axs[0,0].plot(rot50_lf_acc_2fc[0].times(),lf_amp_norm_2fc,lw=1,alpha=0.7,c='gold',label='Low-frequency')
axs[0,0].set_xlim(25,300)
axs[0,0].grid(alpha=0.5)
axs[0,0].legend(loc='lower right')
axs[0,0].text(0.025, 0.875, 'Unfiltered', transform=axs[0,0].transAxes, ha='left')

axs[0,1].loglog(hf_freq,hf_amp_rot50_2fc,lw=1,alpha=0.7,c='purple',label='High-Frequency')
axs[0,1].loglog(lf_freq,lf_amp_rot50_acc_2fc,lw=1,alpha=0.7,c='gold',label='Low-Frequency')
axs[0,1].set_xlim(1/(hf_E_2fc[0].stats.npts*hf_E_2fc[0].stats.delta),50)
# axs[0,1].axvline(x=0.998, lw=1, ls='--', c='k', label='fc')
axs[0,1].grid(alpha=0.5)
axs[0,1].legend(loc='lower right')
axs[0,1].text(0.025, 0.875, 'Unfiltered', transform=axs[0,1].transAxes, ha='left')

axs[1,0].plot(rot50_hf_filt_2fc[0].times(),hf_amp_filt_norm_2fc,lw=1,alpha=0.7,c='purple',label='High-frequency')
axs[1,0].plot(rot50_lf_filt_2fc[0].times(),lf_amp_filt_norm_2fc,lw=1,alpha=0.7,c='gold',label='Low-frequency')
axs[1,0].set_xlim(25,300)
axs[1,0].set_ylabel(r'Normalized Amplitude')
axs[1,0].grid(alpha=0.5)
axs[1,0].legend(loc='lower right')
axs[1,0].text(0.025, 0.875, 'Filtered', transform=axs[1,0].transAxes, ha='left')

axs[1,1].loglog(hf_freq,hf_amp_rot50_filt_2fc,lw=1,alpha=0.7,c='purple',label='High-Frequency')
axs[1,1].loglog(lf_freq,lf_amp_rot50_filt_2fc,lw=1,alpha=0.7,c='gold',label='Low-Frequency')
axs[1,1].set_ylabel('Amplitude (m/s)')
axs[1,1].set_xlim(1/(hf_E_2fc[0].stats.npts*hf_E_2fc[0].stats.delta),50)
axs[1,1].axvline(x=0.998, lw=1, ls='--', c='k', label='lowpass fc')
axs[1,1].axvline(x=0.1, lw=1, ls=':', c='k', label='highpass fc')
axs[1,1].grid(alpha=0.5)
axs[1,1].legend(loc='lower right')
axs[1,1].text(0.025, 0.875, 'Filtered', transform=axs[1,1].transAxes, ha='left')

axs[2,0].plot(rot50_bb_2fc[0].times(),bb_amp_filt_norm_2fc,lw=1,alpha=0.7,c='mediumseagreen',label='Broadband')
axs[2,0].set_xlabel('Time (s)')
axs[2,0].set_xlim(25,300)
axs[2,0].grid(alpha=0.5)
axs[2,0].legend(loc='lower right')

axs[2,1].loglog(bb_freq,bb_amp_rot50_2fc,lw=1,alpha=0.7,c='mediumseagreen',label='Broadband')
axs[2,1].set_xlabel('Frequency (Hz)')
axs[2,1].set_xlim(1/(hf_E_2fc[0].stats.npts*hf_E_2fc[0].stats.delta),50)
# axs[2,1].axvline(x=0.998, lw=1, ls='--', c='k', label='fc')
axs[2,1].grid(alpha=0.5)
axs[2,1].legend(loc='lower right')
plt.subplots_adjust(left=0.075,right=0.98,bottom=0.1,top=0.98,hspace=0.225)
plt.savefig(f'/Users/tnye/tsuquakes/manuscript/figures/S_matchedfilt_2fc_{stn}.png',dpi=300)


#%%
################################## Figure 1fc #################################

# BB waveform and spectrum (filtered)
fig, axs = plt.subplots (3,2, figsize=(9,7))
# fig, axs = plt.subplots(3,2)
axs[0,0].plot(rot50_hf_2fc[0].times(),hf_amp_norm_1fc,lw=1,alpha=0.7,c='purple',label='High-frequency')
axs[0,0].plot(rot50_lf_acc_1fc[0].times(),lf_amp_norm_1fc,lw=1,alpha=0.7,c='gold',label='Low-frequency')
axs[0,0].set_xlim(25,300)
axs[0,0].grid(alpha=0.5)
axs[0,0].legend(loc='lower right')
axs[0,0].text(0.025, 0.875, 'Unfiltered', transform=axs[0,0].transAxes, ha='left')
axs[0,1].loglog(hf_freq,hf_amp_rot50_1fc,lw=1,alpha=0.7,c='purple',label='High-Frequency')
axs[0,1].loglog(lf_freq,lf_amp_rot50_acc_1fc,lw=1,alpha=0.7,c='gold',label='Low-Frequency')
axs[0,1].set_xlim(1/(hf_E_1fc[0].stats.npts*hf_E_1fc[0].stats.delta),50)
# axs[0,1].axvline(x=0.998, lw=1, ls='--', c='k', label='fc')
axs[0,1].grid(alpha=0.5)
axs[0,1].legend(loc='lower right')
axs[0,1].text(0.025, 0.875, 'Unfiltered', transform=axs[0,1].transAxes, ha='left')

axs[1,0].plot(rot50_hf_filt_1fc[0].times(),hf_amp_filt_norm_1fc,lw=1,alpha=0.7,c='purple',label='High-frequency')
axs[1,0].plot(rot50_lf_filt_1fc[0].times(),lf_amp_filt_norm_1fc,lw=1,alpha=0.7,c='gold',label='Low-frequency')
axs[1,0].set_xlim(25,300)
axs[1,0].set_ylabel(r'Normalized Amplitude')
axs[1,0].grid(alpha=0.5)
axs[1,0].legend(loc='lower right')
axs[1,0].text(0.025, 0.875, 'Filtered', transform=axs[1,0].transAxes, ha='left')

axs[1,1].loglog(hf_freq,hf_amp_rot50_filt_1fc,lw=1,alpha=0.7,c='purple',label='High-Frequency')
axs[1,1].loglog(lf_freq,lf_amp_rot50_filt_1fc,lw=1,alpha=0.7,c='gold',label='Low-Frequency')
axs[1,1].set_ylabel('Amplitude (m/s)')
axs[1,1].set_xlim(1/(hf_E_1fc[0].stats.npts*hf_E_1fc[0].stats.delta),50)
axs[1,1].axvline(x=0.998, lw=1, ls='--', c='k', label='common fc')
axs[1,1].grid(alpha=0.5)
axs[1,1].legend(loc='lower right')
axs[1,1].text(0.025, 0.875, 'Filtered', transform=axs[1,1].transAxes, ha='left')

axs[2,0].plot(rot50_bb_1fc[0].times(),bb_amp_filt_norm_1fc,lw=1,alpha=0.7,c='mediumseagreen',label='Broadband')
axs[2,0].set_xlabel('Time (s)')
axs[2,0].set_xlim(25,300)
axs[2,0].grid(alpha=0.5)
axs[2,0].legend(loc='lower right')

axs[2,1].loglog(bb_freq,bb_amp_rot50_1fc,lw=1,alpha=0.7,c='mediumseagreen',label='Broadband')
axs[2,1].set_xlabel('Frequency (Hz)')
axs[2,1].set_xlim(1/(hf_E_1fc[0].stats.npts*hf_E_1fc[0].stats.delta),50)
# axs[2,1].axvline(x=0.998, lw=1, ls='--', c='k', label='fc')
axs[2,1].grid(alpha=0.5)
axs[2,1].legend(loc='lower right')
plt.subplots_adjust(left=0.075,right=0.98,bottom=0.1,top=0.98,hspace=0.225)
plt.savefig(f'/Users/tnye/tsuquakes/manuscript/figures/S_matchedfilt_1fc_{stn}.png',dpi=300)

