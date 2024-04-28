#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:12:59 2020

@author: tnye
"""

###############################################################################
# Script that makes a figure comparing the observed spectra to spectra 
# generated with FakeQuakes using a kappa of 0.04s and station-specific kappa 
# for each sm station, as well as the low frequency component of the synthetic 
# spectra. 
###############################################################################

# Imports
import numpy as np
import pandas as pd
from glob import glob
from obspy import read
from mtspec import mtspec
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# synthetic run
run = 'mentawai.000000'

# Get waveforms
obs_wfs_list = np.array(sorted(glob('/Users/tnye/tsuquakes/data/Mentawai2010/accel_corr/' + '*HNE*.mseed')))
setK_wfs_list = np.array(sorted(glob(f'/Users/tnye/FakeQuakes/parameters/standard/std/acc/{run}/' + '*HNE.mseed')))
varK_wfs_list = np.array(sorted(glob(f'/Users/tnye/FakeQuakes/parameters/test/kappa_test/acc/{run}/' + '*HNE.mseed')))

# Station specific kappa values
kappa_df = pd.read_csv('/Users/tnye/tsuquakes/flatfiles/vs30_kappa.csv')
kappa = kappa_df.Kappa.values

# Get list of station names
stn_list = []
for file in obs_wfs_list:
    stn = file.split('/')[-1].split('.')[0]
    stn_list.append(stn)

# Read in waveforms
obs_wfs = []
setK_wfs = []
varK_wfs = []

for i in range(len(stn_list)):
    obs_wfs.append(read(obs_wfs_list[i]))
    setK_wfs.append(read(setK_wfs_list[i]))
    varK_wfs.append(read(varK_wfs_list[i]))
    
# Get spectra amplitudes and frequencies
obs_freqs = []
obs_spec = []
setK_freqs = []
setK_spec = []
varK_freqs = []
varK_spec = []

for i in range(len(stn_list)):
    
    # Observed spectra
    amp_squared, freq =  mtspec(obs_wfs[i][0].data, delta=obs_wfs[i][0].stats.delta, time_bandwidth=4, 
                                  number_of_tapers=5, nfft=obs_wfs[i][0].stats.npts, quadratic=True)
    amp = np.sqrt(amp_squared)
    obs_freqs.append(freq)
    obs_spec.append(amp)
    
    # Set Kappa spectra
    amp_squared, freq =  mtspec(setK_wfs[i][0].data, delta=setK_wfs[i][0].stats.delta, time_bandwidth=4, 
                                  number_of_tapers=5, nfft=setK_wfs[i][0].stats.npts, quadratic=True)
    amp = np.sqrt(amp_squared)
    setK_freqs.append(freq)
    setK_spec.append(amp)
    
    # Station-specific Kappa spectra
    amp_squared, freq =  mtspec(varK_wfs[i][0].data, delta=varK_wfs[i][0].stats.delta, time_bandwidth=4, 
                                  number_of_tapers=5, nfft=varK_wfs[i][0].stats.npts, quadratic=True)
    amp = np.sqrt(amp_squared)
    varK_freqs.append(freq)
    varK_spec.append(amp)

# Get hypdists
hypdists = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/stations/sm_hypdist.csv')['Hypdist(km)'])

# Sort hypdist and get indices
sort_id = np.argsort(np.argsort(kappa))

# Sort freq and amps based off hypdist
def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1) 
    z = [x for _, x in sorted(zipped_pairs)] 
    return z

sort_obs_freqs = sort_list(obs_freqs, sort_id)
sort_obs_spec = sort_list(obs_spec, sort_id)
sort_setK_freqs = sort_list(setK_freqs, sort_id)
sort_setK_spec = sort_list(setK_spec, sort_id)
sort_varK_freqs = sort_list(varK_freqs, sort_id)
sort_varK_spec = sort_list(varK_spec, sort_id)
sort_stn_name = sort_list(stn_list, sort_id)
sort_hypdists = sort_list(hypdists, sort_id)
sort_kappa = sort_list(kappa, sort_id)

# Set up figure
units = 'm/s'
# xlim = 0.17,1
xlim = -2.4,1
ylim = -21,0
dim = 6,3
fig, axs = plt.subplots(dim[0],dim[1],figsize=(8,10))
k = 0
# Loop rhough rows
for i in range(dim[0]):
    # Loop through columns
    for j in range(dim[1]):
        # Only make enough subplots for length of station list
        if k+1 <= len(stn_list):
            axs[i][j].plot(np.log10(sort_obs_freqs[k]),np.log(sort_obs_spec[k]),lw=.5,c='black',ls='-',label='observed')
            axs[i][j].plot(np.log10(sort_setK_freqs[k]),np.log(sort_setK_spec[k]),lw=.5,c='C1',ls='-',label='Kappa=0.04s')
            axs[i][j].plot(np.log10(sort_varK_freqs[k]),np.log(sort_varK_spec[k]),lw=.5,c='darkturquoise',ls='-',label='varied Kappa')
            axs[i][j].grid(linestyle='--')
            if sort_kappa[k] < 0.04:
                axs[i][j].set_facecolor('azure')
            else:
                axs[i][j].set_facecolor('ivory')
            axs[i][j].text(0.6,5E-2,f'Kappa={round(sort_kappa[k],3)}s',
                           transform=axs[i][j].transAxes,size=7)
            axs[i][j].text(0.025,8E-1,'HNE',transform=axs[i][j].transAxes,size=7)
            axs[i][j].text(0.025,5E-2,f'Hypdist={int(sort_hypdists[k])}km',
                           transform=axs[i][j].transAxes,size=7)
            axs[i][j].set_xlim(xlim)
            axs[i][j].set_ylim(ylim)
            axs[i][j].tick_params(axis='x', labelrotation=45, labelsize=8)
            axs[i][j].tick_params(axis='y', labelsize=8)
            axs[i][j].set_title(sort_stn_name[k],fontsize=10)
            if i < dim[0]-2:
                axs[i][j].set_xticklabels([])  
            if i == dim[0]-2 and j == 0:
                axs[i][j].set_xticklabels([])    
            if j > 0:
                axs[i][j].set_yticklabels([])  
            k += 1

# Make legend for background colors
legend_elements = [Patch(facecolor='azure',edgecolor='black',label='Kappa < 0.04s'),
                   Patch(facecolor='ivory',edgecolor='black',label='Kappa > 0.04s')]
fig.text(0.5, 0.005, 'log10 Frequency (Hz)', ha='center')
fig.text(0.005, 0.5, f'log Amplitude ({units})', va='center', rotation='vertical')
handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles=legend_elements, loc=(0.405,0.07), framealpha=None)
fig.legend(handles, labels, loc=(0.695,0.04), framealpha=None)
fig.delaxes(axs[5][1])
fig.delaxes(axs[5][2])
fig.suptitle('Kappa Comparison', fontsize=12, y=1)
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0.2, hspace=0.3)
plt.savefig('/Users/tnye/tsuquakes/plots/kappa/kappa_spectra_comp.png', dpi=300)


