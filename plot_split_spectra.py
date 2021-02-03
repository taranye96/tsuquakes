#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:12:29 2021

@author: tnye
"""

###############################################################################
# Script that makes a figure comparing the full synthetic spectra to spectra 
# generated with FakeQuakes to the low and high frequency components. 
###############################################################################

# Imports
import numpy as np
import pandas as pd
from glob import glob
from obspy import read
from mudpy import forward
from numpy import r_, diff
from mtspec import mtspec
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


################################# One station #################################

# Get waveforms
syn_wf = read('/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/mentawai.000000/CGJI.bb.HNE.sac')
lf_wf = read('/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/mentawai.000000/CGJI.LYE.sac')
hf_wf = read('/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/mentawai.000000/CGJI.HNE.mpi.sac')

    
                ############## Synthetic spectra ##############

# Calculate spectra 
syn_amp_squared, syn_freq =  mtspec(syn_wf[0].data, delta=syn_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=syn_wf[0].stats.npts, quadratic=True)
syn_amp = np.sqrt(syn_amp_squared)


              ############## Low frequency spectra ##############

# Double differentiate disp waveform to get it acc
lf_vel_data = r_[0,diff(lf_wf[0].data)/lf_wf[0].stats.delta]
lf_acc_data = r_[0,diff(lf_vel_data)/lf_wf[0].stats.delta]

# Low pass filter low frequency data
lf_acc_data = forward.lowpass(lf_acc_data,0.998,lf_wf[0].stats.sampling_rate,4,zerophase=True)

# Calculate spectra
lf_amp_squared, lf_freq =  mtspec(lf_acc_data, delta=lf_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_wf[0].stats.npts, quadratic=True)
lf_amp = np.sqrt(lf_amp_squared)


             ############## High frequency spectra ##############

# High pass filter high frequency data
hf_wf[0].data = forward.highpass(hf_wf[0].data,0.998,hf_wf[0].stats.sampling_rate,4,zerophase=True)

# Calculate spectra
hf_amp_squared, hf_freq =  mtspec(hf_wf[0].data, delta=hf_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=hf_wf[0].stats.npts, quadratic=True)
hf_amp = np.sqrt(hf_amp_squared)


                   ############## Make figure ##############

# Plot spectra
plt.loglog(lf_freq,lf_amp,lw=.8,c='mediumpurple',ls='-',label='low frequency')
plt.loglog(hf_freq,hf_amp,lw=.8,c='darkturquoise',ls='-',label='high frequency')
plt.loglog(syn_freq,syn_amp,lw=.8,c='C1',ls='-',label='full synthetic')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s)')
plt.title('Mentawai, Station CGJI')

# Save figure
plt.savefig('/Users/tnye/tsuquakes/plots/split_spectra/mentawai.000000.CGJI_nofilter.png', dpi=300)
plt.close()



################################ All stations #################################

# synthetic run
project = 'standard'
run = 'mentawai.000000'
data_type = 'acc'

# Get waveforms
# syn_wfs_list = np.array(sorted(glob(f'/Users/tnye/FakeQuakes/parameters/standard/std/acc/{run}/' + '*HNE.mseed')))
syn_wfs_list = np.array(sorted(glob(f'/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/{run}/' + '*bb.HNE.sac')))
lf_wfs_list = np.array(sorted(glob(f'/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/{run}/' + '*LYE.sac')))
hf_wfs_list = np.array(sorted(glob(f'/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/{run}/' + '*HNE.mpi.sac')))

# Get list of station names
stn_list = []
for file in syn_wfs_list:
    stn = file.split('/')[-1].split('.')[0]
    stn_list.append(stn)

# Read in waveforms
syn_wfs = []
lf_wfs = []
hf_wfs = []

for i in range(len(stn_list)):
    syn_wfs.append(read(syn_wfs_list[i]))
    lf_wfs.append(read(lf_wfs_list[i]))
    hf_wfs.append(read(hf_wfs_list[i]))
    
# Get spectra amplitudes and frequencies
syn_freqs = []
syn_spec = []
lf_freqs = []
lf_spec = []
hf_freqs = []
hf_spec = []

for i in range(len(stn_list)):
    
                 ############## Synthetic spectra ##############
    
    # Calculate spectra 
    amp_squared, freq =  mtspec(syn_wfs[i][0].data, delta=syn_wfs[i][0].stats.delta, time_bandwidth=4, 
                                  number_of_tapers=5, nfft=syn_wfs[i][0].stats.npts, quadratic=True)
    amp = np.sqrt(amp_squared)
    syn_freqs.append(freq)
    syn_spec.append(amp)
    
    
               ############## Low frequency spectra ##############
    
    # Double differentiate disp waveform to get it acc
    lf_wfs[i][0].data = r_[0,diff(lf_wfs[i][0].data)/lf_wfs[i][0].stats.delta]
    acc_data = r_[0,diff(lf_wfs[i][0].data)/lf_wfs[i][0].stats.delta]
    
    # Low pass filter low frequency data
    lf_wfs[i][0].data = forward.lowpass(acc_data,0.998,lf_wfs[i][0].stats.sampling_rate,4,zerophase=True)
    
    # Calculate spectra
    amp_squared, freq =  mtspec(acc_data, delta=lf_wfs[i][0].stats.delta, time_bandwidth=4, 
                                  number_of_tapers=5, nfft=lf_wfs[i][0].stats.npts, quadratic=True)
    amp = np.sqrt(amp_squared)
    lf_freqs.append(freq)
    lf_spec.append(amp)
    
    
               ############## High frequency spectra ##############
    
    # High pass filter high frequency data
    hf_wfs[i][0].data = forward.highpass(hf_wfs[i][0].data,0.998,hf_wfs[i][0].stats.sampling_rate,4,zerophase=True)
    
    # Calculate spectra
    amp_squared, freq =  mtspec(hf_wfs[i][0].data, delta=hf_wfs[i][0].stats.delta, time_bandwidth=4, 
                                  number_of_tapers=5, nfft=hf_wfs[i][0].stats.npts, quadratic=True)
    amp = np.sqrt(amp_squared)
    hf_freqs.append(freq)
    hf_spec.append(amp)


# Get hypdists
hypdists = np.array(pd.read_csv('/Users/tnye/tsuquakes/data/stations/sm_hypdist.csv')['Hypdist(km)'])

# Sort hypdist and get indices
sort_id = np.argsort(np.argsort(hypdists))

# Function to sort a list based off of another
def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1) 
    z = [x for _, x in sorted(zipped_pairs)] 
    return z

# Sort freq and amps based off hypocentral distance
sort_syn_freqs = sort_list(syn_freqs, sort_id)
sort_syn_spec = sort_list(syn_spec, sort_id)
sort_lf_freqs = sort_list(lf_freqs, sort_id)
sort_lf_spec = sort_list(lf_spec, sort_id)
sort_hf_freqs = sort_list(hf_freqs, sort_id)
sort_hf_spec = sort_list(hf_spec, sort_id)
sort_stn_name = sort_list(stn_list, sort_id)
sort_hypdists = sort_list(hypdists, sort_id)


                  ############## Make figure ##############

# Set up figure
xlim = .002, 10
ylim = 7*10**-15, 6*10**-1
dim = 6,3
fig, axs = plt.subplots(dim[0],dim[1],figsize=(8,10))

# Initialize k. This is the station index. 
k = 0
# Loop rhough rows
for i in range(dim[0]):
    # Loop through columns
    for j in range(dim[1]):
        # Only make enough subplots for length of station list
        if k+1 <= len(stn_list):
           
            # Plot spectra
            axs[i][j].loglog(sort_lf_freqs[k],sort_lf_spec[k],lw=.8,c='mediumpurple',ls='-',label='low frequency')
            axs[i][j].loglog(sort_hf_freqs[k],sort_hf_spec[k],lw=.8,c='darkturquoise',ls='-',label='high frequency')
            axs[i][j].loglog(sort_syn_freqs[k],sort_syn_spec[k],lw=.8,c='C1',ls='-',label='full synthetic')
            # axs[i][j].loglog(sort_syn_freqs[k],sort_syn_spec[k],lw=.8,c='C1',ls='-',alpha=.25,label='full synthetic')
            
            # Plot subplot text
            axs[i][j].text(0.025,8.5E-1,'HNE',transform=axs[i][j].transAxes,size=7)
            axs[i][j].text(0.025,5E-2,f'Hypdist={int(sort_hypdists[k])}km',
                           transform=axs[i][j].transAxes,size=7)
            
            # Format subplot
            axs[i][j].grid(linestyle='--')
            axs[i][j].set_xlim(xlim)
            axs[i][j].set_ylim(ylim)
            axs[i][j].tick_params(axis='x', labelrotation=45, labelsize=8)
            axs[i][j].tick_params(axis='y', labelsize=8)
            axs[i][j].set_title(sort_stn_name[k],fontsize=10)
            
            # Remove tick labels on certain subplots to save room
            if i < dim[0]-2:
                axs[i][j].set_xticklabels([])  
            if i == dim[0]-2 and j == 0:
                axs[i][j].set_xticklabels([])    
            if j > 0:
                axs[i][j].set_yticklabels([])

            k += 1

# Format figure 
fig.text(0.5, 0.005, 'Frequency (Hz)', ha='center')
fig.text(0.005, 0.5, f'Amplitude m/s', va='center', rotation='vertical')
handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.685,0.05), framealpha=None)
fig.delaxes(axs[5][1])
fig.delaxes(axs[5][2])
fig.suptitle('Split Synthetic Spectra', fontsize=12, y=1)
fig.text(0.4, 0.115, (r"$\bf{" + 'Project:' + "}$" + '' + project))
fig.text(0.4, 0.09, (r'$\bf{' + 'Run:' + '}$' + '' + run))
fig.text(0.4, 0.065, (r'$\bf{' + 'DataType:' '}$' + '' + data_type))
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0.2, hspace=0.3)

# Save figure
plt.savefig(f'/Users/tnye/tsuquakes/plots/split_spectra/{run}.all.png', dpi=300)
plt.close()


