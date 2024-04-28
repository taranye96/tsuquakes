#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:51:25 2021

@author: tnye
"""

###############################################################################
# Script that makes a figure comparing Dara's full synthetic spectra to spectra 
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
import tsueqs_main_fns as tmf

# Get waveforms
syn_wf = read('/Users/tnye/tsuquakes/data/cascadia_pwave.000190/2133.bb.HNE.sac')
lf_wf = read('/Users/tnye/tsuquakes/data/cascadia_pwave.000190/2133.LYE.sac')
hf_wf = read('/Users/tnye/tsuquakes/data/cascadia_pwave.000190/2133.HNE.mpi.sac')

    
############################### Synthetic spectra #########################

# Calculate spectra 
syn_amp_squared, syn_freq =  mtspec(syn_wf[0].data, delta=syn_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=syn_wf[0].stats.npts, quadratic=True)
syn_amp = np.sqrt(syn_amp_squared)


############################# Low frequency spectra #######################

# Double differentiate disp waveform to get it acc
lf_wf[0].data = r_[0,diff(lf_wf[0].data)/lf_wf[0].stats.delta]
# lf_wf[0].data = r_[0,diff(lf_wf[0].data)/lf_wf[0].stats.delta]

# # Low pass filter low frequency data
# lf_wf[0].data = forward.lowpass(lf_wf[0].data,0.998,lf_wf[0].stats.sampling_rate,4,zerophase=True)

# Calculate spectra
lf_amp_squared, lf_freq =  mtspec(lf_wf[0].data, delta=lf_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_wf[0].stats.npts, quadratic=True)
lf_amp = np.sqrt(lf_amp_squared)


############################ High frequency spectra #######################

# # High pass filter high frequency data
# hf_wf[0].data = forward.highpass(hf_wf[0].data,0.998,hf_wf[0].stats.sampling_rate,4,zerophase=True)

# Calculate spectra
hf_amp_squared, hf_freq =  mtspec(hf_wf[0].data, delta=hf_wf[0].stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=hf_wf[0].stats.npts, quadratic=True)
hf_amp = np.sqrt(hf_amp_squared)


################################ Make figure ##################################

# Plot spectra
plt.loglog(lf_freq,lf_amp,lw=.8,c='mediumpurple',ls='-',label='low frequency: disp')
plt.loglog(hf_freq,hf_amp,lw=.8,c='darkturquoise',ls='-',label='high frequency: acc')
plt.loglog(syn_freq,syn_amp,lw=.8,c='C1',ls='-',label='full synthetic')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Cascadia, Station 2133: Pre-Filtering')

# Save figure
plt.savefig('/Users/tnye/tsuquakes/plots/matched_filter_analysis/cascadia_pwave.000190.2133_prefilt.png', dpi=300)
plt.close()


# Plot spectra
plt.loglog(lf_freq,lf_amp,lw=.8,c='mediumpurple',ls='-',label='low frequency: disp')
plt.loglog(hf_freq,hf_amp,lw=.8,c='darkturquoise',ls='-',label='high frequency: acc')
plt.loglog(syn_freq,syn_amp,lw=.8,c='C1',ls='-',label='full synthetic')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Cascadia, Station 2133: Pre-Filtering')

