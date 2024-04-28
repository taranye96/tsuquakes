#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:46:58 2021

@author: tnye
"""

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


################################## Waveforms ##################################

# Get waveforms
syn_wf = read('/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/mentawai.000000/CGJI.bb.HNE.sac')[0]
lf_wf = read('/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/mentawai.000000/CGJI.LYE.sac')[0]
hf_wf = read('/Users/tnye/FakeQuakes/parameters/standard/std/sm/output/waveforms/mentawai.000000/CGJI.HNE.mpi.sac')[0]

# Double differentiate disp waveform to get it acc
lf_vel_data = r_[0,diff(lf_wf.data)/lf_wf.stats.delta]
lf_acc_data = r_[0,diff(lf_vel_data)/lf_wf.stats.delta]

# Plot waveforms
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(lf_wf.times('matplotlib'),lf_acc_data,label=".LYE\" acc")
ax.plot(lf_wf.times('matplotlib'),lf_vel_data,label=".LYE' vel")
ax.plot(lf_wf.times('matplotlib'),lf_wf.data,label='.LYE')
plt.legend()
ax.xaxis_date()
fig.autofmt_xdate()
ax.set_ylabel('Amplitude')
ax.set_title('CGJI .LYE Waveforms')
plt.savefig('/Users/tnye/tsuquakes/plots/matched_filter_analysis/CGJI.LYE_waveforms.png', dpi=300)
plt.close()

################################### Spectra ###################################

# Low pass filter low frequency data
# lf_filtered_data = forward.lowpass(lf_wf,0.998,lf_wf.stats.sampling_rate,4,zerophase=True)

# Calculate spectra
lf_disp_amp2, lf_freq =  mtspec(lf_wf.data, delta=lf_wf.stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_wf.stats.npts, quadratic=True)
lf_disp_amp = np.sqrt(lf_disp_amp2)

lf_vel_amp2, lf_freq =  mtspec(lf_vel_data, delta=lf_wf.stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_wf.stats.npts, quadratic=True)
lf_vel_amp = np.sqrt(lf_vel_amp2)

lf_acc_amp2, lf_freq =  mtspec(lf_acc_data, delta=lf_wf.stats.delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=lf_wf.stats.npts, quadratic=True)
lf_acc_amp = np.sqrt(lf_acc_amp2)


# Make figure 
plt.loglog(lf_freq,lf_disp_amp,lw=.8,c='mediumpurple',ls='-',label=".LYE")
plt.loglog(lf_freq,lf_vel_amp,lw=.8,c='darkturquoise',ls='-',label=".LYE' vel")
plt.loglog(lf_freq,lf_acc_amp,lw=.8,c='C1',ls='-',label=".LYE\" acc")
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('CGJI .LYE Spectra')
plt.savefig('/Users/tnye/tsuquakes/plots/matched_filter_analysis/CGJI.LYE_spectra.png', dpi=300)
plt.close()
