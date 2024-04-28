#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:51:52 2020

@author: vjs
"""

###############################################################################
# Script used to to help determine why the velocity spectra were taking so long
# to compute.  This is Valerie's code, and it ran much faster.  I compared this
# with my code in test_velocity_tan. 
###############################################################################

# Imports
import numpy as np
import mtspec as mtspec
import pandas as pd
import obspy as obs
import matplotlib.pyplot as plt
import time
from scipy.signal import butter,filtfilt,lfilter
from scipy.integrate import cumtrapz

# datadirectory = '/Users/vjs/tara/velocitytest/'
datadirectory = '/Users/tnye/vel_test/'


station = 'CGJI'
## either: 'CGJI', 'CNJI', or 'KASI'

num_poles = 2
fcorner = 1/15

obs_gain = 100 ## counts / cm/s2

# %%
## REad in files into stream:
obs_file = datadirectory + station +  '.HNE.mseed'
syn_file = datadirectory + station +  '.bb.HNE.sac'

acc_obs_stream = obs.read(obs_file)
acc_syn_stream = obs.read(syn_file)

## gain the observed data, then convert it from cm to m:
acc_obs_stream[0].data = acc_obs_stream[0].data / obs_gain / 100

## also start a figure:
wf_fig,wf_axes = plt.subplots(nrows=4,ncols=1)
spec_fig,spec_axes = wf_fig = plt.subplots(nrows=4,ncols=1)

# In first panel, plot waveforms:
wf_axes[0].plot(acc_obs_stream[0].times(),acc_obs_stream[0].data,color='blue')
wf_axes[0].plot(acc_syn_stream[0].times(),acc_syn_stream[0].data,color='orange')



## Get spectra...
obs_acc_start = time.time()
[obs_acc_spec,obs_acc_freq] = mtspec.mtspec(acc_obs_stream[0].data, acc_obs_stream[0].stats.delta, time_bandwidth=4,number_of_tapers=7)
obs_acc_end = time.time()
print('obs acc time %f' % (obs_acc_end - obs_acc_start))

syn_acc_start = time.time() 
[syn_acc_spec,syn_acc_freq] = mtspec.mtspec(acc_syn_stream[0].data, acc_syn_stream[0].stats.delta, time_bandwidth=4,number_of_tapers=7)
syn_acc_end = time.time()
print('syn acc time %f' % (syn_acc_end - syn_acc_start))

## plot spectra
spec_axes[0].loglog(obs_acc_freq,obs_acc_spec,color='blue')
spec_axes[0].loglog(syn_acc_freq,syn_acc_spec,color='orange')


# %%
## Highpass Filter waveforms. give it the corner frequency divided by nyquist,
##   since the filter goes from 0 to 1 where the nyquist is 1
b_obs, a_obs = butter(num_poles, np.array(fcorner)/(acc_obs_stream[0].stats.sampling_rate/2),'highpass')
acc_obs_filt = filtfilt(b_obs,a_obs,acc_obs_stream[0].data)

b_syn, a_syn = butter(num_poles, np.array(fcorner)/(acc_syn_stream[0].stats.sampling_rate/2),'highpass')
acc_syn_filt = filtfilt(b_syn,a_syn,acc_syn_stream[0].data)

## plot the filtered waveforms:
wf_axes[1].plot(acc_obs_stream[0].times(),acc_obs_filt,color='blue')
wf_axes[1].plot(acc_syn_stream[0].times(),acc_syn_filt,color='orange')

## Get spectra...
obs_acc_start = time.time()
[obs_acc_filt_spec,obs_acc_filt_freq] = mtspec.mtspec(acc_obs_filt, acc_obs_stream[0].stats.delta, time_bandwidth=4,number_of_tapers=7)
obs_acc_end = time.time()
print('obs acc filtered time %f' % (obs_acc_end - obs_acc_start))

syn_acc_start = time.time() 
[syn_acc_filt_spec,syn_acc_filt_freq] = mtspec.mtspec(acc_syn_filt, acc_syn_stream[0].stats.delta, time_bandwidth=4,number_of_tapers=7)
syn_acc_end = time.time()
print('syn acc filtered time %f' % (syn_acc_end - syn_acc_start))

## plot spectra
spec_axes[1].loglog(obs_acc_filt_freq,obs_acc_filt_spec,color='blue')
spec_axes[1].loglog(syn_acc_filt_freq,syn_acc_filt_spec,color='orange')

# %%
## INtegrate waveforms
vel_obs_amplitude = cumtrapz(acc_obs_filt,x=acc_obs_stream[0].times(),initial=0)
vel_syn_amplitude = cumtrapz(acc_syn_filt,x=acc_syn_stream[0].times(),initial=0)

#plot the velocity waveforms:
wf_axes[2].plot(acc_obs_stream[0].times(),vel_obs_amplitude,color='blue')
wf_axes[2].plot(acc_syn_stream[0].times(),vel_syn_amplitude,color='orange')

## Get spectra...
obs_vel_start = time.time()
[obs_vel_spec,obs_vel_freq] = mtspec.mtspec(vel_obs_amplitude, acc_obs_stream[0].stats.delta, time_bandwidth=4,number_of_tapers=7)
obs_vel_end = time.time()
print('obs vel time %f' % (obs_vel_end - obs_vel_start))

syn_vel_start = time.time() 
[syn_vel_spec,syn_vel_freq] = mtspec.mtspec(vel_syn_amplitude, acc_syn_stream[0].stats.delta, time_bandwidth=4,number_of_tapers=7)
syn_vel_end = time.time()
print('synvel time %f' % (syn_vel_end - syn_vel_start))

## plot spectra
spec_axes[2].loglog(obs_vel_freq,obs_vel_spec,color='blue')
spec_axes[2].loglog(syn_vel_freq,syn_vel_spec,color='orange')


# %%
## Filter again...
## Highpass Filter waveforms. give it the corner frequency divided by nyquist,
##   since the filter goes from 0 to 1 where the nyquist is 1
b_obs, a_obs = butter(num_poles, np.array(fcorner)/(acc_obs_stream[0].stats.sampling_rate/2),'highpass')
vel_obs_filt = filtfilt(b_obs,a_obs,vel_obs_amplitude)

b_syn, a_syn = butter(num_poles, np.array(fcorner)/(acc_syn_stream[0].stats.sampling_rate/2),'highpass')
vel_syn_filt = filtfilt(b_syn,a_syn,vel_syn_amplitude)

## plot the filtered waveforms:
wf_axes[3].plot(acc_obs_stream[0].times(),vel_obs_filt,color='blue')
wf_axes[3].plot(acc_syn_stream[0].times(),vel_syn_filt,color='orange')

## Get spectra...
obs_vel__start = time.time()
[obs_vel_filt_spec,obs_vel_filt_freq] = mtspec.mtspec(vel_obs_filt, acc_obs_stream[0].stats.delta, time_bandwidth=4,number_of_tapers=7)
obs_vel_end = time.time()
print('obs vel filtered time %f' % (obs_acc_end - obs_acc_start))

syn_vel_start = time.time() 
[syn_vel_filt_spec,syn_vel_filt_freq] = mtspec.mtspec(vel_syn_filt, acc_syn_stream[0].stats.delta, time_bandwidth=4,number_of_tapers=7)
syn_vel_end = time.time()
print('syn vel filtered time %f' % (syn_vel_end - syn_vel_start))

## plot spectra
spec_axes[3].loglog(obs_vel_filt_freq,obs_vel_filt_spec,color='blue')
spec_axes[3].loglog(syn_vel_filt_freq,syn_vel_filt_spec,color='orange')



