#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:49:52 2020

@author: tnye
"""

import numpy as np
import pandas as pd
from obspy.core import UTCDateTime
import datetime
from mtspec import mtspec
from scipy.stats import binned_statistic 
import matplotlib.pyplot as plt
import seaborn as sns


def calc_tPGD(pgd, trace, IMarray, origintime, hypdist,):
    """
    Calculates time to peak intensity measure (IM) from origin time and from
    estimated p-arrival.
    
    Inputs:
        pgd(float): Peak ground displacement (m).
        trace: Trace object with times for this station. 
        IMarray: Array pgd was calculated on.  Could be the same as stream
                        if peak IM was calculated on one component. 
        origintime(datetime): Origin time of event.
        hypdist(float): Hypocentral distance (km). 
        
    """
    
    # Calculate time from origin 
    pgd_index = np.where(IMarray==pgd)
    tPGD_orig = trace.times(reftime=UTCDateTime(origintime))[pgd_index]
    
    # Calculate time from p-arrival
    p_time = hypdist/6.5
    dp = datetime.timedelta(seconds=p_time)
    p_arrival = origintime+dp
    tPGD_parriv = trace.times(reftime=UTCDateTime(p_arrival))[pgd_index]
    
    return(tPGD_orig, tPGD_parriv)


def calc_tPGA(E_trace, N_trace, pga_E, pga_N, origintime, hypdist,):
    """
    Calculates time to peak intensity measure (IM) from origin time and from
    estimated p-arrival.
    
    Inputs:
        pgd(float): Peak ground displacement (m).
        trace: Trace object with times for this station. 
        origintime(datetime): Origin time of event.
        hypdist(float): Hypocentral distance (km). 
        
    """
    
    # Get pga index for both horizontal components
    pgaE_index = np.where(np.abs(E_trace)==pga_E)
    pgaN_index = np.where(np.abs(N_trace)==pga_N)
    # Calculuate geometric average of indexes 
    pga_index = np.sqrt(pgaE_index[0][0] * pgaN_index[0][0])
    
    # Calculate time from origin 
    tPGA_orig = E_trace.times(reftime=UTCDateTime(origintime))[pga_index]
    
    # Calculate time from p-arrival
    p_time = hypdist/6.5
    dp = datetime.timedelta(seconds=p_time)
    p_arrival = origintime+dp
    tPGA_parriv = E_trace.times(reftime=UTCDateTime(p_arrival))[pga_index]
    
    return(tPGA_orig, tPGA_parriv)


def calc_spectra(stream, data_type):
    """
    Calculates average spectra values in 25 bins for displacement, acceleration,
    and velocity waveforms.

    Inputs:
        stream: Obspy stream. 
        sm (str): Data type used for determining bin edges.
                    Options:
                        disp
                        acc
                        vel

    Return:
        bin_means(list): Binned spectra for given station
        freq(list): FFT frequencies
        amp(list): FFT amplitudes
    """

    # Read in file 
    tr = stream[0]
    data = tr.data
    delta = tr.stats.delta
    samprate = tr.stats.sampling_rate

    nyquist = 0.5 * samprate

    # Calc spectra amplitudes and frequencies
    amp_squared, freq =  mtspec(data, delta=delta, time_bandwidth=4, 
                             number_of_tapers=7, quadratic=True)
    # amp_squared, freq =  mtspec(data, delta=delta, time_bandwidth=4, 
    #                          number_of_tapers=7, nfft=8192, quadratic=True)
    # Take square root to get amplitude 
    amp = np.sqrt(amp_squared)

    # Remove zero frequencies so that I can take log, and remove data after fc
    indexes = []

    for i, val in enumerate(freq):
        if val == 0:
            indexes.append(i)

        elif val > nyquist: 
            indexes.append(i)

    freq = np.delete(freq,indexes)
    amp = np.delete(amp,indexes) 

    # Create bins for sm data: sampling rate not the same for each station so I 
    # created bin edges off of the max sampling rate: 100Hz
    
    if data_type == 'accel':
        data_type = 'sm'
    
    if data_type == 'sm':
        bins = [-2.68124124, -2.50603279, -2.33082434, -2.15561589,
                      -1.98040744, -1.80519899, -1.62999054, -1.45478209,
                      -1.27957364, -1.10436519, -0.92915674, -0.75394829,
                      -0.57873984, -0.40353139, -0.22832294, -0.05311449, 
                      0.12209396, 0.29730241, 0.47251086, 0.64771931, 0.82292776, 
                      0.99813621, 1.17334466, 1.3485531 , 1.52376155, 1.69897]
    
    # GNSS stations all had same sampling rate, so just using 25 is fine 
    elif data_type == 'disp':
        bins = 25
    
    bin_means, bin_edges, binnumber = binned_statistic(np.log10(freq),
                                                       np.log10(amp),
                                                       statistic='mean',
                                                       bins=bins)
    for i in range(len(bin_means)):
        bin_means[i] = 10**bin_means[i]
        
    return(bin_means, freq, amp)


def plot_spectra(stream, freqs, amps, data_type, synthetic=True, parameter='none', project='none', run='none'):
    """
    Plots Fourier spectra that have already been calculated. 
    
        Inputs:
            stream: Obspy stream for one component (just to get station name) 
            freqs(list): Array of list of frequencies obtained when computing Fourier
                         Spectra for E, N, and Z components
            amps(list): Array of list of amplitudes obtained when computing Fourier
                        Spectra for E, N, and Z components
            data_type(str): Type of streams data
                            Options:
                                Disp
                                Acc
                                Vel
            synthetic(True/False): Plotting synthetics
            parameter(str): Name of parameter folder.
            project(str): Name of simulation project.  This will be the main
                          directory where the different runs will be store. 
            run(str): Synthetics run number.  This will be the directory where the
                      plots are stored.
    """
    
    # Read in file 
    fig, axs = plt.subplots(3)
    
    tr = stream[0]
    station = tr.stats.station
    
    # Loop through frequencies and amplitudes 
    for i in range(len(freqs)):
        
        # Units
        if data_type == 'disp':
            title = 'Disp'
            units = 'm*s'
            code = 'LX' 
            ylim = 10**-4, 6*10**-1
            xlim = 2*10**-3, 5*10**-1
        elif data_type == 'acc':
            title = 'Acc'
            units = 'm/s'
            code = 'HN'
            ylim = 6*10**-15, 6*10**-1
            xlim = .002, 10
        elif data_type == 'vel':
            title = 'Vel'
            units = 'm'
            code = 'HN'
            ylim = 6*10**-15, 8*10**-2
            xlim = .002, 10
            
        # Define label 
        if i == 0:
            component = 'E'
        elif i == 1:
            component = 'N'
        elif i == 2:
            component = 'Z'
        label = code + component 
        
        # Plot spectra
        axs[i].loglog(freqs[i],amps[i], lw=.8, label=label)
        axs[i].grid(linestyle='--')
        axs[i].set_ylim(ylim)
        axs[i].set_xlim(xlim)
        axs[i].legend()

    # Format whole figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    fig.suptitle(f'{station} {title} Fourier Spectra', fontsize=14, y=1.08)
    fig.text(-.03, 0.5, f'Amplitude {units}', va='center', rotation='vertical')
    plt.xlabel('Frequency (Hz)')
    
    # if synthetic:
    #     plt.savefig(f'/Users/tnye/tsuquakes/plots/fourier_spec/synthetic/{parameter}/{project}/{run}/{data_type}/{station}.{code}.png',bbox_inches='tight',dpi=300)
    # else:
    #     plt.savefig(f'/Users/tnye/tsuquakes/plots/fourier_spec/obs/{data_type}/{station}.{code}.png',bbox_inches='tight',dpi=300)
        
    if synthetic:
        plt.savefig(f'/Users/tnye/FakeQuakes/parameters/{parameter}/{project}/plots/fourier_spec/{run}/{data_type}/{station}.{code}.png',bbox_inches='tight',dpi=300)
    else:
        plt.savefig(f'/Users/tnye/tsuquakes/plots/fourier_spec/obs/{data_type}/{station}.{code}.png',bbox_inches='tight',dpi=300)    
    
    plt.close()


    return()

