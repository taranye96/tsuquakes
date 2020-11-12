#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:49:52 2020

@author: tnye
"""

###############################################################################
# Module with functions used to calculate various IMs and plot spectra. These
# functions are imported into synthetic_calc_mpi.py. 
###############################################################################


def calc_time_to_peak(pgm, trace, IMarray, origintime, hypdist):
    """
    Calculates time to peak intensity measure (IM) from origin time and from
    estimated p-arrival.
    
    Inputs:
        pgm(float): Peak ground motion.
        trace: Trace object with times for this station. 
        IMarray: Array pgd was calculated on.  Could be the same as stream
                        if peak IM was calculated on one component. 
        origintime(datetime): Origin time of event.
        hypdist(float): Hypocentral distance (km). 
        
    """
    
    import numpy as np
    from obspy.core import UTCDateTime
    import datetime
    
    # Calculate time from origin 
    pgm_index = np.where(IMarray==pgm)
    tPGM_orig = trace.times(reftime=UTCDateTime(origintime))[pgm_index]
    
    # Calculate time from p-arrival
    p_time = hypdist/6.5
    dp = datetime.timedelta(seconds=p_time)
    p_arrival = origintime+dp
    tPGM_parriv = trace.times(reftime=UTCDateTime(p_arrival))[pgm_index]
    
    return(tPGM_orig, tPGM_parriv)


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
    
    import numpy as np
    from mtspec import mtspec
    from scipy import interpolate
    from scipy.stats import binned_statistic 

    # Read in file 
    tr = stream[0]
    data = tr.data
    delta = tr.stats.delta
    samprate = tr.stats.sampling_rate
    npts = tr.stats.npts
    
    nyquist = 0.5 * samprate

    ## Calc spectra amplitudes and frequencies
    
    # Switched number of tapers from 7 to 5.  Decreases computation time and
        # results are similar
    amp_squared, freq =  mtspec(data, delta=delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=npts, quadratic=True)
    
    # Take square root to get amplitude 
    amp = np.sqrt(amp_squared)
    
    # Use scipy interpolate function to fill in data in missing bins
    f = interpolate.interp1d(freq, amp)
    freq_new = np.arange(np.min(freq), np.max(freq), 0.0001)
    amp_new = f(freq_new)

    # Remove certain frequencies that are too low or high. 
    indexes = []
    
    for i, val in enumerate(freq_new):
        
        # Remove frequencies below 1/2 length of record
        if val <= 1/(delta*npts*0.5) :
            indexes.append(i)
        
        # Remove frequencies above 10 Hz for sm data because of the way it was processed 
        elif val > 10 and data_type in ('acc', 'vel'):
            indexes.append(i)

        # Remove frequencies above nyquist frequency for disp data
            # (it's already removed in the previous step for sm data)
        elif val > nyquist and data_type == 'disp': 
            indexes.append(i)
    
    # Remove any duplicate indexes
    indexes = np.unique(indexes)

    freq_new = np.delete(freq_new,indexes)
    amp_new = np.delete(amp_new,indexes) 
    
    if data_type == 'accel':
        data_type = 'sm'
    
    if data_type == 'sm':
        # Starting bins at 0.004 Hz (that is about equal to half the length
            # of the record for the synthetic and observed data) and ending at
            # 10 Hz because after that the sm data is unusable due to how it was
            # processed. 
        bins = np.logspace(np.log10(0.004), np.log10(10), num=21)
    
    elif data_type == 'disp':
        # Starting bins at 0.004 Hz (that is about equal to half the length
            # of the record for the synthetic and observed data) and ending at
            # 0.5 Hz because that is the nyquist frequency .
        bins = np.logspace(np.log10(0.004), np.log10(0.5), num=21)
    
    bin_means, bin_edges, binnumber = binned_statistic(freq_new,
                                                       amp_new,
                                                       statistic='mean',
                                                       bins=bins)
    
    # for i in range(len(bin_means)):
    #     bin_means[i] = 10**bin_means[i]
        
        
    return(bin_means, freq, amp)


def plot_spectra(stream, freqs, amps, data_type, plot_dir, synthetic=True, parameter='none', project='none', run='none'):
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
            plot_dir(str): Home part of path to save plots.
            synthetic(True/False): Plotting synthetics
            parameter(str): Name of parameter folder.
            project(str): Name of simulation project.  This will be the main
                          directory where the different runs will be store. 
            run(str): Synthetics run number.  This will be the directory where the
                      plots are stored.
        
        Output:
            Just saves the plots to ther respective directories. 
            
    """
    
    import matplotlib.pyplot as plt
    
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
        plt.savefig(f'{plot_dir}/parameters/{parameter}/{project}/plots/fourier_spec/{run}/{data_type}/{station}.{code}.png',bbox_inches='tight',dpi=300)
    else:
        plt.savefig(f'/Users/tnye/tsuquakes/plots/fourier_spec/obs/{data_type}/{station}.{code}.png',bbox_inches='tight',dpi=300)    
    
    plt.close()


    return()

