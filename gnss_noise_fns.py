#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:29:41 2023

@author: tnye
"""

# Imports
from mudpy.hfsims import windowed_gaussian,apply_spectrum
from mudpy.forward import gnss_psd
import numpy as np
from obspy import read

def add_synthetic_gnss_noise(st_E, st_N, st_Z, percentile=50):
    
    dt = st_E[0].stats.delta
    
    if dt == 0.5:
        st_E[0].decimate(factor=2)
        st_N[0].decimate(factor=2)
        st_Z[0].decimate(factor=2)
        
        st_E[0].stats.delta = 1.0
        st_N[0].stats.delta = 1.0
        st_Z[0].stats.delta = 1.0
    
    dt = st_E[0].stats.delta
    duration=st_E[0].stats.npts*dt
    
    # get white noise
    std=1.0 # this is a dummy parameter give it any value
    E_noise=windowed_gaussian(duration,dt,std=std,window_type=None)
    N_noise=windowed_gaussian(duration,dt,std=std,window_type=None)
    Z_noise=windowed_gaussian(duration,dt,std=std,window_type=None)
    
    f,Epsd,Npsd,Zpsd=gnss_psd(level=percentile,return_as_frequencies=True,return_as_db=False)
    
    #Covnert PSDs to amplitude spectrum
    Epsd = Epsd**0.5
    Npsd = Npsd**0.5
    Zpsd = Zpsd**0.5
    
    #apply the spectrum
    E_noise=apply_spectrum(E_noise,Epsd,f,dt,is_gnss=True)
    N_noise=apply_spectrum(N_noise,Npsd,f,dt,is_gnss=True)
    Z_noise=apply_spectrum(Z_noise,Zpsd,f,dt,is_gnss=True)
    
    #Remove mean for good measure
    E_noise -= np.mean(E_noise)
    N_noise -= np.mean(N_noise)
    Z_noise -= np.mean(Z_noise)
    
    st_E_noisy = st_E.copy()
    st_N_noisy = st_N.copy()
    st_Z_noisy = st_Z.copy()
    
    st_E_noisy[0].data = st_E_noisy[0].data + E_noise[:-1]
    st_N_noisy[0].data = st_N_noisy[0].data + N_noise[:-1]
    st_Z_noisy[0].data = st_Z_noisy[0].data + Z_noise[:-1]

    return(st_E_noisy,st_N_noisy,st_Z_noisy)


def add_real_gnss_noise(st_E, st_N, st_Z):
    
    from mtspec import mtspec
    from numpy.fft import fft, ifft
    import tsueqs_main_fns as tmf
    
    stn = st_E[0].stats.station
    
    dt = st_E[0].stats.delta
    
    if dt == 0.5:
        st_E[0].decimate(factor=2)
        st_N[0].decimate(factor=2)
        st_Z[0].decimate(factor=2)
        
        st_E[0].stats.delta = 1.0
        st_N[0].stats.delta = 1.0
        st_Z[0].stats.delta = 1.0
    
    dt = st_E[0].stats.delta
    
    if stn == 'KTET':
        E_noise = read('/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/noise/PKRT.LXE.mseed')
        N_noise = read('/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/noise/PKRT.LXN.mseed')
        Z_noise = read('/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/noise/PKRT.LXZ.mseed')
    else:
        E_noise = read(f'/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/noise/{stn}.LXE.mseed')
        N_noise = read(f'/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/noise/{stn}.LXN.mseed')
        Z_noise = read(f'/Users/tnye/tsuquakes/data/Mentawai2010/GNSS_data_processed_Dara_SAC/noise/{stn}.LXZ.mseed')
        
    signal_npts = st_E[0].stats.npts
    noise_npts = E_noise[0].stats.npts
    
    # Remove mean for good measure
    # E_noise -= np.mean(E_noise)
    # N_noise -= np.mean(N_noise)
    # Z_noise -= np.mean(Z_noise)
    
    st_E_noisy = st_E.copy()
    st_N_noisy = st_N.copy()
    st_Z_noisy = st_Z.copy()
    
    st_E_noisy[0].data = st_E_noisy[0].data + E_noise[0].data
    st_N_noisy[0].data = st_N_noisy[0].data + N_noise[0].data
    st_Z_noisy[0].data = st_Z_noisy[0].data + Z_noise[0].data
    
    # Baseline offset
    baseline_E = tmf.compute_baseline(st_E_noisy,10)
    baseline_N = tmf.compute_baseline(st_N_noisy,10)
    baseline_Z = tmf.compute_baseline(st_Z_noisy,10)
    
    basecorr_E = tmf.correct_for_baseline(st_E_noisy,baseline_E)
    basecorr_N = tmf.correct_for_baseline(st_N_noisy,baseline_N)
    basecorr_Z = tmf.correct_for_baseline(st_Z_noisy,baseline_Z)
    
    # E_mean = np.mean(st_E_noisy[0].data[:6])
    # N_mean = np.mean(st_N_noisy[0].data[:6])
    # Z_mean = np.mean(st_Z_noisy[0].data[:6])
    
    # st_E_noisy_demeaned[0].data = st_E_noisy[0].data - E_mean
    # st_N_noisy_demeaned[0].data = st_E_noisy[0].data - N_mean
    # st_Z_noisy_demeaned[0].data = st_E_noisy[0].data - Z_mean
    
    # E_noise_spec2, freq = mtspec(E_noise[0].data, delta=dt, time_bandwidth=4,
    #                             number_of_tapers=5, nfft=noise_npts, quadratic=True)
    # N_noise_spec2, freq = mtspec(N_noise[0].data, delta=dt, time_bandwidth=4,
    #                             number_of_tapers=5, nfft=noise_npts, quadratic=True)
    # Z_noise_spec2, freq = mtspec(Z_noise[0].data, delta=dt, time_bandwidth=4,
    #                             number_of_tapers=5, nfft=noise_npts, quadratic=True)
    
    # E_spec2, freq = mtspec(st_E[0].data, delta=dt, time_bandwidth=4,
    #                             number_of_tapers=5, nfft=signal_npts, quadratic=True)
    # N_spec2, freq = mtspec(st_N[0].data, delta=dt, time_bandwidth=4,
    #                             number_of_tapers=5, nfft=signal_npts, quadratic=True)
    # Z_spec2, freq = mtspec(st_Z[0].data, delta=dt, time_bandwidth=4,
    #                             number_of_tapers=5, nfft=signal_npts, quadratic=True)
    
    # E_noise_spec = np.sqrt(E_noise_spec2)
    # N_noise_spec = np.sqrt(N_noise_spec2)
    # Z_noise_spec = np.sqrt(Z_noise_spec2)
    
    # E_spec = np.sqrt(E_spec2)
    # N_spec = np.sqrt(N_spec2)
    # Z_spec = np.sqrt(Z_spec2)
    
    # E_noisy_spec = E_noise_spec * E_spec
    # N_noisy_spec = N_noise_spec * N_spec
    # Z_noisy_spec = Z_noise_spec * Z_spec
    
    # st_E_noisy[0].data = np.fft.ifft(E_noisy_spec)
    # st_N_noisy[0].data = np.fft.ifft(N_noisy_spec)
    # st_Z_noisy[0].data = np.fft.ifft(Z_noisy_spec)
    

    return(basecorr_E,basecorr_N,basecorr_Z)