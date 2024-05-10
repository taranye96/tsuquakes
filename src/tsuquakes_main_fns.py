#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:40:24 2022

@author: tnye
"""

##############################################################################

##############################################################################


def highpass(datastream,fcorner,fsample,order,zerophase=True):
    '''
    Make a highpass zero phase filter on stream object
    Input:
        datastream:                 Obspy stream object with data to filter
        fcorner:                    Corner frequency at which to highpass filter
        fsample:                    Sample rate in (Hz) - or 1/dt
        order:                      The numper of poles in the filter (use 2 or 4)
        zerophase:                  Boolean for whether or not the filter is zero phase
                                        (that does/doesn't advance or delay 
                                        certain frequencies). Butterworth filters 
                                        are not zero phase, they introduce a phase
                                        shift. If want zero phase, need to filter twice.
    Output:
        highpassedstream:           Obspy stream wth highpass filtered data
    '''
    from scipy.signal import butter,filtfilt,lfilter
    from numpy import array
    
    data = datastream[0].data
    
    fnyquist=fsample/2
    b, a = butter(order, array(fcorner)/(fnyquist),'highpass')
    if zerophase==True:
        data_filt=filtfilt(b,a,data)
    else:
        data_filt=lfilter(b,a,data)
    
    
    ## Make a copy of the original stream object:
    highpassedstream = datastream.copy()
    
    ## Add the highpassed data to it:
    highpassedstream[0].data = data_filt
    
    return highpassedstream


def lowpass(datastream,fcorner,fsample,order,zerophase=True):
    '''
    Make a highpass zero phase filter on stream object
    Input:
        datastream:                 Obspy stream object with data to filter
        fcorner:                    Corner frequency at which to highpass filter
        fsample:                    Sample rate in (Hz) - or 1/dt
        order:                      The numper of poles in the filter (use 2 or 4)
        zerophase:                  Boolean for whether or not the filter is zero phase
                                        (that does/doesn't advance or delay 
                                        certain frequencies). Butterworth filters 
                                        are not zero phase, they introduce a phase
                                        shift. If want zero phase, need to filter twice.
    Output:
        highpassedstream:           Obspy stream wth highpass filtered data
    '''
    from scipy.signal import butter,filtfilt,lfilter
    from numpy import array
    
    data = datastream[0].data
    
    fnyquist=fsample/2
    b, a = butter(order, array(fcorner)/(fnyquist),'lowpass')
    if zerophase==True:
        data_filt=filtfilt(b,a,data)
    else:
        data_filt=lfilter(b,a,data)
    
    
    ## Make a copy of the original stream object:
    lowpassedstream = datastream.copy()
    
    ## Add the highpassed data to it:
    lowpassedstream[0].data = data_filt
    
    return lowpassedstream


def compute_baseline(timeseries_stream,numsamples=100):
    '''
    Given a time series stream object, gain corrected, find what the average 
    baseline is before the event begins
    Input:
        timeseries_stream:      Obspy stream object with a time series on which
                                    to determine the baseline, units in meters
        numsamples:             Float with the number of samples to use in 
                                    computing the pre-event baseline. 
                                    Defualt: 100 for strong motion
                                    ***Displacement should be lower
    Output:
        baseline:               Float with the baseline to use to subtract from 
                                    a future time series, in m/sec/sec
                                    I.e., if the baseline is >0 this number
                                    will be positive and should be subtracted.
                                    from the whole time series
    '''
    import numpy as np
    
    seismic_amp = timeseries_stream[0].data
    
    ## Define the baseline as the mean amplitude in the first 100 samples:
    baseline = np.median(seismic_amp[0:numsamples])
    
    ## Return value:
    return baseline


def correct_for_baseline(timeseries_stream_prebaseline,baseline):
    '''
    Correct a time series by the pre-event baseline (units in meters)
    Input:
        timeseries_stream_prebaseline:      Obspy stream object with a time series
                                                units in meters, gain corrected
                                                but not baseline corrected.
        baseline:                           Float with the pre-event baseline to 
                                                use to subtract, in distance units
                                                of meters. 
    Output:
        timeseries_stream_baselinecorr:     Obspy stream object with a time series,
                                                units in meters, gain and baseline
                                                corrected.
    '''
    
    ## Get the pre-baseline corrected time series amplitudes:
    amplitude_prebaseline = timeseries_stream_prebaseline[0].data
    
    ## Correct by baseline:
    amplitude_baselinecorr = amplitude_prebaseline - baseline
    
    ## Get a copy of the current stream object:
    timeseries_stream_baselinecorr = timeseries_stream_prebaseline.copy()
    
    ## Put the corrected amplitudes into the data of the baselinecorr object:
    timeseries_stream_baselinecorr[0].data = amplitude_baselinecorr
    
    ## Return:
    return timeseries_stream_baselinecorr


def correct_for_gain(timeseries_stream,gain):
    '''
    Correct a given time series stream object for the gain, as specified in the
    .chan file.
    Input:
        timeseries_stream:      Obspy stream object with a time series, to 
                                    correct for gain
        gain:                   Float with the value of gain, to divide the 
                                    time series by to obtain the cm/s/s units.
    Output:
        timeseries_stream_corr: Gain-corrected obspy stream object with a time 
                                    series, data in units of m/s/s or m
    '''
    
    
    timeseries_cm = timeseries_stream[0].data/gain
    
    ## Convert to m: (either cm/s/s to m/s/s, or cm to m):
    timeseries_m = timeseries_cm/100.
    
    ## Add this back into the corrected stream object:
    timeseries_stream_corr = timeseries_stream.copy()
    timeseries_stream_corr[0].data = timeseries_m
    
    ## Return:
    return timeseries_stream_corr


def determine_Td(threshold,timeseries_stream):
    '''
    Given a time series and threshold (acceleration), determine the duration
    of shaking for that event. The threshold must be passed for 1 sec, and have
    ended for 1 sec nonoverlapping to define the beginning and end of the event,
    respectively.
    Input:
        threshold:              Float with the threshold (m/sec/sec) value of acceleration
                                    to use for the beginning/end of event.
        timeseries_stream:      Obspy stream object with a time series on which 
                                    to compute the duration, units in meters,
                                    gain and baseline corrected.
                                    
    Output:
        Td:                     Duration, in seconds, of the shaking with threshold
        t_start:                Start time (in seconds) for this time series
        t_end:                  End time (in seconds) for this time series
    '''
    
    import numpy as np
    
    
    ## Get the times - this should give time in seconds for the waveform:
    times = timeseries_stream[0].times()
    
    ## Get the amplitudes of acceleration:
    timeseries = timeseries_stream[0].data
       
    
    ## Find the indices where the threshold is passed:
    grtr_than_threshold = np.where(np.abs(timeseries) > threshold)[0]
    
    ## If there are no places where the amplitude is greater than this threshold,
    ##    set the Td to 0, as well as t_start and t_end:
    if len(grtr_than_threshold) < 1:
        Td = 0
        t_start = 0
        t_end = 0
        
    else:
        # Total duration, start, and end time:
        t_start = times[grtr_than_threshold[0]]
        t_end = times[grtr_than_threshold[-1]]
    
        Td = t_end - t_start
    
    ## Return the duration, start time, and end time:
    return Td, t_start, t_end


def get_geom_avg_3comp(E_record, N_record, Z_record):
    """
    Get the geometric average of the three components of a record.

    Inputs:
        E_record(array): East-West component trace data.
        N_record(array): North-South component trace data.
        Z_record(array): Vertical component trace data.
    
    Return:
        geom_avg(array): Record of geometric average.
    """

    import numpy as np

    geom_avg = np.cbrt(E_record * N_record * Z_record)

    return (geom_avg)


def get_geom_avg_2comp(E_record, N_record):
    """
    Get the geometric average of two components of a record (most likely the
    horizontal components).

    Inputs:
        E_record(array): East-West component trace data.
        N_record(array): North-South component trace data.
    
    Return:
        geom_avg(array): Record of geometric average.
    """

    import numpy as np

    geom_avg = np.sqrt(E_record * N_record)

    return (geom_avg)


def get_eucl_norm_3comp(E_record, N_record, Z_record):
    """
    Get the euclidean norm of the three components of a record.  This is
    equivalent to calculating the magnitude of a vector. 

    Inputs:
        E_record(array): East-West component trace data.
        N_record(array): North-South component trace data.
        Z_record(array): Vertical component trace data.
    
    Return:
        eucl_norm(array): Record of euclidian norm.
    """

    import numpy as np

    eucl_norm = np.sqrt(E_record**2 + N_record**2 + Z_record**2)

    return (eucl_norm)


def get_eucl_norm_2comp(E_record, N_record):
    """
    Get the euclidean norm of two components of a record (most likely the
    horizontal components).  This is equivalent to calculating the magnitude of
    a vector. 

    Inputs:
        E_record(array): East-West component trace data.
        N_record(array): North-South component trace data.
    
    Return:
        eucl_norm(array): Full record of euclidian norm.
    """

    import numpy as np
    
    if len(E_record)!=len(N_record):
        ind = np.min(np.array([len(E_record),len(N_record)]))
        eucl_norm = np.sqrt(E_record[:ind]**2 + N_record[:ind]**2)
    else:
        eucl_norm = np.sqrt(E_record**2 + N_record**2)

    return (eucl_norm)


def compute_rotd50(accel1, accel2):
    """
    Computes rotd50 of a timeseries
    
    Input:
        accel1(array): Timeseries 1 accelerations
        accel2(array): Timeseries 2 accelerations
    
    Return:
        rotd50_accel(array): 50th percentile rotated timeseries
    """
    
    import numpy as np
    
    percentile = 50
    
    accels = np.array([accel1,accel2])
    angles = np.arange(0, 180, step=0.5)
    
    # Compute rotated time series
    radians = np.radians(angles)
    coeffs = np.c_[np.cos(radians), np.sin(radians)]
    rotated_time_series = np.dot(coeffs, accels)
    
    # Sort this array based on the response
    peak_responses = np.abs(rotated_time_series).max(axis=1)
    
    # Get the peak response at the requested percentiles
    p_peak_50 = np.percentile(peak_responses, percentile)
    
    # Get index of 50th percentile
    ind = np.where(peak_responses==peak_responses.flat[np.abs(peak_responses - p_peak_50).argmin()])[0][0]
    
    # Get rotd50 timeseries
    rotd50_accel = rotated_time_series[ind]
    
    return(rotd50_accel)


def compute_repi(stlon,stlat,hypolon,hypolat):
    '''
    Compute the hypocentral distance for a given station lon, lat and event hypo
    Input:
        stlon:          Float with the station/site longitude
        stlat:          Float with the station/site latitude
        hypolon:        Float with the hypocentral longitude
        hypolat:        Float with the hypocentral latitude
    Output:
        repi:           Float with the epicentral distance, in km
    '''
    
    from pyproj import Geod
    
    
    ## Make the projection:
    p = Geod(ellps='WGS84')
    
    ## Apply the projection to get azimuth, backazimuth, distance (in meters): 
    az,backaz,horizontal_distance = p.inv(stlon,stlat,hypolon,hypolat)

    ## Put them into kilometers:
    horizontal_distance = horizontal_distance/1000.
 
    ## Epicentral distance is horizontal distance:
    repi = horizontal_distance
    
    return repi


def compute_rhyp(stlon,stlat,stelv,hypolon,hypolat,hypodepth):
    '''
    Compute the hypocentral distance for a given station lon, lat and event hypo
    Input:
        stlon:          Float with the station/site longitude
        stlat:          Float with the station/site latitude
        stelv:          Float with the station/site elevation (m)
        hypolon:        Float with the hypocentral longitude
        hypolat:        Float with the hypocentral latitude
        hypodepth:      Float with the hypocentral depth (km)
    Output:
        rhyp:           Float with the hypocentral distance, in km
    '''
    
    import numpy as np
    from pyproj import Geod
    
    
    ## Make the projection:
    p = Geod(ellps='WGS84')
    
    ## Apply the projection to get azimuth, backazimuth, distance (in meters): 
    az,backaz,horizontal_distance = p.inv(stlon,stlat,hypolon,hypolat)

    ## Put them into kilometers:
    horizontal_distance = horizontal_distance/1000.
    stelv = stelv/1000
    ## Hypo deptha lready in km, but it's positive down. ST elevation is positive
    ##    up, so make hypo negative down (so the subtraction works out):
    hypodepth = hypodepth * -1
    
    ## Get the distance between them:
    rhyp = np.sqrt(horizontal_distance**2 + (stelv - hypodepth)**2)
    
    return rhyp


def compute_rrup(rupt_file, stlon, stlat):
    """
    Calculates closest rupture distance.
    
    Inputs:
        rupt_file(str): Path to .rupt file
        stlon(float): Station longitude
        stlat(float): Station latitude
    
    Returns:
        rrup(float): Closest rupture distance
    """
    
    import numpy as np
    from pyproj import Geod   
   
    #get rupture
    rupt = np.genfromtxt(rupt_file)
    Nsubfaults = len(rupt)
    
    #keep only those with slip
    i = np.where(rupt[:,12]>0)[0]
    rupt = rupt[i,:]
    
    #get Rrupt
    #projection obnject
    p = Geod(ellps='WGS84')
    
    #lon will have as many rows as Vs30 points and as many columns as subfautls in rupture
    Nsubfaults = len(rupt)
    lon_surface = np.tile(stlon,(Nsubfaults,1)).T
    lat_surface = np.tile(stlat,(Nsubfaults,1)).T
    lon_subfaults = np.tile(rupt[:,1],(len(stlon),1))-360
    lat_subfaults = np.tile(rupt[:,2],(len(stlon),1))
    az,baz,dist = p.inv(lon_surface,lat_surface,lon_subfaults,lat_subfaults)
    dist = dist/1000
    
    #get 3D distance
    z = np.tile(rupt[:,3],(len(stlon),1))
    xyz_dist = (dist**2 + z**2)**0.5
    rrup = xyz_dist.min(axis=1)
    
    return(rrup)


def add_synthetic_gnss_noise(st_E, st_N, st_Z, percentile=50):
    '''
    Adds synthetic noise to gnss data based off a percetnile noise provided
    (Melgar et al., 2020)

    Inputs:
        st_E(stream): East component GNSS obspy stream
        st_N(stream): North component GNSS obspy stream
        st_Z(stream): Vertical component GNSS obspy stream
        percentile(int): Noise percentile (options are 1, 10, 20, 30, 40, 50,
                                           60, 70, 80, 90)
    
    Return:
        st_E_noisy(stream): East component GNSS obspy stream with syntehtic noise
        st_N_noisy(stream): North component GNSS obspy stream with syntehtic noise
        st_Z_noisy(stream): Vertical component GNSS obspy stream with syntehtic noise
    '''
    
    import numpy as np
    from mudpy.hfsims import windowed_gaussian,apply_spectrum
    from mudpy.forward import gnss_psd
    
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
    '''
    Adds real noise to the 2010 Mentawai GNSS waveforms

    Inputs:
        st_E(stream): East component GNSS obspy stream
        st_N(stream): North component GNSS obspy stream
        st_Z(stream): Vertical component GNSS obspy stream
    
    Return:
        basecorr_E(stream): East component GNSS obspy stream with real noise
        basecorr_N(stream): North component GNSS obspy stream with real noise
        basecorr_Z(stream): Vertical component GNSS obspy stream with real noise
    '''
    
    from obspy import read
    import tsuquakes_main_fns as tmf
    
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
    
    if stn == 'KTET': #No longer have the a longer timeseries for this station (it's been removed) 
        E_noise = read('/Users/tnye/tsuquakes/data/GNSS_data_processed_Dara_SAC/noise/PKRT.LXE.mseed')
        N_noise = read('/Users/tnye/tsuquakes/data/GNSS_data_processed_Dara_SAC/noise/PKRT.LXN.mseed')
        Z_noise = read('/Users/tnye/tsuquakes/data/GNSS_data_processed_Dara_SAC/noise/PKRT.LXZ.mseed')
    else:
        E_noise = read(f'/Users/tnye/tsuquakes/data/GNSS_data_processed_Dara_SAC/noise/{stn}.LXE.mseed')
        N_noise = read(f'/Users/tnye/tsuquakes/data/GNSS_data_processed_Dara_SAC/noise/{stn}.LXN.mseed')
        Z_noise = read(f'/Users/tnye/tsuquakes/data/GNSS_data_processed_Dara_SAC/noise/{stn}.LXZ.mseed')
    
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

    return(basecorr_E,basecorr_N,basecorr_Z)


def get_Rp(stlon, stlat, stelev, rupture, exp=-2.3):
    """
    Calculates generalized mean rupture distance (Rp) for a station.
    
    Inputs:
        stlon(float): Station longitude
        stlat(float): Station latitude
        stelev(float): Station elevation (km)
        rupt_file(string): Path to .rupt file
        exp(float): Power of the mean
    
    Returns:
        Rp(float): Generalized mean rupture distance  
    """
    
    import numpy as np
    import pandas as pd
    import tsueqs_main_fns as main
    
    rupt = np.genfromtxt(rupture)
    ind = np.where((rupt[:,8]!=0) & (rupt[:,9]!=0))[0]
    
    total_slip = np.sum(np.sqrt(rupt[:,8]**2+rupt[:,9]**2))
    
    weighted_dist = []
    for i in ind:
        w_i = np.sqrt(rupt[i,8]**2 + rupt[i,9]**2)/total_slip
        R_i = main.compute_rhyp(stlon,stlat,stelev,rupt[i,1],rupt[i,2],rupt[i,3])
        weighted_dist.append(w_i*(R_i**exp))
    
    Rp = np.sum(weighted_dist)**(1/exp)
    
    
    return(Rp)


def get_pgd_scaling(Mw, R, model):
    """
    Empirically estimates PGD from hypocentral distance using the scaling
    relation from Goldberg et al. (2021).
    
    Inputs:
        MW(float): Moment magnitude
        R(float): Distance, either Rp for GA21 or Rhyp for MA15 
    
    Returns:
        PGD(float): Peak ground displacement (m) 
    """
    
    import numpy as np
    
    if model == 'GA21_joint':
        A = -5.902
        B = 1.303
        C = -0.168
        sigma = 0.255
        logpgd = A + B*Mw + C*Mw*np.log10(R)
        pgd = 10**logpgd
        pgd = pgd/100
        
    elif model == 'GA21_obs':
        A = -3.841
        B = 0.919
        C = -0.122
        sigma = 0.252
        logpgd = A + B*Mw + C*Mw*np.log10(R)
        pgd = 10**logpgd
        pgd = pgd/100
    
    elif model == 'MA15':
        A = -4.434
        B = 1.047
        C = -0.138
        sigma = 0.27
        logpgd = A + B*Mw + C*Mw*np.log10(R)
        pgd = 10**logpgd
        pgd = pgd/100
        
    return(pgd, sigma)


def calc_time_to_peak(pgm, trace, IMarray, origintime, hypdist):
    """
    Calculates time to peak intensity measure (IM) from origin time and from
    estimated p-arrival.
    
    Inputs:
        pgm(float): Peak ground motion.
        trace: Trace object with times for this station. 
        IMarray: Array pgd was calculated on. 
        origintime(datetime): Origin time of event.
        hypdist(float): Hypocentral distance (km). 
        
    """
    
    import numpy as np
    from obspy.core import UTCDateTime
    
    # Calculate time from origin 
    pgm_index = np.where(IMarray==pgm)
    tPGM = trace.times(reftime=UTCDateTime(origintime))[pgm_index]
    
    return(tPGM)


def calc_spectra(stream, data_type):
    """
    Calculates average spectra values in 20 bins for displacement, acceleration,
    and velocity waveforms.

    Inputs:
        stream: Obspy stream. 
        data_type(str): Data type used for determining bin edges.
            Options:
                gnss
                sm

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
    
    # Determine nyquist frequency
    nyquist = 0.5 * samprate
    

    # Calc spectra amplitudes and frequencies 
        # Switched number of tapers from 7 to 5.  Decreases computation time and
        # results are similar
    amp_squared, freq =  mtspec(data, delta=delta, time_bandwidth=4, 
                              number_of_tapers=5, nfft=npts, quadratic=True)
     
    # Convert from power spectra to amplitude spectra
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
        elif val > 10 and data_type == 'sm':
            indexes.append(i)

        # Remove frequencies above nyquist frequency for disp data
            # (it's already removed in the previous step for sm data)
        elif val > nyquist and data_type == 'disp': 
            indexes.append(i)
    
    # Remove any duplicate indexes
    indexes = np.unique(indexes)
    freq_new = np.delete(freq_new,indexes)
    amp_new = np.delete(amp_new,indexes) 
    
    # Set up bins
    if data_type == 'sm':
        # Starting bins at 0.004 Hz (that is about equal to 1/half the length
            # of the record for the synthetic and observed data) and ending at
            # 10 Hz because after that the sm data is unusable due to how it was
            # processed. 
        bins = np.logspace(np.log10(0.1), np.log10(10), num=11)
    
    elif data_type == 'gnss':
        # Starting bins at 0.004 Hz (that is about equal to 1/half the length
            # of the record for the synthetic and observed data) and ending at
            # 0.5 Hz because that is the nyquist frequency .
        bins = np.logspace(np.log10(0.1), np.log10(0.5), num=11)
    
    bin_means, bin_edges, binnumber = binned_statistic(freq_new,amp_new,
                                                       statistic='mean',bins=bins)
    # Obtain bin means from bin edges
    freq_means = []
    for i in range(len(bin_edges)):
        if i != 0:
            # mean = np.exp((np.log10(bin_edges[i])+np.log10(bin_edges[i-1]))/2)
            mean = np.sqrt(bin_edges[i]*bin_edges[i-1])
            freq_means.append(mean)
        
    # return(bin_means, freq, amp, sigma)
    return(freq_means, bin_means)



def calc_res(obs_file, home_dir, parameter, project, run, dtype):
    """
    Calculates residuals between synthetic and observed data, and puts residuals
    into a dataframe.
    
    Inputs:
        obs_file(str): Path to observed data flatfile.
        parameter(str): Folder name of parameter being varied.
        project(str): Folder name of specific project within parameter folder.
        run(str): Individual run name within certain project. 
        ln(T/F): If true, calculates the natural log of the residuals.
    Return:
        pgd_res(float): PGD residual. 
        pga_res(float): PGA residual.
        tPGD(float): tPGD residual.
        spectra_res(array): Array of residuals for all the spectra bins.

    """
    
    import numpy as np
    import pandas as pd
    
    # Synthetic values
    syn_df = pd.read_csv(f'{home_dir}/{parameter}/{project}/flatfiles/IMs/{run}_{dtype}.csv')
    
    # Observed values
    obs_df = pd.read_csv(obs_file)
    
    if dtype=='gnss':
        # syn_pgd = np.array(syn_df['pgd'])
        # syn_tPGD_orig = np.array(syn_df['tPGD_origin'])
        # syn_tPGD_parriv = np.array(syn_df['tPGD_parriv'])
        # syn_spectra = np.array(syn_df.iloc[:,24:])
        
        # Remove station KTET from dataframe
            # Don't have access to the data to reprocess like we do with the
            # other stations
        drop_ind = obs_df[obs_df['station'] == 'KTET'].index
        obs_df = obs_df.drop(drop_ind)
        
        syn_pgd = np.delete(np.array(syn_df['pgd']),1) # delete results for ktet
        syn_tPGD = np.delete(np.array(syn_df['tPGD']),1) # delete results for ktet
        syn_spectra = np.delete(np.array(syn_df.iloc[:,23:33]),1,axis=0)
        
        # Observed values
        obs_pgd = np.array(obs_df['pgd'])
        obs_tPGD = np.array(obs_df['tPGD_origin'])
        obs_spectra = np.array(obs_df.iloc[:,23:]) 
        
        # Calc residuals
        pgd_res = np.log(obs_pgd) - np.log(syn_pgd)
        tPGD_res_linear = obs_tPGD - syn_tPGD
        tPGD_res_ln = np.log(obs_tPGD) - np.log(syn_tPGD)
        spectra_res = np.log(obs_spectra) - np.log(syn_spectra)
        
        out = [pgd_res,tPGD_res_ln,tPGD_res_linear,spectra_res]
    
    elif dtype=='sm':
        syn_pga = np.array(syn_df['pga'])
        syn_tPGA = np.array(syn_df['tPGA'])
        syn_spectra = np.array(syn_df.iloc[:,23:33]) # syn_df only has 1 tPGD column whereas obs_df has 2
    
        # Observed values
        obs_pga = np.array(obs_df['pga'])
        obs_tPGA = np.array(obs_df['tPGA_origin'])
        obs_spectra = np.array(obs_df.iloc[:,23:])

        # Calc residuals
        pga_res = np.log(obs_pga) - np.log(syn_pga)
        tPGA_res_linear = obs_tPGA - syn_tPGA
        tPGA_res_ln = np.log(obs_tPGA) - np.log(syn_tPGA)
        spectra_res = np.log(obs_spectra) - np.log(syn_spectra)
        
        out = [pga_res,tPGA_res_ln,tPGA_res_linear,spectra_res]
    
    return(out)


def zhao2006(M,hypodepth,rrup,vs30):
    '''
    Computes PGA with Zhao et al. (2006) using OpenQuake engine.
    
        Inputs:
            M(float): Magnitude
            hypodepth(float): Hypocenter depth (km)
            rrup: Rrup (km)
            vs30: Vs30 (m/s)
            
        Return:
            lmean_zhao06(array): Mean PGA
            sd_zhao06(array): Standard deviation of PGA
    '''
    
    ## Open quake stuff:
    from openquake.hazardlib import imt, const
    from openquake.hazardlib.gsim.base import RuptureContext
    from openquake.hazardlib.gsim.base import DistancesContext
    from openquake.hazardlib.gsim.base import SitesContext
    from openquake.hazardlib.gsim.zhao_2006 import ZhaoEtAl2006SInter
    import pandas as pd
    import numpy as np
             
    ## Define intensity measure and uncertainty
    im_type = imt.PGA()
    uncertaintytype = const.StdDev.TOTAL
    
    ## Set GMPEs:
    zhao = ZhaoEtAl2006SInter() 
    
    ## Make contexts:
    rctx = RuptureContext()
    dctx = DistancesContext()
    sctx = SitesContext()
    
    ## Create rupture context
    dctx.rrup = rrup
    rctx.mag = M
    rctx.hypo_depth = hypodepth

    ## Create site context
    sctx.vs30 = np.ones_like(dctx.rrup) * vs30
    
    sitecol_dict = sitecol_dict = {'sids':[1]*len(vs30),'vs30':vs30,
                    'vs30measured':[None]*len(vs30),'z1pt0':[None]*len(vs30),
                    'z2pt5':[None]*len(vs30)}
    sitecollection = pd.DataFrame(sitecol_dict)

    sctx = SitesContext(sitecol=sitecollection)    
    
    ln_median_zhao,sd_zhao = zhao.get_mean_and_stddevs(sctx, rctx, dctx, im_type, [uncertaintytype])
    
    return (ln_median_zhao, sd_zhao)


def plot_spec_comp(plot_dir,syn_freqs, syn_spec, obs_freqs, obs_spec, stn_list, hypdists, data_type, home, parameter, project, run, spec_type):
    """
    Makes a figure comparing observed spectra to synthetic spectra with
    subplots for each station. 

    Inputs:
        syn_freqs(list): Array of list of frequencies obtained when computing
                         Fourier spectra of the synthetics for each station
        syn_amps(list): Array of list of amplitudes obtained when computing
                        Fourier spectra of the synthetics for each station
        obs_freqs(list): Array of list of frequencies obtained when computing
                         Fourier spectra of the observed data for each station
        obs_amps(list): Array of list of amplitudes obtained when computing
        stn_list(list): List of station names
                        Fourier spectra of the observed data for each station
        hypdists(list): List of hypocentral distances correlating with the 
                        stations used to get spectra
        data_type(str): Type of data
                            Options:
                                disp
                                acc
                                vel
        home(str): Base of path to save plots.
        parameter(str): Name of parameter folder.
        project(str): Name of simulation project.
        run(str): Synthetics run number.

    Output:
            Just saves the plots to ther respective directories. 
            
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from os import path, makedirs
            
    # Set figure axes
    if data_type == 'disp':
       units = 'm*s'
       ylim = 10**-5, 9*10**-1
       xlim = 0.004, 5*10**-1
       dim = 3,3
    elif data_type == 'acc':
       units = 'm/s'
       ylim = 7*10**-15, 6*10**-1
       xlim = .002, 10
       dim = 3,3
    elif data_type == 'vel':
       units = 'm'
       ylim = 6*10**-15, 8*10**-2
       xlim = .002, 10
       dim = 3,3
    
    # Sort hypdist and get indices
    sort_id = np.argsort(np.argsort(hypdists))
    sort_hypdists = np.sort(hypdists)
    
    # Sort freq and amps based off hypdist
    def sort_list(list1, list2): 
        zipped_pairs = zip(list2, list1) 
        z = [x for _, x in sorted(zipped_pairs)] 
        return z
    
    sort_syn_freqs = sort_list(syn_freqs, sort_id)
    sort_syn_spec = sort_list(syn_spec, sort_id)
    sort_obs_freqs = sort_list(obs_freqs, sort_id)
    sort_obs_spec = sort_list(obs_spec, sort_id)
    sort_stn_name = sort_list(stn_list, sort_id)
    
    if data_type == 'disp':
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,8))
        k = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].loglog(sort_syn_freqs[k],sort_syn_spec[k],lw=1,c='C1',ls='-',label='synthetic')
                    axs[i][j].loglog(sort_obs_freqs[k],sort_obs_spec[k],lw=1,c='steelblue',ls='-',label='observed')
                    axs[i][j].grid(linestyle='--')
                    axs[i][j].text(0.025,5E-2,'2-comp eucnorm',transform=axs[i][j].transAxes,size=10)
                    axs[i][j].text(0.98,5E-2,f'Hypdist={int(sort_hypdists[k])}km',horizontalalignment='right',
                                    transform=axs[i][j].transAxes,size=10)
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    axs[i][j].set_xlim(xlim)
                    axs[i][j].set_ylim(ylim)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    if i < 1:
                        axs[i][j].set_xticklabels([])
                    if i == 1 and j == 0:
                        axs[i][j].set_xticklabels([])
                    if j > 0:
                        axs[i][j].set_yticklabels([])
                    k += 1
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.delaxes(axs[2][1])
        fig.delaxes(axs[2][2])
        fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
        fig.supylabel(f'Amplitude ({units})',fontsize=12)
        fig.supxlabel('Frequency (Hz)',fontsize=12)
        fig.legend(handles, labels, loc=(0.72,0.25), framealpha=None, frameon=False)
        fig.text(0.72, 0.2, (r"$\bf{" + 'Project:' + "}$" + '' + project), size=10, horizontalalignment='left')
        fig.text(0.72, 0.175, (r'$\bf{' + 'Run:' + '}$' + '' + run), size=10, horizontalalignment='left')
        fig.text(0.72, 0.15, (r'$\bf{' + 'DataType:' '}$' + '' + data_type), size=10, horizontalalignment='left')
        plt.subplots_adjust(left=0.11, bottom=0.09, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
        
        if not path.exists(f'{plot_dir}/comparison/spectra/{data_type}'):
            makedirs(f'{plot_dir}/comparison/spectra/{data_type}')
            
        plt.savefig(f'{plot_dir}/comparison/spectra/{data_type}/{run}_{data_type}_{spec_type}.png', dpi=300)
        plt.close()
    
    else:
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,9.5))
        k = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].loglog(sort_syn_freqs[k],sort_syn_spec[k],lw=1,c='C1',ls='-',label='synthetic')
                    axs[i][j].loglog(sort_obs_freqs[k],sort_obs_spec[k],lw=1,c='steelblue',ls='-',label='observed')
                    axs[i][j].grid(linestyle='--')
                    axs[i][j].text(0.025,5E-2,'rotd50',transform=axs[i][j].transAxes,size=10)
                    axs[i][j].text(0.98,5E-2,f'Hypdist={int(sort_hypdists[k])}km',horizontalalignment='right',
                                    transform=axs[i][j].transAxes,size=10)
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    axs[i][j].set_xlim(xlim)
                    axs[i][j].set_ylim(ylim)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    if i < dim[0]-1:
                        axs[i][j].set_xticklabels([])
                    if i == dim[0]-2 and j == 0:
                        axs[i][j].set_xticklabels([])
                    if j > 0:
                        axs[i][j].set_yticklabels([])
                    k += 1
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.suptitle('Fourier Spectra Comparison', fontsize=12, y=0.98)
        fig.supylabel(f'Amplitude ({units})',fontsize=12)
        fig.supxlabel('Frequency (Hz)',fontsize=12,y=0.125)
        fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
        fig.text(0.72, 0.075, (r"$\bf{" + 'Project:' + "}$" + '' + project), size=10, horizontalalignment='left')
        fig.text(0.72, 0.05, (r'$\bf{' + 'Run:' + '}$' + '' + run), size=10, horizontalalignment='left')
        fig.text(0.72, 0.025, (r'$\bf{' + 'DataType:' '}$' + '' + data_type), size=10, horizontalalignment='left')
        plt.subplots_adjust(left=0.11, bottom=0.2, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
        
        if not path.exists(f'{plot_dir}/comparison/spectra/{data_type}'):
            makedirs(f'{plot_dir}/comparison/spectra/{data_type}')
        
        plt.savefig(f'{plot_dir}/comparison/spectra/{data_type}/{run}_{data_type}_{spec_type}.png', dpi=300)
        plt.close()

        
def plot_wf_comp(plot_dir,syn_times,syn_amps,stn_list,hypdists,data_type,wf_type,home,parameter,project,run,component,start,end):
    """
    Makes a figure comparing observed waveforms to synthetic waveforms with
    subplots for each station. 

    Inputs:
        syn_freqs(list): Array of list of times for synthetic waveforms
        syn_amps(list): Array of list of amplitudes for synthetic waveforms
        obs_freqs(list): Array of list of times for observed waveforms
        obs_amps(list): Array of list of amplitudes for observed waveforms
        stn_list(list): List of station names
                        Fourier spectra of the observed data for each station
        hypdists(list): List of hypocentral distances correlating with the 
                        stations used to get spectra
        data_type(str): Type of data
                            Options:
                                disp
                                acc
                                vel
        home(str): Base of path to save plots.
        parameter(str): Name of parameter folder.
        project(str): Name of simulation project.
        run(str): Synthetics run number.

    Output:
            Just saves the plots to ther respective directories. 
            
    """
    
    import numpy as np
    from obspy import read
    from glob import glob
    from os import path, makedirs
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MultipleLocator, ScalarFormatter
    
    if data_type == 'disp':
        if component != 'avg':
            obs_files = sorted(glob(f'/Users/tnye/tsuquakes/data/processed_waveforms/individual/disp/*LX{component}*'))
        else:
            obs_files = sorted(glob(f'/Users/tnye/tsuquakes/data/processed_waveforms/average/eucnorm_3comp/disp/*'))
    
    if data_type == 'acc':
        exclusions = ['/CGJI','/CNJI','/LASI','/MLSI','/PPBI','/PSI','/TSI'] # far stations containing surface waves
        if component != 'avg':
            obs_files = [file for file in sorted(glob(f'/Users/tnye/tsuquakes/data/processed_waveforms/individual/acc/*HN{component}*')) \
                     if not any(exclude in file for exclude in exclusions)]
        else:
            obs_files = [file for file in sorted(glob(f'/Users/tnye/tsuquakes/data/processed_waveforms/average/rotd50/acc/*')) \
                     if not any(exclude in file for exclude in exclusions)]
    
    if data_type == 'vel':
        exclusions = ['/CGJI','/CNJI','/LASI','/MLSI','/PPBI','/PSI','/TSI'] # far stations containing surface waves
        if component != 'avg':
            obs_files = [file for file in sorted(glob(f'/Users/tnye/tsuquakes/data/processed_waveforms/individual/vel/*HN{component}*')) \
                     if not any(exclude in file for exclude in exclusions)]
        else:
            obs_files = [file for file in sorted(glob(f'/Users/tnye/tsuquakes/data/processed_waveforms/average/rotd50/vel/*')) \
                     if not any(exclude in file for exclude in exclusions)]
    obs_amps = []
    obs_times = []
    for file in obs_files:
        obs_amps.append(read(file)[0].data.tolist())
        obs_times.append(read(file)[0].times('matplotlib').tolist())
    
    # Set figure parameters based on data type
    if data_type == 'disp':
           units = 'm'
           dim = 3,3
    elif data_type == 'acc':
           units = 'm/s/s'
           dim = 3,3
    elif data_type == 'vel':
           units = 'm/s'
           dim = 3,3
    
    # Sort hypdist and get sorted indices
    sort_hypdists = np.sort(hypdists)
    sort_syn_times = [syn_times[i] for i in np.argsort(hypdists)]
    sort_syn_amps = [syn_amps[i] for i in np.argsort(hypdists)]
    sort_obs_times = [obs_times[i] for i in np.argsort(hypdists)]
    sort_obs_amps = [obs_amps[i] for i in np.argsort(hypdists)]
    sort_stn_name = [stn_list[i] for i in np.argsort(hypdists)]
    
    
    # Make figure and subplots
    if data_type == 'disp': 
        if component != 'avg':
            label = f'LY{component}'
        else:
            label = '3-comp eucnorm'
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,8))
        k = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].plot(sort_syn_times[k],sort_syn_amps[k],
                                     color='C1',alpha=0.7,lw=0.4,label='synthetic')
                    axs[i][j].plot(sort_obs_times[k],sort_obs_amps[k],
                                     'steelblue',alpha=0.7,lw=0.4,label='observed')
                    axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    axs[i][j].text(0.98,5E-2,f'Hypdist={int(sort_hypdists[k])}km',horizontalalignment='right',
                                    transform=axs[i][j].transAxes,size=10)
                    axs[i][j].text(0.98,0.9,label,transform=axs[i][j].transAxes,size=10,horizontalalignment='right')
                    if i < 1:
                        axs[i][j].set_xticklabels([])
                    if i == 1 and j == 0:
                        axs[i][j].set_xticklabels([])
                    
                    # Ticks
                    if np.max(np.abs(sort_obs_amps[k])) > 0.2:
                        axs[i][j].yaxis.set_major_locator(MultipleLocator(0.1))
                    else:
                        axs[i][j].yaxis.set_major_locator(MultipleLocator(0.05))
                    k += 1

        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.72,0.25), framealpha=None, frameon=False)
        fig.delaxes(axs[2][1])
        fig.delaxes(axs[2][2])
        fig.suptitle('Waveform Comparison', fontsize=12, y=0.98)
        fig.supylabel(f'Amplitude ({units})',fontsize=12)
        fig.supxlabel('UTC Time(hr:min:sec)',fontsize=12)
        fig.text(0.72, 0.2, (r"$\bf{" + 'Project:' + "}$" + '' + project), size=10, horizontalalignment='left')
        fig.text(0.72, 0.175, (r'$\bf{' + 'Run:' + '}$' + '' + run), size=10, horizontalalignment='left')
        fig.text(0.72, 0.15, (r'$\bf{' + 'DataType:' '}$' + '' + data_type), size=10, horizontalalignment='left')
        plt.subplots_adjust(left=0.11, bottom=0.09, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
         
        if not path.exists(f'{plot_dir}/comparison/wf/{data_type}'):
            makedirs(f'{plot_dir}/comparison/wf/{data_type}')
      
        plt.savefig(f'{plot_dir}/comparison/wf/{data_type}/{run}_{data_type}_{component}.png', dpi=300)
        plt.close()
        
        
    else:
        if component != 'avg':
            label = f'HN{component}'
        else:
            label = 'rotd50'
        # Set up figure
        fig, axs = plt.subplots(dim[0],dim[1],figsize=(10,9.5))
        k = 0 
        for i in range(dim[0]):
            for j in range(dim[1]):
                if k+1 <= len(stn_list):
                    axs[i][j].plot(sort_syn_times[k],sort_syn_amps[k],
                                     color='C1',alpha=0.7,lw=0.4,label='synthetic')
                    axs[i][j].plot(sort_obs_times[k],sort_obs_amps[k],
                                    'steelblue',alpha=0.7,lw=0.4,label='observed')
                    axs[i][j].text(0.98,0.9,label,horizontalalignment='right',
                                   transform=axs[i][j].transAxes,size=10)
                    axs[i][j].text(0.98,5E-2,f'Hypdist={int(sort_hypdists[k])}km',horizontalalignment='right',
                                    transform=axs[i][j].transAxes,size=10)
                    axs[i][j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    # axs[i][j].set_xlim(start, end)
                    axs[i][j].set_title(sort_stn_name[k],fontsize=10)
                    axs[i][j].tick_params(axis='both', which='major', labelsize=10)
                    if i < dim[0]-1:
                        axs[i][j].set_xticklabels([])
                    if i == dim[0]-2 and j == 0:
                        axs[i][j].set_xticklabels([])
                    k += 1
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.suptitle('Waveform Comparison', fontsize=12, y=0.98)
        fig.supylabel(f'Amplitude ({units})',fontsize=12)
        fig.supxlabel('UTC Time(hr:min:sec)',fontsize=12,y=0.125)
        fig.legend(handles, labels, loc=(0.72,0.1), framealpha=None, frameon=False, fontsize=10)
        fig.text(0.72, 0.075, (r"$\bf{" + 'Project:' + "}$" + '' + project), size=10, horizontalalignment='left')
        fig.text(0.72, 0.05, (r'$\bf{' + 'Run:' + '}$' + '' + run), size=10, horizontalalignment='left')
        fig.text(0.72, 0.025, (r'$\bf{' + 'DataType:' '}$' + '' + data_type), size=10, horizontalalignment='left')
        plt.subplots_adjust(left=0.11, bottom=0.2, right=0.95, top=0.925, wspace=0.2, hspace=0.2)
        
        if not path.exists(f'{plot_dir}/comparison/wf/{data_type}'):
            makedirs(f'{plot_dir}/comparison/wf/{data_type}')
        
        plt.savefig(f'{plot_dir}/comparison/wf/{data_type}/{run}_{data_type}_{component}.png', dpi=300)
        plt.close()
        


    