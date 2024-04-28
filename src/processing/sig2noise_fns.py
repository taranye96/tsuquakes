#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:07:18 2020

@author: tnye
"""

###############################################################################
# Module with functions used to calculate the signal to noise ratio of records.
# These functions are imported and used in sig2noise.py.  
###############################################################################


def get_P_arrival(epi_lon, epi_lat, hypdepth, st_lon, st_lat, p_vel, origintime):
    """
    Calculates the P-wave arrival time at a station.

    Inputs:
        epi_lon(float): Epicentral longitude. 
        epi_lat(float): Epicentral latitude.
        hypdepth(float): Hypocentral depth (km). 
        st_lon(float): Station longitude.
        st_lat(float): Station latitue. 
        p_vel(float): P-wave velocioty (km/s).
        origintime(timestamp): Origin time of event.

    Return:
        p_datetime(datetime): P-wave arrival time in UTC time. 
        
    """
    
    import numpy as np
    import geopy.distance
    import datetime

    # Calcualte epicentral and hypcentral distances (km)
    epdist = geopy.distance.geodesic((epi_lat, epi_lon), (st_lat, st_lon)).km
    hypdist = np.sqrt(hypdepth**2 + epdist**2)

    # Calcualte time in seconds for P-wave to arrive
    p_time = hypdist/p_vel

    # Get P-wave arrival time as a UTC DateTime object
    dp = datetime.timedelta(seconds=p_time)
    p_UTC = origintime+dp
    
    # Convert to datetime.datetime object
    p_datetime = p_UTC.to_pydatetime()

    return(p_datetime)


def get_SNR(record, p_arrival):
    """
    Calculates signal to noise ratio on a record (acc, vel, or disp) using
    variance.
    
    Inputs:
        record(stream): Obspy stream record for acc, vel, or disp.
        p_arrival(matplotlib): P-wave arrival time as a matplotlib date. 

    Return:
        SNR(float): Signal to noise ratio.

    """

    import numpy as np

    # Get trace from stream in file
    tr = record[0]

    # Compute signal
    absolute_difference_function = lambda list_value : abs(list_value - p_arrival)
    sig_start = min(tr.times('matplotlib'), key=absolute_difference_function)

    # Crop data to just include signal 
    sig_data = []
    for i, time in enumerate(tr.times('matplotlib')):
        if time >= sig_start:
            sig_data.append(tr.data[i])
    
    # Crop data to just include noise
    noise_data = []
    for i, time in enumerate(tr.times('matplotlib')):
        if time < sig_start:
            noise_data.append(tr.data[i])

    # Calculate SNR
    SNR = np.var(sig_data)/np.var(noise_data)

    return(SNR)       