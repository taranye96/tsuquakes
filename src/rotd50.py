#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 21:27:29 2022

@author: tnye
"""

###############################################################################
# This module contains a function to calculate the rotd50 of two horizontal 
# components of a timeseries. 
###############################################################################

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
