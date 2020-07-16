#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:50:59 2018

@author: vjs
"""

############  Module for functions for Tsunami Earthquakes Project ############
# VJS 10/2018


def compute_rhyp(stlon,stlat,stelv,hypolon,hypolat,hypodepth):
    '''
    Compute the hypocentral distance for a given station lon, lat and event hypo
    Input:
        stlon:          Float with the station/site longitude
        stlat:          Float with the station/site latitude
        stelv:          Float with the station/site elevation (km)
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


###############################################################################
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

###############################################################################
def compute_geometric_mean(corrected_timeseries_list):
    '''
    THIS IS WRONG!!
    Compute the geometric mean of several time series. This is the geometric 
    mean of each discrete point in teh time series, resulting in a new time 
    series. They must be different components of the same instrument, so they 
    must have the same time stamp per time series.
    Input:
        corrected_timeseries_list:          A list of time series [ts1,ts2,ts3,...]
                                                that have been gain and baseline
                                                corrected.
    Output:
        timeseries_geommean:                A new stream object with the averaged
                                                time series.
    '''
    
    ## Get the number of time series to avreage:
    num_timeseries = len(corrected_timeseries_list)
    
    ## Multiply the time series element by element:
    for seriesi in range(num_timeseries):
        if seriesi == 0:
            product_series = corrected_timeseries_list[seriesi][0].data
        else:
            product_series = product_series * corrected_timeseries_list[seriesi][0].data
    
    ## Get the geometric mean, which is now the root of these:
    gmean_data = product_series ** (1./num_timeseries)
    
    ## Add these into a new series - take the first int eh list, and replace the times:
    timeseries_geommean = corrected_timeseries_list[0].copy()
    timeseries_geommean[0].data = gmean_data
    
    ## return:
    return timeseries_geommean
    


###############################################################################
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
    
###############################################################################
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


###############################################################################
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


###############################################################################
def accel_to_veloc(acc_timeseries_stream):
    '''
    Integrate an acceleration time series to get velocity
    Input:
        acc_timeseries_stream:              Obspy stream object with a time series
                                                of acceleration, units in m/s/s
                                                baseline and gain corrected.
    Output:
        vel_timeseries_stream:          
    '''
    
    from scipy.integrate import cumtrapz
    
    ## Get the bsaeline corrected and gain corrected time series:
    acc_amplitude = acc_timeseries_stream[0].data
    
    ## And times:
    acc_times = acc_timeseries_stream[0].times()
    
    ## INtegrate the acceration time series to get velocity:
    vel_amplitude = cumtrapz(acc_amplitude,x=acc_times)
    
    ## Make a copy of the old stream object:
    vel_timeseries_stream = acc_timeseries_stream.copy()
    
    ## Put the integrated velocity in there in place of accel:
    vel_timeseries_stream[0].data = vel_amplitude
    
    ## Return:
    return vel_timeseries_stream
    
    
###############################################################################
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

###############################################################################
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


###############################################################################
def arias_intensity(timeseries_stream,t_start,t_end,dt):
    '''
    Compute the arias intensity (AI) for the given stream object
    Input:
        timeseries_stream:              Obspy stream object with an accel. time series 
                                            on which to compute the intensity, 
                                            gain and baseline corrected.
        t_start:                        Start time (seconds) for AI
        t_end:                          End time (seconds) for AI
        dt:                             Time interval for this acceleration record
    Output:
        arias_intensity:                Float with the arias intensity
    '''        
    import numpy as np
    from scipy.integrate import trapz
    
    ## Get the times - this should give time in seconds for the waveform:
    times = timeseries_stream[0].times()
    
    ## Get the acceleration time series
    timeseries = timeseries_stream[0].data
    
    ## Get the square of the acceleration time series:
    accel_squared = timeseries**2
    
    ## Integrate this (sum it) over the start and end time.
    ## Get the index for the start time:
    t_start_index = np.where(times == t_start)[0][0]
    t_end_index = np.where(times == t_end)[0][0]

    
    ## If tstart and tend are the same, then it's because the duration fucntion
    ##   identified that there are no amplitudes above the threshold. 
    ##   In this case, set accel_squared_sum to 0.
    if t_start_index == t_end_index == 0:
        accel_squared_interval = 0
    else:
        accel_squared_interval = trapz(accel_squared[t_start_index:t_end_index],x=times[t_start_index:t_end_index],dx=dt)
    
    ## Arias intensity is 1/2g times this:
    g = 9.81
    arias_intensity = (1./(2*g)) * accel_squared_interval
    
    return arias_intensity

## Original:

#    import numpy as np
#    
#    ## Get the times - this should give time in seconds for the waveform:
#    times = timeseries_stream[0].times()
#    
#    ## Get the acceleration time series
#    timeseries = timeseries_stream[0].data
#    
#    ## Get the square of the acceleration time series:
#    accel_squared = timeseries**2
#    
#    ## Integrate this (sum it) over the start and end time.
#    ## Get the index for the start time:
#    t_start_index = np.where(times == t_start)[0][0]
#    t_end_index = np.where(times == t_end)[0][0]
#
#    
#    ## If tstart and tend are the same, then it's because the duration fucntion
#    ##   identified that there are no amplitudes above the threshold. 
#    ##   In this case, set accel_squared_sum to 0.
#    if t_start_index == t_end_index == 0:
#        accel_squared_sum = 0
#    else:
#        accel_squared_sum = np.sum(accel_squared[t_start_index:t_end_index])
#    
#    ## Arias intensity is 1/2g times this:
#    g = 9.81
#    arias_intensity = (1./(2*g)) * accel_squared_sum
#    
#    return arias_intensity


###############################################################################
def cav(timeseries_stream,t_start,t_end,dt):
    '''
    Compute the CAV (cumulative absolute velocity) engineering parameter for 
    a given stream object.
    Input:
        timeseries_stream:              Obspy stream object with an accel. time 
                                            series (m/s/s)on which to compute 
                                            the CAV. Gain, and baseline 
                                            corrected, maybe filtered.
        t_start:                        Start time (seconds) for CAV
        t_end:                          End time (seconds) for CAV
        dt:                             Time interval for this accel. reecord
    Output:
        cav:                            Float with the cumulative absolute velocity
    '''
    
    import numpy as np
    from scipy.integrate import trapz
    
    ## Get the times - this should give time in seconds for the waveform:
    times = timeseries_stream[0].times()
    
    ## Get tjhe time series of absolute value of the acceleration:
    accel_abs = np.abs(timeseries_stream[0].data)

    ## Integrate this (sum it) over the start and end time.
    ## Get the index for the start time:
    t_start_index = np.where(times == t_start)[0][0]
    t_end_index = np.where(times == t_end)[0][0]
    
    ## If tstart and tend are the same, then it's because the duration fucntion
    ##   identified that there are no amplitudes above the threshold. 
    ##   In this case, set accel_squared_sum to 0.
    if t_start_index == t_end_index == 0:
        accel_abs_interval = 0
    else:
        accel_abs_interval = trapz(accel_abs[t_start_index:t_end_index],x=times[t_start_index:t_end_index],dx=dt)
    
    ## CAV is equal to this:
    cav = accel_abs_interval
    
    return cav

###############################################################################
def get_peak_value(timeseries_stream):
    '''
    Get the peak value of an acceleration (PGA) or velocity (PGV) time series.
    Input:
        timeseries_stream:              Obspy stream object with the time series
    Output:
        peak_value:                     Peak value (PGA or PGV)
    '''
    
    import numpy as np

    ## Get the data:
    data = timeseries_stream[0].data
    
    ## Get the peak:
    peak_value = np.max(np.abs(data))
    
    ## Return:
    return peak_value

###############################################################################
def get_horiz_geom_avg(peakvalue_E,peakvalue_N):
    '''
    Get the geometric average of the peak value of a time series for two 
    horizontal components 
    Input:
        peakvalue_E:            Float with the peak value (PGA or PGV) E component
        peakvalue_N:            Float with the peak value (PGA or PGV) N component
    Output:
        peakvalue_horiz:        Float with the geometrically average peak value 
                                    of the two horizontal components
    '''
    import numpy as np
    
    peakvalue_horiz = 10**((np.log10(peakvalue_E) + np.log10(peakvalue_N))/2.)

    ## REturn:
    return peakvalue_horiz

###############################################################################
def get_geom_avg(E_record, N_record):
    '''
    Get the geometric average of the two components of a record
    Input:
        E_record:            Array of full record East-West component
        N_record:            Array of full record North-South component
    Output:
        geom_avg:            Array of full record geometric average
    ''' 
    import numpy as np
    geom_avg = np.sqrt(E_record * N_record)

    ## REturn:
    return geom_avg


    
###############################################################################
#################           Some PLotting Stuff         #######################
###############################################################################

def plot_5comparison(y_value_looplist,x_value,colorby_value,subplot_titles,colormapname,clims,xlims,ylims,xlabel,ylabel,clabel,figuresize,xscale='linear',yscale='linear',index_select=[],symbol_select='*',selectsize=160,selectlinecolor='yellow'):
    '''
    Make a plot of comparison between two parameters for each of the 5 components
    (E, N, Z, horizontal, all 3), for the same x value. 
    Input:
        y_value_looplist:           The list of things to loop through and plot on the y axis, each an array
        x_value:                    Array with the x values to plot
        colorby_value:              Array with the values to color each point by
        subplot_titles:             Array with the titles for each of the 5 subplots (strings)
        colormapname:               String with the name of the colormap to use for the plots
        clims:                      List with the minimum and maximu value for coloring points [cmin,cmax,cinterval]
        xlims:                      List with the x axis limits [xmin,xmax]
        ylims:                      List with the y axis limits [ymin,ymax]
        xlabel:                     String with the x axis label
        ylabel:                     STring with the y axis label
        clabel:                     String with the colorbar label
        figuresize:                 Figure size in inches (xwidth,yheight)
        index_select:               Array with the indices to plot select points with a different symbol
        symbol_select:              String with the symbol to plot select points
    '''
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    
    fig,axes = plt.subplots(nrows=2,ncols=3,figsize=figuresize)
    axes = axes.reshape(1,6)[0]
    
    #Make colormap:
    colormap=plt.get_cmap(colormapname)
    #Make a normalized colorscale
    cNorm=colors.Normalize(vmin=clims[0], vmax=clims[1])
    #Apply normalization to colormap:
    scalarMap=plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
    
    #Make a fake contour plot for the colorbar:
    Z=[[0,0],[0,0]]
    levels=np.arange(clims[0],clims[1],clims[2])
    c=plt.contourf(Z, levels, cmap=colormap)
    
    colorVal=scalarMap.to_rgba(colorby_value)
    
    
    for component_i in range(len(y_value_looplist)):
        y_i = y_value_looplist[component_i]
    
        axis_i = axes[component_i].scatter(x_value,y_i,facecolor=colorVal,edgecolor='black',marker='o',linewidths=1)
        
        if len(index_select) > 0:
            axes[component_i].scatter(x_value[index_select],y_i[index_select],facecolor=colorVal[index_select],edgecolor=selectlinecolor,marker=symbol_select,linewidths=1,s=selectsize)
    
        axes[component_i].set_yscale(yscale)
        axes[component_i].set_xscale(xscale)
        axes[component_i].set_xlim(xlims)
        axes[component_i].set_ylim(ylims)
        axes[component_i].set_xlabel(xlabel)
        axes[component_i].set_ylabel(ylabel)
        axes[component_i].set_title(subplot_titles[component_i])
    
    plt.delaxes(axes[-1])
    
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    cbaxis = plt.axes([0.7,0.0005,0.2,0.4])
    cbaxis.axis('off')
    cb = plt.colorbar(c,ax=cbaxis,orientation='horizontal',fraction=0.9)
    cb.set_label(clabel)
    
    ## Return figure:
    return fig



##############################################################################
def read_scardec(scardec_file):
    '''
    Read in a SCARDEC file, and output the time and source time function.
    (Moment rate)
    Input:
        scardec_file:           Path to the SCARDEC STF file
    Output:
        times:                  Array with time 
        momentrate:             Array with Moment rate in time
        dt:                     Float with the time interval
    '''
    
    import numpy as np
    
    ## Open the file with genfromtxt. The first two lines ar headers
    
    scardec_data = np.genfromtxt(scardec_file,skip_header=2)
    
    times = scardec_data[:,0]
    momentrate = scardec_data[:,1]
    
    ## Check that the time increments are all the same:
    if len(np.unique(np.diff(times))) == 1:
        dt = np.diff(times)[0]
    else:
        print('Different time intervals for file %s, exiting' % scardec_file)
        print(np.unique(np.diff(times)))
        
    return times, momentrate, dt
    

##############################################################################
def zero_pad(times,data,dt,location_flag,numzeros):
    '''
    Zero pad a time series, before or after.
    Input:
        times:              Array with the time (s)
        data:               Array with the data to zero pad, same length as time
        dt:                 Float with the time sampling interval
        location_flag:      String with location to zero pad: "before" or "after"
        numzeros:           Float with number of zeros to use in padding the time series
    Output:
        zeropad_time:       Array with the new times for zero padding
        zeropad_data:       Array with the new data, zero padded
    '''
    
    import numpy as np
    
    if location_flag == 'before':
        # Get the values of time to add on to the array, and the data:
        padon_times = np.arange(-1*numzeros,0)*dt + times[0]
        padon_data = np.zeros(numzeros)
        
        # Concatenate these before the existing data
        zeropad_time = np.r_[padon_times,times]
        zeropad_data = np.r_[padon_data,data]
        
    elif location_flag == 'after':
        # Get the values of time to add on to the array, and the zero pad data:
        padon_times = times[-1] + np.arange(1,numzeros+1)
        padon_data = np.zeros(numzeros)
        
        # Concatenate these after the existing data
        zeropad_time = np.r_[times,padon_times]
        zeropad_data = np.r_[data,padon_data]
        
    # Return
    return zeropad_time, zeropad_data
    
    
##############################################################################
def integrate_momentrate(times,momentrate):
    '''
    Integrate a moment rate function from SCARDEC to get moment as a function of time
    Input:
        times:              Array with time (seconds)
        momentrate:         Array with moment rate (Nm/s)
    Output:
        moment:             Array of moment as a function of time
    '''
    
    from scipy.integrate import cumtrapz
        
    ## Integrate, with time as the x value:
    moment = cumtrapz(momentrate,x=times,initial=0)
    
    ## Return:
    return moment
    

#############################################################################
def stf_from_scardec(times,moment,dt,time_bandwidth,ntapers):
    '''
    Get the fourier transform of the SCARDEC Source time function, to obtain
    the source spectrum (I think that's what this is).
    Input:
        times:              Array with the time of the STF
        moment:             Array with the moment of the STF
        dt:                 Float with the time sampling interval
        time_bandwidth:     Float with the time-bandwidth product. Common values: 2, 3, 4 and numbers in between.
        ntapers:            Integer with the number of tapers to use. Defaults to int(2*time_bandwidth) - 1. This is maximum senseful amount.
    Output:
        frequencies:        Array with the frequency
        powerspectrum:      Array with the power spectrum
    '''
    
    from mtspec import mtspec
    
    
#    [ampspectrum,frequencies] = mtspec(moment,dt,time_bandwidth,nfft=nfft,number_of_tapers=ntapers)
    [powerspectrum,frequencies] = mtspec(moment,dt,time_bandwidth,number_of_tapers=ntapers)
    
    return frequencies,powerspectrum
        
    
    
    
    