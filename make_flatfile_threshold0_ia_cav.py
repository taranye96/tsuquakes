#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:07:05 2018

@author: vjs
"""

##### Main script for making the dataset for tsunami earthquakes project #####

### Versin of the databse with arais acceleration = 0, to catch all for PGA/PGV ###

import numpy as np
import obspy
import pandas as pd
from glob import glob
import tsueqs_main_fns as tmf
import matplotlib.pyplot as plt

## Set paths and parameters:
project_directory = '/Users/vjs/tsueqs'                         # Directory for project
eq_table_path = project_directory + '/data/events.csv'          # Path for events metadata table

arias_acc_threshold = 0.0                                       # Value of threshold (m/s/s) 
                                                                #   to use for computing duration/Arias Intensity
                                                                
## Parameters for velocity filtering:
fcorner = 1/15.                                                 # Frequency at which to high pass filter
order = 2                                                       # Number of poles for filter

# Set data directory base:
data_directory_base = project_directory + '/data'

## Output directory for plots:
plot_directory = project_directory + '/flatfile_plots/threshold_0cm_ia_cav/'

## Output directory for file:
flatfile_path = project_directory + '/flatfiles/threshold_0cm_ia_cav.csv'

## Plot parameters:
plot_nrows = 5          ## Number of rows per waveform plot
plot_ncols = 2          ## Number of columns per waveform plot


###############################################################################
## Get all the earthquake subdirectories for acceleration, and displacement:
all_data_directories = np.array(glob(data_directory_base + '/*/'))

## Read in the main earthquakes dataset, use the first number column as the index column:
eq_table = pd.read_csv(eq_table_path)


## Start the columns that will be in the final flatfile:
eqnumber = np.array([])
eventname = np.array([])
country = np.array([])
origintime = np.array([])
hypolon = np.array([])
hypolat = np.array([])
hypodepth = np.array([])
mw = np.array([])
m0 = np.array([])
network = np.array([])
station = np.array([])
stlon = np.array([])
stlat = np.array([])
stelev = np.array([])
instrumentcode = np.array([])
nostations = np.array([])
mechanism = np.array([])
rhypo = np.array([])
duration_e = np.array([])
duration_n = np.array([])
duration_z = np.array([])
duration_horiz = np.array([])
duration_3comp = np.array([])
risetime = np.array([])
#t2pgd = np.array([])
arias_e = np.array([])
arias_n = np.array([])
arias_z = np.array([])
arias_horiz = np.array([])
arias_3comp = np.array([])
cav_e = np.array([])
cav_n = np.array([])
cav_z = np.array([])
cav_horiz = np.array([])
cav_3comp = np.array([])
pga = np.array([])
pgv = np.array([])



## For every earthquake, find the data directory for that earthquake, and 
##   make a glob of every ACCELERATION file in that dataset:
for earthquakei in range(len(eq_table['EQnumber'])):
    earthquake_name_i = eq_table['Event Name'][earthquakei]
    print('Working on %s' % earthquake_name_i)
    
    data_directory_i = data_directory_base + '/' + earthquake_name_i
    
    ## Get the earthquake metadata for the table:
    eqnumber_i = eq_table['EQnumber'][earthquakei]
    eventname_i = eq_table['Event Name'][earthquakei]
    country_i = eq_table['Country'][earthquakei]
    origintime_i = eq_table['Origin Time (UTC)*'][earthquakei]
    hypolon_i = eq_table['Longitude'][earthquakei]
    hypolat_i = eq_table['Latitude'][earthquakei]
    hypodepth_i = eq_table['Depth (km)'][earthquakei]
    mw_i = eq_table['Mw'][earthquakei]
    m0_i = 10**(mw_i*(3/2.) + 9.1)
    nostations_i = eq_table['No. Sta'][earthquakei]
    mechanism_i = eq_table['Mechanism'][earthquakei]
    
    
    ## Get an array of all the channels in this directory, with an N or L instrument
    ##    code, and an E,N,or Z direction:
    acc_chan_earthquakei = np.array(glob(data_directory_i + '/accel/*.H[N,L][E,N,Z]*'))
    
    ## Get the gain and units from the chan file in this directory:
    accmetadata_file_earthquakei = data_directory_i + '/' + earthquake_name_i + '_sm.chan'
    accmetadata_chan_earthquakei = pd.read_csv(accmetadata_file_earthquakei,sep='\t',header=0,names=['net','sta','loc','chan','lat','lon','elev','samplerate','gain','units'])
    
    ## This might result in white spaces in the station name, so remove those:
    accmetadata_chan_earthquakei.sta = accmetadata_chan_earthquakei.sta.astype(str)
    accmetadata_chan_earthquakei.sta = accmetadata_chan_earthquakei.sta.str.replace(' ','')
    
    acc_gain_earthquakei = accmetadata_chan_earthquakei['gain'][0]
    acc_units_earthquakei = accmetadata_chan_earthquakei['units'][0]
    
    
    ## Set a station array to loop over for channels:
    stationname_counter = []
    chancounter = []
    mseedcounter = []
    
    stationname_list_i = []
    channel_list_i = []
    mseed_list_i = []

    
    ## Pull out the stations and corresponding channel components for them, from
    ##    the file list. 
    ## For every acceleration station/channel in the earthquake:
    for channelj in range(len(acc_chan_earthquakei)):
        
        ## Get the channel code and station name:
        station_name_j = acc_chan_earthquakei[channelj].split('.')[0].split('/')[-1]
        channelcode_name_j = acc_chan_earthquakei[channelj].split('.')[1]
        
        ## Append this station to the station counter:
        stationname_counter.append(station_name_j)
        
        ## If this station is the first station, append the component to the counter:
        if channelj == 0:
            chancounter.append(channelcode_name_j)
            mseedcounter.append(acc_chan_earthquakei[channelj])
            
        ## Otherwise, check to see if this station is the same as the last...
        else:
            ## If this is the same as the last staion, it's a new channel/component,
            ##    so add the channel/component on to the counter, and miniseed file path:
            if station_name_j == stationname_counter[channelj - 1]:
                chancounter.append(channelcode_name_j)
                mseedcounter.append(acc_chan_earthquakei[channelj])
            
            ## If this is a different station than the last... 
            else:
                ## Then add the information of the last station to the 
                ##    earthquake's station, channel, and mseed lists:
                stationname_list_i.append(stationname_counter[channelj - 1])
                channel_list_i.append(chancounter)
                mseed_list_i.append(mseedcounter)
                
                ## And then set the counters back to empty:
                chancounter = []
                mseedcounter = []
                
                ## And re-append with the current station to start over for this station:
                chancounter.append(channelcode_name_j)
                mseedcounter.append(acc_chan_earthquakei[channelj])

            
    ## Start a counter for the number of stations:
    nostations_i = 0
    
    ## Start a plot:
    plot_i,axes_i = plt.subplots(nrows=plot_nrows,ncols=plot_ncols,figsize=(12,6))
    plot_counter = 0
    plot_names_counter = []        
    
    ## Now loop over the stations for this earthquake, and start to run the computations:
    for stationj in range(len(stationname_list_i)):
        station_name_ij = stationname_list_i[stationj]
        
        ## First get the components for this station (E, N, Z, etc.):
        components_j = []
        for comp_k in range(len(channel_list_i[stationj])):
            components_j.append(channel_list_i[stationj][comp_k][2])
            
        ## Also get the type of instrument - want HL or HN:
        instruments_j = []
        for comp_k in range(len(channel_list_i[stationj])):
            instruments_j.append(channel_list_i[stationj][comp_k][0:2])


        ## If there is an E, N, or Z, and if there is an HL or HN instrument code, 
        ##    add this station's metadata to the dataframe arrays:
        if (('E' in components_j) or ('N' in components_j) or ('Z' in components_j)) and (('HL' in instruments_j) or ('HN' in instruments_j)):            
            
            ## Get the metadata for this station from the chan file - put it into
            ##     a new dataframe, and also reset the index so it starts at 0:
            if country_i == 'Japan':
                station_ij_metadata = accmetadata_chan_earthquakei[(accmetadata_chan_earthquakei.net == station_name_ij[0:2]) & (accmetadata_chan_earthquakei.sta == station_name_ij[2:])].reset_index(drop=True)
                
            else:
                station_ij_metadata = accmetadata_chan_earthquakei[accmetadata_chan_earthquakei.sta == station_name_ij].reset_index(drop=True)
                
            
            ## If there is something in here, it means this station was found in the chan file.
            ##    So now start to pull out data - also add to the plot name
            ##    counter for the plot file name.
            plot_names_counter.append(station_name_ij)
        
            
            if len(station_ij_metadata > 0):
                
                ## Pull out the data. Take the first row of the subset dataframe, 
                ##    assuming that the gain, etc. is always the same:
                stnetwork_ij = station_ij_metadata.loc[0].net
                stlon_ij = station_ij_metadata.loc[0].lon
                stlat_ij = station_ij_metadata.loc[0].lat
                stelev_ij = station_ij_metadata.loc[0].elev
                stsamprate_ij = station_ij_metadata.loc[0].samplerate
                stgain_ij = station_ij_metadata.loc[0].gain
                
                ## Add to the number of statoins counter, because it means this 
                ##   is a station with contributing data:
                nostations_i+=1
                
                ## If the length of instruments_j or comonents_j is 3 or less, then assume
                ##   they're all the same instrument but 3 components. Then take the
                ##   first one, and also get the instrument code from instruments_j:
                if (len(instruments_j) <= 3) & (len(np.unique(instruments_j))) == 1:
                    instrumentcode_ij = instruments_j[0]
                else:
                    print('WARNING - there are more than 3 channels for this station or not all are the same!!!')
                    print('Event %s, Station %s' % (earthquake_name_i,station_name_ij))
                    print(channel_list_i[stationj])
                
                
                ## Compute the hypocentral distance:
                rhypo_ij = tmf.compute_rhyp(stlon_ij,stlat_ij,stelev_ij,hypolon_i,hypolat_i,hypodepth_i)
                
                ## Append the earthquake and station info line by line for this station:
                eqnumber = np.append(eqnumber,eqnumber_i)
                eventname = np.append(eventname,eventname_i)
                country = np.append(country,country_i)
                origintime = np.append(origintime,origintime_i)
                hypolon = np.append(hypolon,hypolon_i)
                hypolat = np.append(hypolat,hypolat_i)
                hypodepth = np.append(hypodepth,hypodepth_i)
                mw = np.append(mw,mw_i)
                m0 = np.append(m0,m0_i)
                mechanism = np.append(mechanism,mechanism_i)
                network = np.append(network,stnetwork_ij)
                station = np.append(station,station_name_ij)
                stlon = np.append(stlon,stlon_ij)
                stlat = np.append(stlat,stlat_ij)
                stelev = np.append(stelev,stelev_ij)
                rhypo = np.append(rhypo,rhypo_ij)
                instrumentcode = np.append(instrumentcode,instrumentcode_ij)
                
                
            
                ## Now start the computations....          
                
                    
                ## Get the values for the E component, if it exists:
                if 'E' in components_j:
                    ## Read it into a stream object
                    e_mseedcounter_index = np.where('E' in components_j)[0][0]
                    e_stream_channel_raw_ij = obspy.read(mseed_list_i[stationj][e_mseedcounter_index])
                    
                    ## Correct by gain, so everything is in meters:
                    e_stream_channel_gaincorr_ij = tmf.correct_for_gain(e_stream_channel_raw_ij,stgain_ij)
                    
                    ## Get the pre-event baseline:
                    e_acc_baseline_ij = tmf.compute_baseline(e_stream_channel_gaincorr_ij)
                    
                    ## Get the baseline corrected stream object:
                    e_streamunfilt_channel_ij = tmf.correct_for_baseline(e_stream_channel_gaincorr_ij,e_acc_baseline_ij)
                    
                    ## High pass filter at fcorner specified above:
                    e_stream_channel_ij = tmf.highpass(e_streamunfilt_channel_ij,fcorner,stsamprate_ij,order,zerophase=True)
                    
                    ## Get the duration, stream file time of start, and time of stop of shaking:
                    e_Td_ij, e_t_start_ij, e_t_end_ij = tmf.determine_Td(arias_acc_threshold,e_stream_channel_ij)
                    

                    ## Compute the Arias Intensity for this channel only:
                    e_arias_intensity_ij = tmf.arias_intensity(e_stream_channel_ij,e_t_start_ij,e_t_end_ij,stsamprate_ij)
                    
                    ## Compute the CAV:
                    e_cav_ij = tmf.cav(e_stream_channel_ij,e_t_start_ij,e_t_end_ij,stsamprate_ij)
                    
                    ## Append to the overall arrays:
                    arias_e = np.append(arias_e,e_arias_intensity_ij)
                    duration_e = np.append(duration_e,e_Td_ij)
                    cav_e = np.append(cav_e,e_cav_ij)
                    
                    ## Plot it:
                    min_y = np.min(e_stream_channel_ij[0].data)
                    max_y = np.max(e_stream_channel_ij[0].data)
                    axes_i[0][plot_counter].fill_between(np.array([e_t_start_ij,e_t_end_ij]),min_y,max_y,color='gray',alpha=0.5)
                    axes_i[0][plot_counter].plot(e_stream_channel_ij[0].times(),e_stream_channel_ij[0].data,color='black',label='E')
                    axes_i[0][plot_counter].axhline(arias_acc_threshold,xmin=0,xmax=np.max(e_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[0][plot_counter].axhline(-1*arias_acc_threshold,xmin=0,xmax=np.max(e_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[0][plot_counter].set_xlabel('Time (s)')
                    axes_i[0][plot_counter].set_ylabel('Accel. (m/s/s)')
                    axes_i[0][plot_counter].legend()
                    axes_i[0][plot_counter].set_title(station_name_ij + ' ' + np.str(rhypo_ij) + ' km')                   
                    
                    
                else:
                    ## Append to the overall arrays:
                    arias_e = np.append(arias_e,float('nan'))
                    duration_e = np.append(duration_e,float('nan'))
                    cav_e = np.append(cav_e,float('nan'))
                    axes_i[plot_counter][0].set_title(station_name_ij)
                    
                    
                ## Get the values for the N component, if it exists:
                if 'N' in components_j:
                    ## Read it into a stream object
                    n_mseedcounter_index = np.where('N' in components_j)[0][0]
                    n_stream_channel_raw_ij = obspy.read(mseed_list_i[stationj][n_mseedcounter_index])
                    
                    ## Correct by gain, so everything is in meters:
                    n_stream_channel_gaincorr_ij = tmf.correct_for_gain(n_stream_channel_raw_ij,stgain_ij)
                    
                    ## Get the pre-event baseline:
                    n_acc_baseline_ij = tmf.compute_baseline(n_stream_channel_gaincorr_ij)
                    
                    ## Get the baseline corrected stream object:
                    n_streamunfilt_channel_ij = tmf.correct_for_baseline(n_stream_channel_gaincorr_ij,n_acc_baseline_ij)
                    
                    ## High pass filter at fcorner specified above:
                    n_stream_channel_ij = tmf.highpass(n_streamunfilt_channel_ij,fcorner,stsamprate_ij,order,zerophase=True)
                    
                    ## Get the duration, stream file time of start, and time of stop of shaking:
                    n_Td_ij, n_t_start_ij, n_t_end_ij = tmf.determine_Td(arias_acc_threshold,n_stream_channel_ij)
                    

                    ## Compute the Arias Intensity for this channel only:
                    n_arias_intensity_ij = tmf.arias_intensity(n_stream_channel_ij,n_t_start_ij,n_t_end_ij,stsamprate_ij)
                    
                    ## Compute the CAV:
                    n_cav_ij = tmf.cav(n_stream_channel_ij,n_t_start_ij,n_t_end_ij,stsamprate_ij)
                    
                    ## Append to the overall arrays:
                    arias_n = np.append(arias_n,n_arias_intensity_ij)
                    duration_n = np.append(duration_n,n_Td_ij)
                    cav_n = np.append(cav_n,n_cav_ij)
                    
                    ## Plot it:
                    min_y = np.min(n_stream_channel_ij[0].data)
                    max_y = np.max(n_stream_channel_ij[0].data)
                    axes_i[1][plot_counter].fill_between(np.array([n_t_start_ij,n_t_end_ij]),min_y,max_y,color='gray',alpha=0.5)
                    axes_i[1][plot_counter].plot(n_stream_channel_ij[0].times(),n_stream_channel_ij[0].data,color='black',label='N')
                    axes_i[1][plot_counter].axhline(arias_acc_threshold,xmin=0,xmax=np.max(n_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[1][plot_counter].axhline(-1*arias_acc_threshold,xmin=0,xmax=np.max(n_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[1][plot_counter].set_xlabel('Time (s)')
                    axes_i[1][plot_counter].set_ylabel('Accel. (m/s/s)')
                    axes_i[1][plot_counter].legend()
                    
                    
                else:
                    ## Append to the overall arrays:
                    arias_n = np.append(arias_n,float('nan'))
                    duration_n = np.append(duration_n,float('nan'))
                    cav_n = np.append(cav_n,float('nan'))
                    
                    
                    
                ## Get the values for the Z component, if it exists:
                if 'Z' in components_j:
                    ## Read it into a stream object
                    z_mseedcounter_index = np.where('Z' in components_j)[0][0]
                    z_stream_channel_raw_ij = obspy.read(mseed_list_i[stationj][z_mseedcounter_index])
                    
                    ## Correct by gain, so everything is in meters:
                    z_stream_channel_gaincorr_ij = tmf.correct_for_gain(z_stream_channel_raw_ij,stgain_ij)
                    
                    ## Get the pre-event baseline:
                    z_acc_baseline_ij = tmf.compute_baseline(z_stream_channel_gaincorr_ij)
                    
                    ## Get the baseline corrected stream object:
                    z_streamunfilt_channel_ij = tmf.correct_for_baseline(e_stream_channel_gaincorr_ij,e_acc_baseline_ij)
                    
                    ## High pass filter at fcorner specified above:
                    z_stream_channel_ij = tmf.highpass(z_streamunfilt_channel_ij,fcorner,stsamprate_ij,order,zerophase=True)
                    
                    ## Get the duration, stream file time of start, and time of stop of shaking:
                    z_Td_ij, z_t_start_ij, z_t_end_ij = tmf.determine_Td(arias_acc_threshold,z_stream_channel_ij)
                    

                    ## Compute the Arias Intensity for this channel only:
                    z_arias_intensity_ij = tmf.arias_intensity(z_stream_channel_ij,z_t_start_ij,z_t_end_ij,stsamprate_ij)
                    
                    ## Compute the CAV:
                    z_cav_ij = tmf.cav(z_stream_channel_ij,z_t_start_ij,z_t_end_ij,stsamprate_ij)
                    
                    ## Append to the overall arrays:
                    arias_z = np.append(arias_z,z_arias_intensity_ij)
                    duration_z = np.append(duration_z,z_Td_ij)
                    cav_z = np.append(cav_z,z_cav_ij)
                    
                    ## Plot it:
                    min_y = np.min(z_stream_channel_ij[0].data)
                    max_y = np.max(z_stream_channel_ij[0].data)
                    axes_i[2][plot_counter].fill_between(np.array([z_t_start_ij,z_t_end_ij]),min_y,max_y,color='gray',alpha=0.5)
                    axes_i[2][plot_counter].plot(z_stream_channel_ij[0].times(),z_stream_channel_ij[0].data,color='black',label='Z')
                    axes_i[2][plot_counter].axhline(arias_acc_threshold,xmin=0,xmax=np.max(z_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[2][plot_counter].axhline(-1*arias_acc_threshold,xmin=0,xmax=np.max(z_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[2][plot_counter].set_xlabel('Time (s)')
                    axes_i[2][plot_counter].set_ylabel('Accel. (m/s/s)')
                    axes_i[2][plot_counter].legend()
                    
                else:
                    ## Append to the overall arrays:
                    arias_z = np.append(arias_z,float('nan'))
                    duration_z = np.append(duration_z,float('nan'))
                    cav_z = np.append(cav_z,float('nan'))
                    
                
                ## Get the values for the horizontal, if both E and N are there:
                if ('E' in components_j) and ('N' in components_j):
                    
                    ## Take the minimum time of E and N start times to be the start:
                    en_t_start_ij = np.min([e_t_start_ij,n_t_start_ij])
                    
                    ## And the maximum of end times to be the end:
                    en_t_end_ij = np.max([e_t_end_ij,n_t_end_ij])
                    
                    ## Get the duration to be the time between these:
                    en_Td_ij = en_t_end_ij - en_t_start_ij
                    
                    ## Compute the Arias intensity from these:
                    en_arias_intensity_ij = np.max([e_arias_intensity_ij,n_arias_intensity_ij])
                    
                    ## And the CAV from these:
                    en_cav_ij = np.max([e_cav_ij,n_cav_ij])
                    
                    ## compute the peak acceleration:
                    pga_e_ij = tmf.get_peak_value(e_stream_channel_ij)
                    pga_n_ij = tmf.get_peak_value(n_stream_channel_ij)
                    pga_ij = tmf.get_horiz_geom_avg(pga_e_ij,pga_n_ij)
                    
                    ## compute the peak velocity:
                        # integrate the unfiltered acceleration for velocity:
                    e_velocityunfilt_ij = tmf.accel_to_veloc(e_streamunfilt_channel_ij)
                    e_velocity_ij = tmf.highpass(e_velocityunfilt_ij,fcorner,stsamprate_ij,order,zerophase=True)
                    pgv_e_ij = tmf.get_peak_value(e_velocity_ij)
                    
                    n_velocityunfilt_ij = tmf.accel_to_veloc(n_streamunfilt_channel_ij)
                    n_velocity_ij = tmf.highpass(n_velocityunfilt_ij,fcorner,stsamprate_ij,order,zerophase=True)
                    pgv_n_ij = tmf.get_peak_value(n_velocity_ij)
                    
                    pgv_ij = tmf.get_horiz_geom_avg(pgv_e_ij,pgv_n_ij)
                    
                    ## Append to the overall arrays:
                    arias_horiz = np.append(arias_horiz,en_arias_intensity_ij)
                    duration_horiz = np.append(duration_horiz,en_Td_ij)
                    cav_horiz = np.append(cav_horiz,en_cav_ij)
                    pga = np.append(pga,pga_ij)
                    pgv = np.append(pgv,pgv_ij)
                    
                    ## Plot it:
                    min_y = np.min([np.min(e_stream_channel_ij[0].data),np.min(n_stream_channel_ij[0].data)])
                    max_y = np.max([np.max(e_stream_channel_ij[0].data),np.max(n_stream_channel_ij[0].data)])
                    axes_i[3][plot_counter].fill_between(np.array([en_t_start_ij,en_t_end_ij]),min_y,max_y,color='gray',alpha=0.5)
                    axes_i[3][plot_counter].plot(e_stream_channel_ij[0].times(),e_stream_channel_ij[0].data,color='black',label='E')
                    axes_i[3][plot_counter].plot(n_stream_channel_ij[0].times(),n_stream_channel_ij[0].data,color='blue',linestyle='--',label='N')
                    axes_i[3][plot_counter].axhline(arias_acc_threshold,xmin=0,xmax=np.max(e_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[3][plot_counter].axhline(-1*arias_acc_threshold,xmin=0,xmax=np.max(e_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[3][plot_counter].set_xlabel('Time (s)')
                    axes_i[3][plot_counter].set_ylabel('Accel. (m/s/s)')
                    axes_i[3][plot_counter].legend()
                    
                else:
                    ## Append nan to the overall arrays:
                    arias_horiz = np.append(arias_horiz,float('nan'))
                    duration_horiz = np.append(duration_horiz,float('nan'))
                    cav_horiz = np.append(cav_horiz,float('nan'))
                    pga = np.append(pga,float('nan'))
                    pgv = np.append(pgv,float('nan'))
                    
                    
                ## Get the values for all 3 components, if both E, N, and Z are there:
                if ('E' in components_j) and ('N' in components_j) and ('Z' in components_j):
                    
                    ## Take the minimum time of E and N start times to be the start:
                    enz_t_start_ij = np.min([e_t_start_ij,n_t_start_ij,z_t_start_ij])
                    
                    ## And the maximum of end times to be the end:
                    enz_t_end_ij = np.max([e_t_end_ij,n_t_end_ij,z_t_end_ij])
                    
                    ## Get the duration to be the time between these:
                    enz_Td_ij = enz_t_end_ij - enz_t_start_ij
                    
                    ## Compute the Arias intensity from these:
                    enz_arias_intensity_ij = np.max([e_arias_intensity_ij,n_arias_intensity_ij,z_arias_intensity_ij])
                    
                    ## And the CAV:
                    enz_cav_ij = np.max([e_cav_ij,n_cav_ij,z_cav_ij])
                    
                    ## Append to the overall arrays:
                    arias_3comp = np.append(arias_3comp,enz_arias_intensity_ij)
                    duration_3comp = np.append(duration_3comp,enz_Td_ij)
                    cav_3comp = np.append(cav_3comp,enz_cav_ij)
                    
                    ## Plot it:
                    min_y = np.min([np.min(e_stream_channel_ij[0].data),np.min(n_stream_channel_ij[0].data),np.min(z_stream_channel_ij[0].data)])
                    max_y = np.max([np.max(e_stream_channel_ij[0].data),np.max(n_stream_channel_ij[0].data),np.max(z_stream_channel_ij[0].data)])
                    axes_i[4][plot_counter].fill_between(np.array([enz_t_start_ij,enz_t_end_ij]),min_y,max_y,color='gray',alpha=0.5)
                    axes_i[4][plot_counter].plot(e_stream_channel_ij[0].times(),e_stream_channel_ij[0].data,color='black',label='E')
                    axes_i[4][plot_counter].plot(n_stream_channel_ij[0].times(),n_stream_channel_ij[0].data,color='blue',linestyle='--',label='N')
                    axes_i[4][plot_counter].plot(z_stream_channel_ij[0].times(),z_stream_channel_ij[0].data,color='green',linestyle='-.',label='Z')
                    axes_i[4][plot_counter].axhline(arias_acc_threshold,xmin=0,xmax=np.max(e_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[4][plot_counter].axhline(-1*arias_acc_threshold,xmin=0,xmax=np.max(e_stream_channel_ij[0].times()),linestyle='--',alpha=0.5)
                    axes_i[4][plot_counter].set_xlabel('Time (s)')
                    axes_i[4][plot_counter].set_ylabel('Accel. (m/s/s)')
                    axes_i[4][plot_counter].legend()
                    
                else:
                    ## Append nan to the overall arrays:
                    arias_3comp = np.append(arias_3comp,float('nan'))
                    duration_3comp = np.append(duration_3comp,float('nan'))
                    cav_3comp = np.append(cav_3comp,float('nan'))
                    
                    
                ## Then finally, set all the computed E, N, Z, EN, ENZ to nothing:
                e_Td_ij = e_t_start_ij = e_t_end_ij = e_stream_channel_ij = e_arias_intensity_ij = e_cav_ij = float('nan')
                n_Td_ij = n_t_start_ij = n_t_end_ij = n_stream_channel_ij = n_arias_intensity_ij = n_cav_ij = float('nan')
                z_Td_ij = z_t_start_ij = z_t_end_ij = z_stream_channel_ij = z_arias_intensity_ij = z_cav_ij = float('nan')
                en_Td_ij = en_t_start_ij = en_t_end_ij = en_stream_channel_ij = en_arias_intensity_ij = en_cav_ij = float('nan')
                enz_Td_ij = enz_t_start_ij = enz_t_end_ij = enz_stream_channel_ij = enz_arias_intensity_ij = enz_cav_ij = float('nan')
                
                
            
                ## If this is one more than the number of rows and columns, then 
                ##   close the last plot, save it, and make a new subplot:
                if (plot_counter % plot_nrows) == 1:
                    plot_i.savefig(plot_directory + earthquake_name_i + '_' + '_'.join(plot_names_counter) + '_' + np.str(arias_acc_threshold) + '.png')
                    plt.close('all')
                
                    ## and start a new one:
                    plot_i,axes_i = plt.subplots(nrows=plot_nrows,ncols=plot_ncols,figsize=(12,6))
                    plot_counter = 0
                    plot_names_counter = []
                    
                else:
                    ## Add to the plot counter, as this station has desired data:
                    plot_counter+=1
                
                
                
    ## Finalize the number of stations for each entry for this earthquake:
    nostations = np.append(nostations,np.full(nostations_i,nostations_i))
                
                
                    
## Now, put all the final arrays together into a pandas dataframe. First mak a dict:
dataset_dict = {'eqnumber':eqnumber, 'eventname':eventname,'country':country, 'origintime':origintime,
                    'hypolon':hypolon, 'hypolat':hypolat, 'hypodepth':hypodepth, 'mw':mw, 'm0':m0, 'network':network,
                    'station':station, 'stlon':stlon, 'stlat':stlat, 'stelev':stelev, 
                    'instrumentcode':instrumentcode, 'nostations':nostations, 'mechanism':mechanism, 
                    'rhypo':rhypo, 'duration_e':duration_e, 'duration_n':duration_n, 'duration_z':duration_z,
                    'duration_horiz':duration_horiz, 'duration_3comp':duration_3comp, 
                    'arias_e':arias_e, 'arias_n':arias_n, 'arias_z':arias_z, 
                    'arias_horiz':arias_horiz, 'arias_3comp':arias_3comp, 'cav_e':cav_e, 'cav_n':cav_n,
                    'cav_z':cav_z, 'cav_horiz':cav_horiz, 'cav_3comp':cav_3comp, 'pga':pga, 'pgv':pgv}

## Make it a dataframe:
flatfile_df = pd.DataFrame(data=dataset_dict)


## Save to file:
flatfile_df.to_csv(flatfile_path,index=False)

#risetime = np.array([])
##t2pgd = np.array([])

    
            
    
