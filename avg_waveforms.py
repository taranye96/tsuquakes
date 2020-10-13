#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:06:41 2020

@author: tnye
"""

###############################################################################
# Script used to create horizontal-averaged accerlation waveforms.  These
# averaged waveforms were used to calcualte signal to noise (SNR) ratios. 
###############################################################################

# Imports
import numpy as np
import pandas as pd
import obspy
from glob import glob
import tsueqs_main_fns as tmf


earthquake_name = 'Mentawai2010'

### Set paths and parameters ####
# Project directory 
proj_dir = '/Users/tnye/tsuquakes' 

# Table of earthquake data
eq_table_path = '/Users/tnye/tsuquakes/data/events.csv'   
eq_table = pd.read_csv(eq_table_path)

# Data directory                          
data_dir = proj_dir + '/data'

# Path to send flatfiles
flatfile_path = proj_dir + '/flatfiles/pga_pgv.csv'     

# Velocity filterin
fcorner = 1/15.                          # Frequency at which to high pass filter
order = 2                                # Number of poles for filter
arias_acc_threshold = 0.0   


### Get event data ###
eventname = earthquake_name
country = eq_table['Country'][11]
origintime = eq_table['Origin Time (UTC)*'][11]
hyplon = eq_table['Longitude'][11]
hyplat = eq_table['Latitude'][11]
hypdepth = eq_table['Depth (km)'][11]
mw = eq_table['Mw'][11]
m0 = 10**(mw*(3/2.) + 9.1)
nostations_i = eq_table['No. Sta'][11]
mechanism = eq_table['Mechanism'][11]

# Create lists for of the event and station info for the df
eventnames = np.array([])
countries = np.array([])
origintimes = np.array([])
hyplons = np.array([])
hyplats = np.array([])
hypdepths = np.array([])
mws = np.array([])
m0s = np.array([])
mechanisms = np.array([])
networks = np.array([])
stations = np.array([])
stlons = np.array([])
stlats = np.array([])
stelevs = np.array([])
hypdists = np.array([])
instrument_codes = np.array([])
nostations = np.array([])
E_Td_list = np.array([])
N_Td_list = np.array([])
Z_Td_list = np.array([])
horiz_Td_list = np.array([])
comp3_Td_list = np.array([])
pga_list = np.array([])
pgv_list = np.array([])
pgd_list = np.array([])

### Read in and organize acceleration data ###
# Get an array of all channels with instruments with L or N instrument code and
#    an E, N, or Z direction
acc_chan_files = np.array(sorted(glob(data_dir + '/' + earthquake_name +
                                    '/accel/*.H[L,N][E,N,Z]*')))

# Read in acceleration metadata file for the earthquake
accmetadata_file = data_dir + '/' + earthquake_name + '/' + earthquake_name + '_sm.chan'
accmetadata = pd.read_csv(accmetadata_file, sep='\t', header=0,
                          names=['net', 'sta', 'loc', 'chan', 'lat',
                                 'lon', 'elev', 'samplerate', 'gain', 'units'])

# There might be white spaces in the station name, so remove those
accmetadata.sta = accmetadata.sta.astype(str)
accmetadata.sta = accmetadata.sta.str.replace(' ','')

# Obtain gain and units
acc_gain = accmetadata['gain'][0]
acc_units = accmetadata['units'][0]

# Create lists to add station names, channels, and miniseed files to 
stn_name_list = []
channel_list = []
mseed_list = []

# Group all accel files by station
N = 3
stn_files = [acc_chan_files[n:n+N] for n in range(0, len(acc_chan_files), N)]

# Loop over files to get the list of station names, channels, and mseed files 
for station in stn_files:
    components = []
    mseeds = []

    stn_name = station[0].split('.')[0].split('/')[-1]
    stn_name_list.append(stn_name)
    
    for channel in station:
        channel_code = channel.split('.')[1]

        components.append(channel_code)
        mseeds.append(channel)

    channel_list.append(components)
    mseed_list.append(mseeds)


# Start a counter for the number of stations:
nostations_i = 0 

# Loop over the stations for this earthquake, and start to run the computations:
for i, station in enumerate(stn_name_list):
    stn_name = stn_name_list[i]
    
    # Get the components for this station (E, N, and Z):
    components = []
    for comp in range(len(channel_list[i])):
        components.append(channel_list[i][comp][2])
        
    # Get the type of instrument (HL or HN):
    instruments = []
    for comp in range(len(channel_list[i])):
        instruments.append(channel_list[i][comp][0:2])          
        
        # Get the metadata for this station from the chan file - put it into
        #     a new dataframe and reset the index so it starts at 0
        if country == 'Japan':
            station_metadata = accmetadata[(accmetadata.net == stn_name[0:2]) & (accmetadata.sta == stn_name[2:])].reset_index(drop=True)
            
        else:
            station_metadata = accmetadata[accmetadata.sta == stn_name].reset_index(drop=True)       # what is going on here


        if len(station_metadata) > 0:
                    
            # Pull out the data. Take the first row of the subset dataframe, 
            #    assuming that the gain, etc. is always the same:
            stnetwork = station_metadata.loc[0].net
            stlon = station_metadata.loc[0].lon
            stlat = station_metadata.loc[0].lat
            stelev = station_metadata.loc[0].elev
            stsamprate = station_metadata.loc[0].samplerate
            stgain = station_metadata.loc[0].gain
            
            # Add to the number of stations counter, because it means this 
            #   is a station with contributing data:
            nostations_i+=1
            
            # If the length of instruments or comonents is 3 or less, then assume
            #   they're all the same instrument but 3 components. Then take the
            ##  first one, and also get the instrument code from instruments:
            if (len(instruments) <= 3) & (len(np.unique(instruments))) == 1:
                instrument_code = instruments[0]
            else:
                print('WARNING - there are more than 3 channels for this station or not all are the same!!!')
                print('Event %s, Station %s' % (earthquake_name,stn_name))
                print(channel_list[i])


            # Start computations         

            # Compute the hypocentral distance
            hypdist = tmf.compute_rhyp(stlon,stlat,stelev,hyplon,hyplat,hypdepth)

            # Append the earthquake and station info for this station
            eventnames = np.append(eventnames,eventname)
            countries = np.append(countries,country)
            origintimes = np.append(origintimes,origintime)
            hyplons = np.append(hyplons,hyplon)
            hyplats = np.append(hyplats,hyplat)
            hypdepths = np.append(hypdepths,hypdepth)
            mws = np.append(mws,mw)
            m0s = np.append(m0s,m0)
            mechanisms = np.append(mechanisms,mechanism)
            networks = np.append(networks,stnetwork)
            stations = np.append(stations,station)
            stlons = np.append(stlons,stlon)
            stlats = np.append(stlats,stlat)
            stelevs = np.append(stelevs,stelev)
            hypdists = np.append(hypdists,hypdist)
            instrument_codes = np.append(instrument_codes,instrument_code)
            

            ### Correct Horizontal components of acceleration 
            
            ## E component 
            
            # Get the values for the E component
            components = np.asarray(components)
            # Get index for E component 
            E_index = np.where(components=='E')[0][0]
            # Read file into a stream object
            E_acc_raw = obspy.read(mseed_list[i][E_index])
            
            # Correct by gain, so everything is in meters
            E_acc_gaincorr = tmf.correct_for_gain(E_acc_raw ,stgain)
            
            # Get the pre-event baseline
            E_acc_baseline = tmf.compute_baseline(E_acc_gaincorr)
            
            # Get the baseline corrected stream object
            E_acc_basecorr = tmf.correct_for_baseline(E_acc_gaincorr,E_acc_baseline)
            
            # High pass filter at fcorner specified above
            E_acc_filt = tmf.highpass(E_acc_basecorr,fcorner,stsamprate,order,zerophase=True)


            ## N component:
            
            # Get the values for the N component
            N_index = np.where(components=='N')[0][0]
            # Read file into a stream object
            N_acc_raw = obspy.read(mseed_list[i][N_index])
            
            # Correct by gain, so everything is in meters
            N_acc_gaincorr = tmf.correct_for_gain(N_acc_raw,stgain)
            
            # Get the pre-event baseline
            N_acc_baseline = tmf.compute_baseline(N_acc_gaincorr)
            
            # Get the baseline corrected stream object
            N_acc_basecorr = tmf.correct_for_baseline(N_acc_gaincorr,N_acc_baseline)
            
            # High pass filter at fcorner specified above
            N_acc_filt = tmf.highpass(N_acc_basecorr,fcorner,stsamprate,order,zerophase=True)
            

            ## Z component:
            
            # Get the values for the N component
            Z_index = np.where(components=='Z')[0][0]
            # Read file into a stream object
            Z_acc_raw = obspy.read(mseed_list[i][Z_index])
            
            # Correct by gain, so everything is in meters
            Z_acc_gaincorr = tmf.correct_for_gain(Z_acc_raw,stgain)
            
            # Get the pre-event baseline
            Z_acc_baseline = tmf.compute_baseline(Z_acc_gaincorr)
            
            # Get the baseline corrected stream object
            Z_acc_basecorr = tmf.correct_for_baseline(Z_acc_gaincorr,Z_acc_baseline)
            
            # High pass filter at fcorner specified above
            Z_acc_filt = tmf.highpass(Z_acc_basecorr,fcorner,stsamprate,order,zerophase=True)


            ## Average horizontal components together
            geom_avg = tmf.get_geom_avg(E_acc_filt[0].data,
                                        N_acc_filt[0].data,Z_acc_filt[0].data)

            # Turn averaged record into a stream object
            avg_st = E_acc_filt.copy()
            tr = avg_st[0]
            tr.stats.channel = 'Average'

            # Save corrected acc mseed file
            filename = '/Users/tnye/tsuquakes/data/Mentawai2010/avg_acc/' + tr.stats.station + '.avg' 
            tr.write(filename, format='MSEED')
            

