#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:12:03 2020

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import obspy
from glob import glob
import tsueqs_main_fns as tmf
import signal_average_fns as avg


earthquake_name = 'Mentawai2010'

### Set paths and parameters ####
# Project directory 
proj_dir = '/Users/tnye/tsuquakes' 

# Table of earthquake data
eq_table_path = '/Users/tnye/tsuquakes/data/events.csv'   
eq_table = pd.read_csv(eq_table_path)

# Data directory                          
data_dir = proj_dir + '/data'
     

# Velocity filtering                     
arias_disp_threshold = 0.0   


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

### Read in and organize displacement data ###
# Get an array of all channels with instruments with L or N instrument code and
#    an E, N, or Z direction
disp_chan_files = np.array(sorted(glob(data_dir + '/' + earthquake_name +
                                    '/disp/*.LX[E,N,Z]*')))

# Read in displacement metadata file for the earthquake
dispmetadata_file = data_dir + '/' + earthquake_name + '/' + earthquake_name + '_disp.chan'
dispmetadata = pd.read_csv(dispmetadata_file, sep='\t', header=0,
                          names=['net', 'sta', 'loc', 'chan', 'lat',
                                 'lon', 'elev', 'samplerate', 'gain', 'units'])

# There might be white spaces in the station name, so remove those
dispmetadata.sta = dispmetadata.sta.astype(str)
dispmetadata.sta = dispmetadata.sta.str.replace(' ','')

# Obtain gain and units
disp_gain = dispmetadata['gain'][0]
disp_units = dispmetadata['units'][0]

# Create lists to add station names, channels, and miniseed files to 
stn_name_list = []
channel_list = []
mseed_list = []

# Group all disp files by station
N = 3
stn_files = [disp_chan_files[n:n+N] for n in range(0, len(disp_chan_files), N)]

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
            station_metadata = dispmetadata[(dispmetadata.net == stn_name[0:2]) & (dispmetadata.sta == stn_name[2:])].reset_index(drop=True)
            
        else:
            station_metadata = dispmetadata[dispmetadata.sta == stn_name].reset_index(drop=True)       # what is going on here


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
            

            ### Correct Horizontal components of adisplacement
            
            ## E component 
            
            # Get the values for the E component
            components = np.asarray(components)
            # Get index for E component 
            E_index = np.where(components=='E')[0][0]
            # Read file into a stream object
            E_disp_raw = obspy.read(mseed_list[i][E_index])
            
            # Correct by gain, so everything is in meters
            E_disp_gaincorr = tmf.correct_for_gain(E_disp_raw ,stgain)
            
            # Get the pre-event baseline
            E_disp_baseline = tmf.compute_baseline(E_disp_gaincorr, 10)
            
            # Get the baseline corrected stream object
            E_disp_corr = tmf.correct_for_baseline(E_disp_gaincorr,E_disp_baseline)


            ## N component:
            
            # Get the values for the N component
            N_index = np.where(components=='N')[0][0]
            # Read file into a stream object
            N_disp_raw = obspy.read(mseed_list[i][N_index])
            
            # Correct by gain, so everything is in meters
            N_disp_gaincorr = tmf.correct_for_gain(N_disp_raw,stgain)
            
            # Get the pre-event baseline
            N_disp_baseline = tmf.compute_baseline(N_disp_gaincorr, 10)
            
            # Get the baseline corrected stream object
            N_disp_corr = tmf.correct_for_baseline(N_disp_gaincorr,N_disp_baseline)
            

            ## Z component:
            
            # Get the values for the N component
            Z_index = np.where(components=='Z')[0][0]
            # Read file into a stream object
            Z_disp_raw = obspy.read(mseed_list[i][Z_index])
            
            # Correct by gain, so everything is in meters
            Z_disp_gaincorr = tmf.correct_for_gain(Z_disp_raw,stgain)
            
            # Get the pre-event baseline
            Z_disp_baseline = tmf.compute_baseline(Z_disp_gaincorr, 10)
            
            # Get the baseline corrected stream object
            Z_disp_corr = tmf.correct_for_baseline(Z_disp_gaincorr,Z_disp_baseline)


            ## Average horizontal components together
            geom_avg = avg.get_geom_avg_3comp(E_disp_corr[0].data,
                                        N_disp_corr[0].data,Z_disp_corr[0].data)

            # Turn averaged record into a stream object
            avg_st = E_disp_corr.copy()
            tr = avg_st[0]
            tr.stats.channel = 'Average'

            # Save corrected disp mseed file
            filename = '/Users/tnye/tsuquakes/data/Mentawai2010/avg_disp/' + tr.stats.station + '.avg' 
            tr.write(filename, format='MSEED')
            
            
           


