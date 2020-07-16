#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:16:28 2019

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

data_types = ['disp', 'accel']

### Set paths and parameters ####
# Project directory 
proj_dir = '/Users/tnye/tsuquakes' 

# Table of earthquake data
eq_table_path = '/Users/tnye/tsuquakes/data/events.csv'   
eq_table = pd.read_csv(eq_table_path)

# Data directories         
data_dir = proj_dir + '/data' 
disp_dir = data_dir + '/' + earthquake_name + '/disp'
sm_dir = data_dir + '/' + earthquake_name + '/accel'

# Path to send flatfiles
flatfile_path = proj_dir + '/flatfiles/obs_IMs.csv'     

# Velocity filtering
fcorner = 1/15.                          # Frequency at which to high pass filter
order = 2                                # Number of poles for filter  

 # Gather displacement and strong motion files
disp_files = np.array(sorted(glob(disp_dir + '/*.mseed')))
sm_files = np.array(sorted(glob(sm_dir + '/*.mseed')))

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


##################### Data Processing and Calculations ####################

for data in data_types:
    
    if data == 'disp':
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_disp.chan'
        files = disp_files
        IMs = ['pgd']
        threshold = 0.0
        nsamples = 10
        filtering = False
        code = 'LX'
    elif data == 'accel':
        metadata_file = data_dir + '/' + eventname + '/' + eventname + '_sm.chan'
        files = sm_files
        IMs = ['pga', 'pgv']
        threshold = 0.0
        nsamples = 100
        code = 'HN'
        filtering = True

    metadata = pd.read_csv(metadata_file, sep='\t', header=0,
                          names=['net', 'sta', 'loc', 'chan', 'lat',
                                 'lon', 'elev', 'samplerate', 'gain', 'units'])
    
    # There might be white spaces in the station name, so remove those
    metadata.sta = metadata.sta.astype(str)
    metadata.sta = metadata.sta.str.replace(' ','')

    # Create lists to add station names, channels, and miniseed files to 
    stn_name_list = []
    channel_list = []
    mseed_list = []
    
    # Group all files by station
    N = 3
    stn_files = [files[n:n+N] for n in range(0, len(files), N)]
    
    # Obtain gain and units
    gain = metadata['gain'][0]
    units = metadata['units'][0]

    
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
        for j in range(len(channel_list[i])):
            components.append(channel_list[i][j][2])
            
        # Get the type of instrument (HL or HN):
        instruments = []
        for j in range(len(channel_list[i])):
            instruments.append(channel_list[i][j][0:2])          
            
            # Get the metadata for this station from the chan file - put it into
            #     a new dataframe and reset the index so it starts at 0
            if country == 'Japan':
                station_metadata = metadata[(metadata.net == stn_name[0:2]) & (metadata.sta == stn_name[2:])].reset_index(drop=True)
                
            else:
                station_metadata = metadata[metadata.sta == stn_name].reset_index(drop=True)       # what is going on here
    
    
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
    
                        
                # Get the values for the E component
                components = np.asarray(components)
                if 'E' in components:
                    # Get index for E component 
                    E_index = np.where(components=='E')[0][0]
                    # Read file into a stream object
                    E_raw = obspy.read(mseed_list[i][E_index])
                    
                    # Correct by gain, so everything is in meters
                    E_gaincorr = tmf.correct_for_gain(E_raw ,stgain)
                    
                    # Get the pre-event baseline
                    E_baseline = tmf.compute_baseline(E_gaincorr,nsamples)
                    
                    # Get the baseline corrected stream object
                    E_basecorr = tmf.correct_for_baseline(E_gaincorr,E_baseline)

                    # High pass filter strong motion data at fcorner specified above
                    if filtering == True:
                        E_filt = tmf.highpass(E_basecorr,fcorner,stsamprate,order,zerophase=True)
                        E_corr = E_filt
                    else:
                        E_corr = E_basecorr
                    
                    # Get the duration, stream file time of start, and time of stop of shaking
                    E_Td, E_start, E_end = tmf.determine_Td(threshold,E_corr)      
                    E_Td_list = np.append(E_Td_list,E_Td)
    
                    # Save corrected mseed file
                    tra = E_corr[0]
                    tra.stats.channel = code + 'E'
                    filename = '/Users/tnye/tsuquakes/data/Mentawai2010/' + data + '_corr/' + tra.stats.station + '.' + tra.stats.channel + '.corr' 
                    tra.write(filename, format='MSEED')

                    
                # Get the values for the N component
                if 'N' in components:
                    # Get index for E component 
                    N_index = np.where(components=='N')[0][0]
                    # Read file into a stream object
                    N_raw = obspy.read(mseed_list[i][N_index])
                    
                    # Correct by gain, so everything is in meters
                    N_gaincorr = tmf.correct_for_gain(N_raw,stgain)
                    
                    # Get the pre-event baseline
                    N_baseline = tmf.compute_baseline(N_gaincorr,nsamples)
                    
                    # Get the baseline corrected stream object
                    N_basecorr = tmf.correct_for_baseline(N_gaincorr,N_baseline)
                    
                    # High pass filter strong motion data at fcorner specified above
                    if filtering == True:
                        N_filt = tmf.highpass(N_basecorr,fcorner,stsamprate,order,zerophase=True)
                        N_corr = N_filt
                    else:
                        N_corr = N_basecorr
                        
                    # Get the duration, stream file time of start, and time of stop of shaking
                    N_Td, N_start, N_end = tmf.determine_Td(threshold,N_corr)  
                    N_Td_list = np.append(N_Td_list,N_Td)
    
                    # Save corrected acc mseed file
                    tra = N_corr[0]
                    tra.stats.channel = code + 'N'
                    filename = '/Users/tnye/tsuquakes/data/Mentawai2010/' + data + '_corr/' + tra.stats.station + '.' + tra.stats.channel + '.corr' 
                    tra.write(filename, format='MSEED')

    
                # Get the values for the Z component
                if 'Z' in components:
                    # Get index for Z component 
                    Z_index = np.where(components=='Z')[0][0]     
                    # Read file into a stream object                     
                    Z_raw = obspy.read(mseed_list[i][Z_index])
                    
                    # Correct by gain, so everything is in meters
                    Z_gaincorr = tmf.correct_for_gain(Z_raw,stgain)
                    
                    # Get the pre-event baseline
                    Z_baseline = tmf.compute_baseline(Z_gaincorr,nsamples)
                    
                    # Get the baseline corrected stream object
                    Z_basecorr = tmf.correct_for_baseline(Z_gaincorr,Z_baseline)
                    
                    # High pass filter strong motion data at fcorner specified above
                    if filtering == True:
                        Z_filt = tmf.highpass(Z_basecorr,fcorner,stsamprate,order,zerophase=True)
                        Z_corr = Z_filt
                    else:
                        Z_corr = Z_basecorr
                    
                    # Get the duration, stream file time of start, and time of stop of shaking
                    Z_Td, Z_start, Z_end = tmf.determine_Td(threshold,Z_corr)  
                    Z_Td_list = np.append(Z_Td_list,Z_Td)
    
                    # Save corrected mseed file
                    tra = Z_corr[0]
                    tra.stats.channel = code + 'Z'
                    filename = '/Users/tnye/tsuquakes/data/Mentawai2010/' + data + '_corr/' + tra.stats.station + '.' + tra.stats.channel + '.corr' 
                    tra.write(filename, format='MSEED')
    
    
                # Get the values for the horizontal
                if ('E' in components) and ('N' in components):
                    
                    # Take the min time of E and N start times to be the start
                    EN_start = np.min([E_start,N_start])
                    
                    # Take the max time of E and N end times to be the end
                    EN_end = np.max([E_end,N_end])
                    
                    # Get the duration to be the time between these
                    EN_Td = EN_end - EN_start
                    horiz_Td_list = np.append(horiz_Td_list,EN_Td)
    
                else:
                    ## Append nan to the overall arrays:
                    horizon_Td_list = np.append(horiz_Td_list,np.nan)
                    
                    
                # Get the values for all 3 components
                if ('E' in components) and ('N' in components) and ('Z' in components):
                    
                    # Take the min time of the E,N,Z start times to be the start
                    ENZ_start = np.min([E_start,N_start,Z_start])
                    
                    # Take the max of the E,N,Z end times to be the end
                    ENZ_end = np.max([E_end,N_end,Z_end])
                    
                    # Get the duration to be the time between these
                    ENZ_Td = ENZ_end - ENZ_start
                    comp3_Td_list = np.append(comp3_Td_list,ENZ_Td)

                else:
                    ## Append nan to the overall arrays:
                    comp3_Td_list = np.append(comp3_Td_list,np.nan)


                ################### Velocity ###################

                if data == 'accel':

                ### Integrate unfiltered acc data to get velocity data
                
                    ## East component 
                    E_vel_unfilt = tmf.accel_to_veloc(E_basecorr)
                    E_vel = tmf.highpass(E_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
                    
                    # Save filtered velocity mseed file
                    trv = E_vel[0]
                    trv.stats.channel = 'HNE'
                    filename_vel = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr/' + trv.stats.station + '.HNE.corr' 
                    trv.write(filename_vel, format='MSEED')

                    ## North component 
                    N_vel_unfilt = tmf.accel_to_veloc(N_basecorr)
                    N_vel = tmf.highpass(N_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
                    
                    # Save filtered velocity mseed file
                    trv = N_vel[0]
                    trv.stats.channel = 'HNN'
                    filename_vel = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr/' + trv.stats.station + '.HNN.corr' 
                    trv.write(filename_vel, format='MSEED')
                    
                    ## Vertical component 
                    Z_vel_unfilt = tmf.accel_to_veloc(Z_basecorr)
                    Z_vel = tmf.highpass(Z_vel_unfilt,fcorner,stsamprate,order,zerophase=True)
                    
                    # Save filtered velocity mseed file
                    trv = Z_vel[0]
                    trv.stats.channel = 'HNZ'
                    filename_vel = '/Users/tnye/tsuquakes/data/Mentawai2010/vel_corr/' + trv.stats.station + '.HNZ.corr' 
                    trv.write(filename_vel, format='MSEED')
                    

                ####################### IMs ########################

                if 'pgd' in IMs:
                    # Get euclidean norm of displacement components 
                    euc_norm = avg.get_eucl_norm_3comp(E_corr[0].data,
                                                       N_corr[0].data,
                                                       Z_corr[0].data)
                    # Calculate PGD
                    pgd = np.max(np.abs(euc_norm))
                    pgd_list = np.append(pgd_list,pgd)
                else:
                    pgd_list = np.append(pgd_list,np.nan)
                    
                if 'pga' in IMs:
                    # Calcualte PGA on each component 
                    pga_E = tmf.get_peak_value(E_corr)
                    pga_N = tmf.get_peak_value(N_corr)
                    # Get geometric average of the PGAs
                    pga = tmf.get_geom_avg(pga_E,pga_N)
                    pga_list = np.append(pga_list,pga)
                else:
                    pga_list = np.append(pga_list,np.nan)

                if 'pgv' in IMs:
                    # Calculate PGV on each component
                    pgv_E = tmf.get_peak_value(E_vel)
                    pgv_N = tmf.get_peak_value(N_vel)
                    # Get geometric average of the PGVs
                    pgv = tmf.get_geom_avg(pgv_E,pgv_N)
                    pgv_list = np.append(pgv_list,pgv)
                else:
                    pgv_list = np.append(pgv_list,np.nan)

                components = components.tolist()
    
                
    # Finalize the number of stations for each entry for this earthquake:
    nostations = np.append(nostations,np.full(nostations_i,nostations_i))

## Now, put all the final arrays together into a pandas dataframe. First mak a dict:
dataset_dict = {'eventname':eventnames,'country':countries, 'origintime':origintimes,
                    'hyplon':hyplons, 'hyplat':hyplats, 'hypdepth (km)':hypdepths,
                    'mw':mws, 'm0':m0s, 'network':networks, 'station':stations,
                    'stlon':stlons, 'stlat':stlats, 'stelev':stelevs, 
                    'instrumentcode':instrument_codes,
                    'mechanism':mechanisms, 'hypdist':hypdists, 'duration_e':E_Td_list,
                    'duration_n':N_Td_list, 'duration_z':Z_Td_list, 'duration_horiz':horiz_Td_list,
                    'duration_3comp':comp3_Td_list, 'pga':pga_list, 'pgv':pgv_list,
                    'pgd':pgd_list}

# Should nostations be added to flatfile?

## Make it a dataframe:
flatfile_df = pd.DataFrame(data=dataset_dict)


## Save to file:
flatfile_df.to_csv(flatfile_path,index=False)
