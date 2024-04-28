#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 17:03:43 2022

@author: tnye
"""

# Imports 
import os
import numpy as np
from obspy.core import UTCDateTime
from obspy import read
import time
import shutil
from distutils.dir_util import copy_tree
from mudpy import fakequakes,runslip,forward,view
from change_risetime import change_risetime
from change_vrupt import change_vrupt

def run_fq(home,project_name,model_name,GF_list,ruptures,stress_parameter=50,shear_wave_fraction_shallow=0.49):
    ################################## Parameters #################################
    
    run_init=1
    run_GFs=0
    run_lf=0
    run_hf=0
    run_matched=0
    
    G_from_file=1
    
    # Runtime parameters 
    ncpus=2                                        # how many CPUS you want to use for parallelization (needs ot be at least 2)
    Nrealizations=2                                # Number of fake ruptures to generate per magnitude bin
    hot_start=0                                    # If code quits in the middle of running, it will pick back up at this index
    
    # File parameters
    # model_name='mentawai.mod'                    # Velocity model file name
    fault_name='mentawai_fine2.fault'               # Fault model name
    mean_slip_name=f'{home}{project_name}/forward_models/mentawai_fine2.rupt'            # Set to path of .rupt file if patterning synthetic runs after a mean rupture model
    # mean_slip_name=None
    run_name='mentawai'                            # Base name of each synthetic run (i.e. mentawai.000000, mentawai.000001, etc...)
    rupture_list='ruptures.list'                   # Name of list of ruptures that are used to generate waveforms.  'ruptures.list' uses the full list of ruptures FakeQuakes creates. If you create file with a sublist of ruptures, use that file name.
    distances_name='mentawai'                      # Name of matrix with estimated distances between subfaults i and j for every subfault pair
    load_distances=0                               # This should be zero the first time you run FakeQuakes with your fault model.
    
    # Source parameters
    UTM_zone='47M'                                 # UTM_zone for rupture region 
    time_epi=UTCDateTime('2010-10-25T14:42:12Z')   # Origin time of event (can set to any time, as long as it's not in the future)
    target_Mw=np.array([7.8])                      # Desired magnitude(s), can either be one value or an array
    hypocenter=[100.14, -3.49, 11.82]                      # Coordinates of subfault closest to desired hypocenter, or set to None for random
    force_hypocenter=True                         # Set to True if hypocenter specified
    rake=90                                        # Average rake for subfaults
    scaling_law='T'                                # Type of rupture: T for thrust, S for strike-slip, N for normal
    force_magnitude=True                           # Set to True if you want the rupture magnitude to equal the exact target magnitude
    force_area=True                                # Set to True if you want the ruptures to fill the whole fault model
    
    # Correlation function parameters
    hurst=0.4                                      # Hurst exponent form Melgar and Hayes 2019
    Ldip='auto'                                    # Correlation length scaling: 'auto' uses Melgar and Hayes 2019, 'MB2002' uses Mai and Beroza 2002
    Lstrike='auto'                                 # Same as above
    slip_standard_deviation=0.9                    # Standard deviation for slip statistics: Keep this at 0.9
    lognormal=True                                 # Keep this as True to solve the problem of some negative slip subfaults that are produced
    
    # Rupture propagation parameters
    rise_time = 'MH2017'                           # Rise time scaling to use. 'GP2010' uses Graves and Pitarka (2010), 'GP2015' uses Graves and Pitarka (2015), 'S1999' uses Sommerville (1999), and 'MH2017' uses Melgar and Hayes (2017).  
    rise_time_depths=[10,15]                       # Transition depths for rise time scaling (if slip shallower than first index, rise times are twice as long as calculated)
    max_slip=40                                    # Maximum sip (m) allowed in the model
    max_slip_rule=False                            # If true, uses a magntidude-depence for max slip
    # shear_wave_fraction_shallow=0.49               # Shear wave fraction for depths shallower than rise_time_depths[0]
    shear_wave_fraction_deep=0.8                   # Shear wave fraction for depths depper than rise_time_depths [1] (0.8 is a standard value (Mai and Beroza 2002))
    source_time_function='dreger'                  # options are 'triangle' or 'cosine' or 'dreger'
    stf_falloff_rate=4                             # Only affects Dreger STF, 4-8 are reasonable values
    num_modes=72                                   # Number of modes in K-L expansion
    slab_name=None                                 # Slab 2.0 Ascii file for 3D geometry, set to None for simple 2D geometry
    mesh_name=None                                 # GMSH output file for 3D geometry, set to None for simple 2D geometry
    
    # Green's Functions parameters
    # GF_list='sm.gflist'                          # Stations file name
    G_name=GF_list.split('.')[0]                   # Basename you want for the Green's functions matrices
    make_GFs=1                                     # This should be 1 to run Green's functions
    make_synthetics=1                              # This should be 1 to make the synthetics
    G_from_file=0                                  # This should be zero the first time you run FakeQuakes with your fault model and stations.
    
    # fk parameters
    # used to solve wave equation in frequency domain 
    dk=0.1 ; pmin=0 ; pmax=1 ; kmax=20             # Should be set to 0.1, 0, 1, 20
    custom_stf=None                                # Assumes specified source time function above if set to None
    
    # Low frequency waveform parameters
    dt=0.5                                         # Sampling interval of LF data 
    NFFT=512                                       # Number of samples in LF waveforms (should be in powers of 2)
    # dt*NFFT = length of low-frequency dispalcement record
    # want this value to be close to duration (length of high-frequency record)
    
    # High frequency waveform parameters
    # stress_parameter=50                            # Stress drop measured in bars (standard value is 50)
    moho_depth_in_km=30.0                          # Average depth to Moho in this region 
    Pwave=True                                     # Calculates P-waves as well as S-waves if set to True, else just S-Waves
    kappa=None                                     # Station kappa values: Options are GF_list for station-specific kappa, a singular value for all stations, or the default 0.04s for every station if set to None
    hf_dt=0.01                                     # Sampling interval of HF data
    duration=250                                   # Duration (in seconds) of HF record
    
    # Match filter parameters
    zero_phase=True                                # If True, filters waveforms twice to remove phase, else filters once
    order=4                                        # Number of poles for filters
    fcorner_low=0.998                              # Corner frequency at which to filter waveforms (needs to be between 0 and the Nyquist frequency)
    fcorner_high=0.01                              # Corner frequency at which to filter waveforms (needs to be between 0 and the Nyquist frequency)
    
    ###############################################################################
    
    
    # Initalize project folders
    if run_init==1:
        fakequakes.init(home,project_name)
        shutil.copyfile(f'/Users/tnye/FakeQuakes/files/{model_name}', f'{home}{project_name}/structure/{model_name}')
        # shutil.copyfile('/Users/tnye/FakeQuakes/files/mentawai_fine.fault', f'{home}{project_name}/data/model_info/mentawai_fine.fault')
        shutil.copyfile('/Users/tnye/tsuquakes/files/mentawai_fine2.fault', f'{home}{project_name}/data/model_info/mentawai_fine2.fault')
        shutil.copyfile('/Users/tnye/FakeQuakes/files/mentawai_fine2.rupt', f'{home}{project_name}/forward_models/mentawai_fine2.rupt')
        shutil.copyfile(f'/Users/tnye/FakeQuakes/files/{GF_list}', f'{home}{project_name}/data/station_info/{GF_list}')
    
    # Generate ruptures
    if ruptures == 'new':
        fakequakes.generate_ruptures(home,project_name,run_name,fault_name,slab_name,mesh_name,load_distances,
                distances_name,UTM_zone,target_Mw,model_name,hurst,Ldip,Lstrike,num_modes,Nrealizations,rake,rise_time,
                rise_time_depths,time_epi,max_slip,source_time_function,lognormal,slip_standard_deviation,scaling_law,
                ncpus,mean_slip_name=mean_slip_name,force_magnitude=force_magnitude,force_area=force_area,
                hypocenter=hypocenter,force_hypocenter=force_hypocenter,shear_wave_fraction_shallow=shear_wave_fraction_shallow,
                shear_wave_fraction_deep=shear_wave_fraction_deep,max_slip_rule=max_slip_rule)
    else:
       copy_tree('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/ruptures', f'{home}{project_name}/output/ruptures')
       shutil.copyfile(f'/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/standard_parameters/data/ruptures.list', f'{home}{project_name}/data/ruptures.list')
       load_distances=1
       copy_tree('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/distances', f'{home}{project_name}/data/distances')
       copy_tree('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/structure', f'{home}{project_name}/structure')
     
        
    if G_from_file==1:
        copy_tree('/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/GFs/matrices', f'{home}{project_name}/GFs/matrices')
    
    if 'sd' in project_name:
        stress_parameter = float(project_name[3:])*10
    elif 'rt' in project_name:
        mf = float(project_name[3])
        change_risetime(home, project_name, mf)
    elif 'sf' in project_name:
        shear_wave_fraction_shallow = float(project_name[3:])
        change_vrupt(home, project_name, model_name, shear_wave_fraction_shallow)
    
    # Make GFs
    ncpus=4  
    if run_GFs==1:
        runslip.inversionGFs(home,project_name,GF_list,None,fault_name,model_name,
            dt,None,NFFT,None,make_GFs,make_synthetics,dk,pmin,
            pmax,kmax,0,time_epi,hot_start,ncpus,custom_stf,impulse=True) 
    
    #Make low frequency waveforms
    if run_lf==1:
        forward.waveforms_fakequakes(home,project_name,fault_name,rupture_list,GF_list,
                    model_name,run_name,dt,NFFT,G_from_file,G_name,source_time_function,
                    stf_falloff_rate)
    
    # Make high-frequency waveforms
    if run_hf==1:
        forward.hf_waveforms(home,project_name,fault_name,rupture_list,GF_list,
                        model_name,run_name,dt,NFFT,G_from_file,G_name,rise_time_depths,
                        moho_depth_in_km,ncpus,source_time_function=source_time_function,
                        duration=duration,stf_falloff_rate=stf_falloff_rate,hf_dt=hf_dt,
                        Pwave=Pwave,hot_start=hot_start,stress_parameter=stress_parameter)
        
    # Combine LF and HF waveforms with match filter
    if run_matched==1:                              
        forward.match_filter(home,project_name,fault_name,rupture_list,GF_list,zero_phase,order,
                             fcorner_high=fcorner_high,fcorner_low=fcorner_low)
    
    return()