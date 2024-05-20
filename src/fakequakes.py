#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 17:03:43 2022

@author: tnye
"""

# Imports 
import os
from os import path
import numpy as np
from glob import glob
from obspy.core import UTCDateTime
from obspy import read
import time
import shutil
from distutils.dir_util import copy_tree
from mudpy import fakequakes,runslip,forward,view
from change_risetime import change_risetime
from change_vrupt import change_vrupt
from stochastic_parameters import change_parameters
from change_rise_vrupt_finalsuite import change_rise_vrupt

def run_fq(main_ncpus,home,project_name,fault,model_name,GF_list,copy_ruptures,ruptures_folder,modify_ruptures,Qmethod,Qexp,scattering,Qc_exp,baseline_Qc,fcorner_high,load_distances,G_from_file,init,generate_ruptures,GFs,LF,HF,matched,stress_parameter=50,shear_wave_fraction_shallow=0.49):

    copy_ruptures = int(copy_ruptures)
    modify_ruptures = int(modify_ruptures)
    load_distances = int(load_distances)
    G_from_file = int(G_from_file)
    run_init = int(init)
    generate_ruptures = int(generate_ruptures)
    run_GFs = int(GFs)
    run_lf = int(LF)
    run_hf = int(HF)
    run_matched = int(matched)
    fcorner_high = float(fcorner_high)
    main_ncpus = int(main_ncpus)

    print('')
    print('##### Parameters #####')
    print(f'load_distances = {load_distances}')
    print(f'G_from_file = {G_from_file}')
    print(f'run_init = {run_init}')
    print(f'generate_ruptures = {generate_ruptures}')
    print(f'modify_ruptures = {modify_ruptures}')
    print(f'run_GFs = {run_GFs}')
    print(f'run_lf = {run_lf}')
    print(f'run_hf = {run_hf}')
    print(f'run_matched = {run_matched}')
    print(f'stress parameter = {stress_parameter}')
    print(f'baseline Q = {baseline_Qc}')
    print(f'Qexp = {Qexp}')
    print(f'Qc_exp = {Qc_exp}')
    print(f'fc high = {fcorner_high}')
    print('######################')
    print('')

    ################################## Parameters #################################
    
    # # Define variables for project location 
    
    # Runtime parameters 
    ncpus=16                                        # how many CPUS you want to use for parallelization (needs ot be at least 2)
    Nrealizations=16                                # Number of fake ruptures to generate per magnitude bin
    hot_start=0                                    # If code quits in the middle of running, it will pick back up at this index
    
    # File parameters
    # model_name='mentawai.mod'                    # Velocity model file name
    fault_name=f'{fault}.fault'               # Fault model name
    mean_slip_name=f'{home}{project_name}/forward_models/{fault}.rupt'            # Set to path of .rupt file if patterning synthetic runs after a mean rupture model
    run_name='mentawai'                            # Base name of each synthetic run (i.e. mentawai.000000, mentawai.000001, etc...)
    rupture_list='ruptures.list'                   # Name of list of ruptures that are used to generate waveforms.  'ruptures.list' uses the full list of ruptures FakeQuakes creates. If you create file with a sublist of ruptures, use that file name.
    distances_name=fault                      # Name of matrix with estimated distances between subfaults i and j for every subfault pair
    
    # Source parameters
    UTM_zone='47M'                                 # UTM_zone for rupture region 
    time_epi=UTCDateTime('2010-10-25T14:42:12Z')   # Origin time of event (can set to any time, as long as it's not in the future)
    target_Mw=np.array([7.8])                      # Desired magnitude(s), can either be one value or an array
    hypocenter=[100.14, -3.49, 8.82]                                # Coordinates of subfault closest to desired hypocenter, or set to None for random
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
    #rise_time_depths=[0,5]
    max_slip=30                                    # Maximum sip (m) allowed in the model
    max_slip_rule=False                            # If true, uses a magntidude-depence for max slip
    shear_wave_fraction_deep=0.8                   # Shear wave fraction for depths depper than rise_time_depths [1] (0.8 is a standard value (Mai and Beroza 2002))
    source_time_function='dreger'                  # options are 'triangle' or 'cosine' or 'dreger'
    stf_falloff_rate=4                             # Only affects Dreger STF, 4-8 are reasonable values
    num_modes=72                                   # Number of modes in K-L expansion
    slab_name=None                                 # Slab 2.0 Ascii file for 3D geometry, set to None for simple 2D geometry
    mesh_name=None                                 # GMSH output file for 3D geometry, set to None for simple 2D geometry
    
    # Green's Functions parameters
    G_name=GF_list.split('.')[0]                                    # Basename you want for the Green's functions matrices
    make_GFs=1                                     # This should be 1 to run Green's functions
    make_synthetics=1                              # This should be 1 to make the synthetics

    # fk parameters
    # used to solve wave equation in frequency domain 
    dk=0.1 ; pmin=0 ; pmax=1 ; kmax=20             # Should be set to 0.1, 0, 1, 20
    custom_stf=None                                # Assumes specified source time function above if set to None
    
    # Low frequency waveform parameters
    dt=0.5                                         # Sampling interval of LF data 
    NFFT=1024                                       # Number of samples in LF waveforms (should be in powers of 2)
    # dt*NFFT = length of low-frequency dispalcement record
    # want this value to be close to duration (length of high-frequency record)
    
    #intrinsic attenuation aprams    
    # High frequency waveform parameters
    # stress_parameter=50                            # Stress drop measured in bars (standard value is 50)
    moho_depth_in_km=30.0                          # Average depth to Moho in this region 
    Pwave=True                                     # Calculates P-waves as well as S-waves if set to True, else just S-Waves
    kappa=None                                     # Station kappa values: Options are GF_list for station-specific kappa, a singular value for all stations, or the default 0.04s for every station if set to None
    hf_dt=0.01                                     # Sampling interval of HF data
    duration=500                                   # Duration (in seconds) of HF record
    
    # Match filter parameters
    zero_phase=True                                # If True, filters waveforms twice to remove phase, else filters once
    order=4                                        # Number of poles for filters
    fcorner_low=0.998                              # Corner frequency at which to filter waveforms (needs to be between 0 and the Nyquist frequency)
    fcorner_high=0.1                              # Corner frequency at which to filter waveforms (needs to be between 0 and the Nyquist frequency)
    
    ###############################################################################
    
    
    # Initalize project folders
    if run_init==1:
        print('initializing project folder')
        fakequakes.init(home,project_name)
        shutil.copyfile(f'/home/tnye/fakequakes/data/files/velmods/{model_name}', f'{home}{project_name}/structure/{model_name}')
        shutil.copyfile(f'/home/tnye/fakequakes/data/files/fault_info/{fault_name}', f'{home}{project_name}/data/model_info/{fault_name}')
        shutil.copyfile(f"/home/tnye/fakequakes/data/files/model_info/{mean_slip_name.split('/')[-1]}", mean_slip_name)
        print('')
    else:
        print('no init')
        print('')
    
    if not path.exists(f'{home}{project_name}/data/station_info/{GF_list}'):
        shutil.copyfile(f'/home/tnye/fakequakes/data/files/stn_info/{GF_list}', f'{home}{project_name}/data/station_info/{GF_list}')
    
    if load_distances == True:
        copy_tree(f'/media/yellowstone/tnye/fakequakes/distances/{fault}', f'{home}{project_name}/data/distances')

    # Generate ruptures
    if generate_ruptures == 1:
        print('generating ruptures')
        print(f'hypocenter = {hypocenter}')
        print('')
        fakequakes.generate_ruptures(home,project_name,run_name,fault_name,slab_name,mesh_name,load_distances,
                distances_name,UTM_zone,target_Mw,model_name,hurst,Ldip,Lstrike,num_modes,Nrealizations,rake,rise_time,
                rise_time_depths,time_epi,max_slip,source_time_function,lognormal,slip_standard_deviation,scaling_law,
                ncpus,mean_slip_name=mean_slip_name,force_magnitude=force_magnitude,force_area=force_area,
                hypocenter=hypocenter,force_hypocenter=force_hypocenter,shear_wave_fraction_shallow=shear_wave_fraction_shallow,
                shear_wave_fraction_deep=shear_wave_fraction_deep,max_slip_rule=max_slip_rule)

    if copy_ruptures == 1:
        print('copying over ruptures')
        copy_tree(f'{ruptures_folder}/output/ruptures', f'{home}{project_name}/output/ruptures')
        print('')

        print('copying over rupture list')
        print('')
        shutil.copyfile(f'{ruptures_folder}/data/{rupture_list}', f'{home}{project_name}/data/{rupture_list}')

        print('copying over velocity model info')
        print('')
        shutil.copyfile(f"/home/tnye/fakequakes/data/files/velmods/{model_name.replace('mod','nd')}", f"{home}{project_name}/structure/{model_name.replace('mod','nd')}")
        shutil.copyfile(f"/home/tnye/fakequakes/data/files/velmods/{model_name.replace('mod','npz')}", f"{home}{project_name}/structure/{model_name.replace('mod','npz')}")

    if G_from_file==1:
        print('copying over GFs')
        copy_tree(f"/media/yellowstone/tnye/fakequakes/GFs/matrices/{fault}-{model_name.split('.')[0]}", f'{home}{project_name}/GFs/matrices')
        print('...done copying over GFs')
        print('')
    
    project_split = project_name.split('_')
    print(f'project_name = {project_name}')
    if 'sd' in project_name:
            param_ind = np.flatnonzero(np.core.defchararray.find(project_split,'sd')!=-1)[0]
            stress_parameter = float(project_split[param_ind][2:])*10
            print(f'stress drop = {stress_parameter}')
    if modify_ruptures == 1:
        if 'rt' in project_name:
            param_ind = np.flatnonzero(np.core.defchararray.find(project_split,'rt')!=-1)[0]
            mf = float(project_split[param_ind][2:].strip('x'))
        if 'sf' in project_name:
            param_ind = np.flatnonzero(np.core.defchararray.find(project_split,'sf')!=-1)[0]
            shear_wave_fraction_shallow = float(project_split[param_ind][2:])
    
        print('Changing risetime and vrupt')
        print(f'mf = {mf}')
        print(f'sf = {shear_wave_fraction_shallow}\n')
        change_rise_vrupt(home, project_name, mf, model_name, shear_wave_fraction_shallow)

    ncpus = main_ncpus

    if run_GFs==1:
        # Make GFs
        print('making GFs')
        print('')
        runslip.inversionGFs(home,project_name,GF_list,None,fault_name,model_name,
            dt,None,NFFT,None,make_GFs,make_synthetics,dk,pmin,
            pmax,kmax,0,time_epi,hot_start,ncpus,custom_stf,impulse=True) 
    
    if run_lf==1:
        print('making LF waveforms')
        print('')
        #Make low frequency waveforms
        forward.waveforms_fakequakes(home,project_name,fault_name,rupture_list,GF_list,
                    model_name,run_name,dt,NFFT,G_from_file,G_name,source_time_function,
                    stf_falloff_rate)
    
    if run_hf==1:
        # Make high-frequency waveforms
        print('making HF waveforms')
        print('')
        forward.hf_waveforms(home,project_name,fault_name,rupture_list,GF_list,
                model_name,run_name,dt,NFFT,G_from_file,G_name,rise_time_depths,
                moho_depth_in_km,ncpus,source_time_function=source_time_function,
                duration=duration,stf_falloff_rate=stf_falloff_rate,hf_dt=hf_dt,
                Pwave=Pwave,hot_start=hot_start,stress_parameter=stress_parameter,
                kappa=kappa,Qexp=Qexp,Qmethod=Qmethod,scattering=scattering,
                Qc_exp=Qc_exp,baseline_Qc=baseline_Qc)
    
    if run_matched==1:
        # Combine LF and HF waveforms with match filter                              
        print('running matched filter')
        print('')
        forward.match_filter(home,project_name,fault_name,rupture_list,GF_list,zero_phase,order,
                             fcorner_high=fcorner_high,fcorner_low=fcorner_low)
    
    return()
