#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:14:44 2020

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from numpy import genfromtxt
import sys
from os import makedirs, path, rename
from distutils.dir_util import copy_tree
from glob import glob

def change_rise_vrupt(home, project_name, mf, model_name, shear_wave_fraction_shallow):

    print(f'changing parameters for {project_name}')
    
    # Parameters 
    rupt_dir = f'{home}{project_name}/output/ruptures'
    orig_rupt_dir = f'{home}{project_name}/output/orig_ruptures'

    # Rename original ruptures folder 
    if not path.exists(orig_rupt_dir):
        # Rename original ruptures folder 
        rename(rupt_dir, orig_rupt_dir)
    
        # Create new folder called 'ruptures' to store new rise time .rupt files in
        makedirs(rupt_dir)
    
        # Copy over .log files
        copy_tree(orig_rupt_dir, rupt_dir)
    
    # Gather original rupture files
    rupt_files = np.array(sorted(glob(f'{rupt_dir}/mentawai*.rupt')))
    log_files = np.array(sorted(glob(f'{rupt_dir}/*.log')))
    
    # Gather modified .rupt files
    rise_rupts = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/test_suite/standard/output/risetime_ruptures/rt{mf}x/*.rupt'))
    rise_log = pd.read_csv(glob(f'/Users/tnye/FakeQuakes/simulations/test_suite/standard/output/risetime_ruptures/rt{mf}x/*.log')[0],delimiter='\t')
    vrupt_rupts = sorted(glob(f'/Users/tnye/FakeQuakes/simulations/test_suite/standard/output/vrupt_ruptures/sf{shear_wave_fraction_shallow}/*.rupt'))
    vrupt_log = pd.read_csv(glob(f'/Users/tnye/FakeQuakes/simulations/test_suite/standard/output/vrupt_ruptures/sf{shear_wave_fraction_shallow}/*.log')[0],delimiter='\t')
    
    # Loop through rupture files and multiple rise time (dura column) by 
    # multiplication factor
    for i, rupt in enumerate(rupt_files):
        
        # Read in original .rupt file
        f_orig = open(rupt, 'r')
        header = f_orig.readline().rstrip('\n')
        fault_array = genfromtxt(rupt)
        f_orig.close()
        
        # Read in modified .rupt files
        rise_rupt = genfromtxt(rise_rupts[i])
        vrupt_rupt = genfromtxt(vrupt_rupts[i])
        
        # Change risetime and onset times
        fault_array[:,7] = rise_rupt[:,7]
        fault_array[:,12] = vrupt_rupt[:,12]
        
        # Get avg risetime and vrupt
        avg_rise = rise_log['Mean Risetime (s)'].values[i]
        avg_vrupt = vrupt_log['Mean Vrupt (km/s)'].values[i]
        
        # Save new .rupt file
        rupt_name = rupt.split('/')[-1]
        np.savetxt(f'{rupt_dir}/{rupt_name}', fault_array, delimiter='\t',
                   header=header, comments='', fmt='%s')
    
        # Change average vrupt in .log file to be new vrupt
        log = log_files[i]
        f = open(log, 'r')
        list_of_lines = f.readlines()
        list_of_lines[-5] = 'Average Rise Time (s): %.2f\n' % avg_rise
        list_of_lines[-4] = 'Average Rupture Velocity (km/s): %.2f\n' % avg_vrupt
        f.close()

        f = open(log, 'w')
        f.writelines(list_of_lines)
        f.close()   
        
    return()
