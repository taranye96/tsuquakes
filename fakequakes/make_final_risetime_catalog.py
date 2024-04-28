#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:14:44 2020

@author: tnye
"""

# Imports
import numpy as np
from numpy import genfromtxt
from os import makedirs, path
from glob import glob

def change_risetime(home, project_name, mf):
    
    # Parameters 
    orig_rupt_dir = f'{home}{project_name}/output/ruptures'
    # orig_rupt_dir = f'/Users/tnye/FakeQuakes/simulations/final_suite/add_ruptures/output/ruptures'
    new_rupt_dir = f"/Users/tnye/FakeQuakes/files/risetime_ruptures/{home.split('/')[-2]}/rt{mf}x"
    
    # Set up new directory for ruptures
    if not path.exists(new_rupt_dir):
        makedirs(new_rupt_dir)
        
    f_rise = open(f'{new_rupt_dir}/risetime.log','w')
    f_rise.write('Run\tMean Risetime (s)\n')
    
    # Gather original rupture files
    rupt_files = np.array(sorted(glob(f'{orig_rupt_dir}/mentawai*.rupt')))

    # Loop through rupture files and multiple rise time (dura column) by 
    # multiplication factor
    for i, rupt in enumerate(rupt_files):
        
        run = rupt.split('/')[-1].strip('.rupt')
        
        f = open(rupt, 'r')
        header = f.readline().rstrip('\n')
        
        fault_array = genfromtxt(rupt)
        dura = fault_array[:,7]
        
        # Create empty array to store new rise times in
        new_dura = np.array([])
        
        for rt in dura:
            new_dura = np.append(new_dura, mf*rt)
        
        # Change dura column to be new_dura column
        fault_array[:,7] = new_dura
        
        # Save new .rupt file
        rupt_name = rupt.split('/')[-1]
        np.savetxt(f'{new_rupt_dir}/{rupt_name}', fault_array, delimiter='\t',
                   header=header, comments='', fmt='%s')
    
        # Calculate average risetime
        avg_rise = np.mean(new_dura[np.where(new_dura>0)[0]])
    
        f_rise.write(f'{run}\t{round(avg_rise,3)}\n')
    
    f_rise.close()
        
    return()

# mf_list = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3]
# mf_list = [1.422,1.742,1.2,2.0]
mf_list = [1.358,1.808,1.1]

home = '/Users/tnye/FakeQuakes/simulations/ideal_runs_m7.8/'
project_name = 'standard'

for mf in mf_list:
    change_risetime(home, project_name, mf)



