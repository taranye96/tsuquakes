#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:56:43 2021

@author: tnye
"""

###############################################################################
# Script that changes the rupture onset times of the .rupt files generated by
# FakeQuakes by changing the shallow shear wave fraction. It also stores the
# original .rupt files in another folder. This script is used when varying
# rupture velocity and should be run after generating the ruptures and before
# calculating the Green's functions. 
###############################################################################

# Imports
import numpy as np
from os import makedirs, path
from glob import glob
import geopy.distance
from obspy.taup import TauPyModel
from mudpy import fakequakes

def change_vrupt(home, project_name, model_name, sf):

    # Parameters
    orig_rupt_dir = f'{home}{project_name}/output/ruptures'
    new_rupt_dir = f"/Users/tnye/tsuquakes/files/fakequakes/vrupt_ruptures/{home.split('/')[-2]}/sf{sf}"
    
    # Set up new directory for ruptures
    if not path.exists(new_rupt_dir):
        makedirs(new_rupt_dir)
        
    f_vrupt = open(f'{new_rupt_dir}/vrupt.log','w')
    f_vrupt.write('Run\tMean Vrupt (km/s)\n')
    
    # Gather original rupture files
    rupt_files = np.array(sorted(glob(f'{orig_rupt_dir}/mentawai*.rupt')))

    # Loop through rupture files and change subfault onset time
    for i, rupt in enumerate(rupt_files):
        
        run = rupt.split('/')[-1].strip('.rupt')
        
        # Get parameters to calculate new onset times
        fault_array = np.genfromtxt(rupt)
        slip = np.sqrt(fault_array[:,8]**2 + fault_array[:,9]**2)
        mu = fault_array[:,13]
        # hypocenter = [100.14, -3.49, 11.82]
        hypocenter = [100.14, -3.49, 8.82]
        rise_time_depths = [10,15]
        M0 = sum(slip*fault_array[:,10]*fault_array[:,11]*mu)
        velmod = TauPyModel(model=home+project_name+'/structure/'+model_name.split('.')[0])
        
        dist=((fault_array[:,1]-hypocenter[0])**2+(fault_array[:,2]-hypocenter[1])**2)**0.5
        shypo=np.argmin(dist)
        
        # Open rupture model
        f = open(rupt, 'r')
        header = f.readline().rstrip('\n')
        
        # Calculate new onset times
        new_onset, length2fault = fakequakes.get_rupture_onset(home,project_name,slip,fault_array,
                    model_name,hypocenter,shypo,rise_time_depths, M0,velmod,sigma_rise_time=0.2,
                    shear_wave_fraction_shallow=sf)
    
        # Change rupt_time column to be new rupture onset times
        fault_array[:,12] = new_onset
        
        # Save new .rupt file
        rupt_name = rupt.split('/')[-1]
        np.savetxt(f'{new_rupt_dir}/{rupt_name}', fault_array, delimiter='\t',
                   header=header, comments='', fmt='%s')
    
        # Calculate average rupture velocity
        lon_array = fault_array[:,1]
        lat_array = fault_array[:,2]
        vrupt = []
        
        for j in range(len(fault_array)):
            if new_onset[j] > 0:
                r = geopy.distance.geodesic((hypocenter[1], hypocenter[0]), (lat_array[j], lon_array[j])).km
                vrupt.append(r/new_onset[j])
    
        avg_vrupt = np.mean(np.array(vrupt)[np.where(np.array(vrupt)>0)[0]])
        
        f_vrupt.write(f'{run}\t{round(avg_vrupt,3)}\n')

    f_vrupt.close()
    
    return()

# sf_list = [0.37,0.4,0.43,0.46,0.49]
# sf_list = [0.415,0.447,0.42,0.42]
sf_list = [0.3,0.4]

home = '/Users/tnye/tsuquakes/simulations/test_simulations/'
project_name = 'standard'
model_name = 'mentawai_v1.mod'

for sf in sf_list:
    change_vrupt(home, project_name, model_name, sf)
