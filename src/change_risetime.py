#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:14:44 2020

@author: tnye
"""

# Imports
import numpy as np
from numpy import genfromtxt
import sys
from os import makedirs, path, rename
from distutils.dir_util import copy_tree
from glob import glob

def change_risetime(home, project_name, mf):

    # print(f'changing risetime for {project_name}, mf={mf}')
    
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

    #rupt_files = np.array(np.genfromtxt(f'{home}{project_name}/data/ruptures.list',dtype=str))
    #rupt_files = [rupt_files[1]]
    #log_files = [log_files[1]]
    print(f'(rupt_files = {rupt_files}')

    # Loop through rupture files and multiple rise time (dura column) by 
    # multiplication factor
    for i, rupt in enumerate(rupt_files):

        print(f'Index {i} working on rupt {rupt}')
        
        f = open(rupt, 'r')
        header = f.readline().rstrip('\n')
        
        fault_array = genfromtxt(rupt)
        dura = fault_array[:,7]
        
        new_dura = dura
        
        # # Create empty array to store new rise times in
        # new_dura = np.array([])
        
        # for rt in dura:
        #     new_dura = np.append(new_dura, mf*rt)
        
        # Change dura column to be new_dura column
        fault_array[:,7] = new_dura
    
        # Save new .rupt file
        rupt_name = rupt.split('/')[-1]
        np.savetxt(f'{rupt_dir}/{rupt_name}', fault_array, delimiter='\t',
                   header=header, comments='', fmt='%s')
    
        # Calculate average risetime
        avg_rise = np.mean(new_dura)
    
        # Change average vrupt in .log file to be new vrupt
        log = log_files[i]
        #log = rupt.replace('rupt','log')
        f = open(log, 'r')
        list_of_lines = f.readlines()
        list_of_lines[-5] = 'Average Rise Time (s): %.2f\n' % avg_rise
        f.close()

        f = open(log, 'w')
        f.writelines(list_of_lines)
        f.close()   
        
    return()
