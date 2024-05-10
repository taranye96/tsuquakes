#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:51:07 2022

@author: tnye
"""

# Imports
import time
import numpy as np
import pandas as pd
from glob import glob
import fakequakes
import shutil
from os import remove

ncpus = 4

working_dir = '/Users/tnye/tsuquakes/simulations/gpr_simulations'

#parameter_file = pd.read_csv('/Users/tnye/tsuquakes/files/parameter_flatfiles/final_suite_tPGD_2.csv')
parameter_file = pd.read_csv('/Users/tnye/tsuquakes/files/parameter_flatfiles/final_suite_HF_0.csv')
#rise_vals = parameter_file['Risetime'].values
#vrupt_vals = parameter_file['Vrupt'].values
stress_vals = parameter_file['Stress Drop'].values


##############
faults = ['mentawai_fine2','mentawai_2km','mentawai_2_5km','mentawai_3km','mentawai_5km','mentawai_5km_old']
velmods = ['mentawai_v0.mod','mentawai_v1.mod','mentawai_v2.mod','mentawai_v3.mod','mentawai_v4.mod','mentawai_v5.mod']
gfs = ['gnss_clean.gflist','sm_close.gflist']
run_copy_rupts = [0,1]
run_modify_rupts = [0,1]
use_load_dists = [0,1]
use_g_from_file = [0,1]
run_init = [0,1]
run_generate_rupts = [0,1]
run_gfs = [0,1]
run_lf = [0,1]
run_hf = [0,1]
run_matched = [0,1]
project_complete=[0,1]

##############

for i in range(len(parameter_file)):

#    parameters_list = np.array([
#        [f'{working_dir}/',f'{rise_vals[i]}_{vrupt_vals[i]}_{stress_vals[i]}',faults[0],velmods[1],gfs[1],run_copy_rupts[1],'/Users/tnye/tsuquakes/simulations/ideal_runs_m7.8/standard',run_modify_rupts[1],'no_moho',float(0.8),'on',float(0),float(150),float(0.1),use_load_dists[1],use_g_from_file[1],run_init[0],run_generate_rupts[0],run_gfs[0],run_lf[0],run_hf[1],run_matched[1],project_complete[1]]])

    parameters_list = np.array([
        [f'{working_dir}/',f'{stress_vals[i]}',faults[0],velmods[1],gfs[1],
         run_copy_rupts[1],'/Users/tnye/tsuquakes/simulations/gpr_simulations/standard',
         run_modify_rupts[0],'no_moho',float(0.8),'on',float(0),float(150),float(0.1),
         use_load_dists[1],use_g_from_file[1],run_init[1],run_generate_rupts[0],run_gfs[0],
         run_lf[1],run_hf[1],run_matched[1],project_complete[1]]
        ])
 
#    parameters_list = np.array([
#        [f'{working_dir}/',f'{rise_vals[i]}_{vrupt_vals[i]}_{stress_vals[i]}',faults[0],velmods[1],gfs[0],run_copy_rupts[1],'/Users/tnye/tsuquakes/simulations/ideal_runs_m7.8/standard',run_modify_rupts[1],'no_moho',float(0.8),'on',float(0),float(150),float(0.1),use_load_dists[1],use_g_from_file[1],run_init[1],run_generate_rupts[0],run_gfs[0],run_lf[1],run_hf[0],run_matched[0],project_complete[0]],
#        [f'{working_dir}/',f'{rise_vals[i]}_{vrupt_vals[i]}_{stress_vals[i]}',faults[0],velmods[1],gfs[1],run_copy_rupts[0],'',run_modify_rupts[0],'no_moho',float(0.8),'on',float(0),float(150),float(0.1),use_load_dists[1],use_g_from_file[1],run_init[0],run_generate_rupts[0],run_gfs[0],run_lf[1],run_hf[1],run_matched[1],project_complete[1]]
#        ])


    ##### run simulations #####

    start = time.time()
    for j, parameters in enumerate(parameters_list):
        home,project_name,fault,model_name,GF_list,copy_ruptures,ruptures_folder,modify_ruptures,Qmethod,Qexp,scattering,Qc_exp,baseline_Qc,fcorner_high,load_distances,G_from_file,init,generate_ruptures,GFs,LF,HF,matched,complete = parameters
        
        fakequakes.run_fq(ncpus, home, project_name, fault, model_name, GF_list, int(copy_ruptures), ruptures_folder, int(modify_ruptures), Qmethod, Qexp, scattering, Qc_exp, baseline_Qc, fcorner_high, int(load_distances), int(G_from_file), int(init),int(generate_ruptures), int(GFs), int(LF), int(HF),int(matched))
        
        if int(complete)==1:

            mpi_files = glob(f'{home}{project_name}/output/ruptures/mpi*')
            for file in mpi_files:
                remove(file)
            
            print('removing GFs folder')
            shutil.rmtree(f'{home}{project_name}/GFs')
            
            # Copy over to media
            #print('copying over run to media')
        
            #if not path.exists(f"{home.replace('/Users/','/media/yellowstone/')}"):
                #makedirs(f"{home.replace('/Users/','/media/yellowstone/')}")
        
            #shutil.move(f'{home}{project_name}', f"{home.replace('/Users/','/media/yellowstone/')}{project_name}")
            #copy_tree(f'{home}{project_name}', f"{home.replace('/Users/','/media/yellowstone/')}{project_name}")

    end = time.time()
    total_time = end-start
    tot_time_min = total_time/60
    #print(f'elasped time = {total_time}')

#    with open(timelog,'a') as f:
#        #f.write(f'{rise_vals[i]}_{vrupt_vals[i]}_{stress_vals[i]}\t{tot_time_min} minutes\n')
#        f.write(f'{rise_vals[i]}_{vrupt_vals[i]}\t{tot_time_min} minutes\n')
#    f.close()
