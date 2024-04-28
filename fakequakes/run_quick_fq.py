#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:51:07 2022

@author: tnye
"""

# Imports
import numpy as np
import quick_fq

working_dir = '/Users/tnye/FakeQuakes/FQ_status'

# parameters_list = np.array([[f'{working_dir}/new_Q_model/','standard_parameters','mentawai_newQ.mod','sm.gflist'],
#                       [f'{working_dir}/orig_Q_model/','standard_parameters','mentawai.mod','sm.gflist'],
#                       [f'{working_dir}/new_Q_model/stress_drop/','sd_1.0','mentawai_newQ.mod','sm_short.gflist'],
#                       [f'{working_dir}/new_Q_model/stress_drop/','sd_0.1','mentawai_newQ.mod','sm_short.gflist'],
#                       [f'{working_dir}/new_Q_model/stress_drop/','sd_0.01','mentawai_newQ.mod','sm_short.gflist'],
#                       [f'{working_dir}/new_Q_model/risetime/','rt_2x','mentawai_newQ.mod','sm_short.gflist'],
#                       [f'{working_dir}/new_Q_model/risetime/','rt_3x','mentawai_newQ.mod','sm_short.gflist'],
#                       [f'{working_dir}/new_Q_model/vrupt/','sf_0.3','mentawai_newQ.mod','sm_short.gflist'],
#                       [f'{working_dir}/new_Q_model/vrupt/','sf_0.4','mentawai_newQ.mod','sm_short.gflist'],
#                       [f'{working_dir}/new_Q_model/vrupt/','sf_0.22','mentawai_newQ.mod','sm_short.gflist']])

# parameters_list = np.array([
#                       [f'{working_dir}/new_Q_model/risetime/','rt_2x','mentawai_newQ.mod','sm_short.gflist','standardized'],
#                       [f'{working_dir}/new_Q_model/risetime/','rt_3x','mentawai_newQ.mod','sm_short.gflist','standardized'],
#                       [f'{working_dir}/new_Q_model/vrupt/','sf_0.3','mentawai_newQ.mod','sm_short.gflist','standardized'],
#                       [f'{working_dir}/new_Q_model/vrupt/','sf_0.4','mentawai_newQ.mod','sm_short.gflist','standardized'],
#                       [f'{working_dir}/new_Q_model/vrupt/','sf_0.22','mentawai_newQ.mod','sm_short.gflist','standardized']])

parameters_list = np.array([[f'{working_dir}/new_Q_model/new_fault_model/risetime/','rt_2x_test','mentawai_newQ.mod','sm_short.gflist','standardized']])

for i, parameters in enumerate(parameters_list):
    # if i == 0:
    #     ruptures = 'new'
    # else:
    #     ruptures = 'standardized'
    home, project_name, model_name, GF_list, ruptures = parameters
    quick_fq.run_fq(home, project_name, model_name, GF_list, ruptures)
