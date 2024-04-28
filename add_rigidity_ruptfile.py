#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:40:55 2023

@author: tnye
"""

# Imports 
import numpy as np

yue_fault = np.genfromtxt('/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/mentawai_fine2.rupt')
rand_fault = np.genfromtxt('/Users/tnye/FakeQuakes/simulations/fault_coarseness/mentawai.000001.rupt')

yue_fault[:,13] = rand_fault[:,13] 

np.savetxt('/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/mentawai_fine2.rupt',yue_fault,fmt='%d\t%10.6f\t%10.6f\t%8.4f\t%7.2f\t%7.2f\t%4.1f\t%5.2f\t%5.2f\t%5.2f\t%10.2f\t%10.2f\t%5.2f\t%.6e',header='No,lon,lat,z(km),strike,dip,rise,dura,ss-slip(m),ds-slip(m),ss_len(m),ds_len(m),rupt_time(s),rigidity(Pa)')




# han_yue = np.genfromtxt('/Users/tnye/FakeQuakes/model_info/han_yue.rupt')
# han_yue[:,3] = han_yue[:,3] - 3

# np.savetxt('/Users/tnye/FakeQuakes/model_info/depth_adjusted/han_yue_depth_adjusted.rupt',han_yue,fmt='%d\t%10.6f\t%10.6f\t%8.4f\t%7.2f\t%7.2f\t%4.1f\t%5.2f\t%5.2f\t%5.2f\t%5.2f\t%.6e',header='No,lon,lat,z(km),strike,dip,rise,dura,ss-slip(m),ds-slip(m),ss_len(m),ds_len(m),rupt_time(s),rigidity(Pa)')


