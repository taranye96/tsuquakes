#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:38:39 2023

@author: tnye
"""

# Imports
import numpy as np
from numpy import genfromtxt,zeros,arctan2,sin,r_,where,log10,isnan,argmin,setxor1d,exp
from numpy.random import rand,randn,randint
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from numpy import nonzero
from mudpy import fakequakes

hypocenter = [100.14, -3.49, 8.82]
shear_wave_fraction_shallow = 0.49
fault = np.genfromtxt('/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/mentawai_fine2.rupt')

rigidity = np.mean(fault[:,13][np.where(np.logical_or(fault[:,8]!=0, fault[:,9]!=0))[0]])
slip = np.sqrt(fault[:,8]**2 + fault[:,9]**2)[np.where(np.logical_or(fault[:,8]!=0, fault[:,9]!=0))[0]]
slip_dura = fault[:,7][np.where(np.logical_or(fault[:,8]!=0, fault[:,9]!=0))[0]]
slip_vel = np.mean(slip)/np.mean(slip_dura)
# slip_vel = np.mean(slip/slip_dura)
vrupt = 1000*np.mean(fault[:,14][np.where(fault[:,12]>0)[0]])
stress = 5e6

avg_vrupt = np.mean(fault[:,14][np.where(np.logical_or(fault[:,8]!=0, fault[:,9]!=0))[0]])
avg_rise = np.mean(slip_dura)

print(f'average vrupt = {avg_vrupt}')
print(f'average rise time = {avg_rise}')
print(avg_vrupt/shear_wave_fraction_shallow)
print('')
print(f'stress/rigidity = {stress/rigidity}')
print(f'Vslip/Vrupt = {slip_vel/vrupt}')

# # Vrupt
# stress_new = 1e6
# slip_vel_new = 0.33
# vrupt_new = slip_vel_new/(stress_new/rigidity)
# print(f'Vrupt = {vrupt_new/1000}')
# print(f'SSF = {vrupt_new/3.3}')


# # Risetime
# stress_new = 1e6
# vrupt_new = 1000
# vslip_new = vrupt_new*stress_new/rigidity
# print(f'Avg Risetime = {vslip_new*np.mean(slip)}')

stress_new = 1e6
vrupt_new = 2000
rise_new = 8
avg_slip = np.mean(slip)
print(f'{stress_new*vrupt_new*rise_new}')
print(f'{avg_slip*rigidity}')
