#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:58:11 2019

@author: tnye
"""

###############################################################################
# Script used to  
###############################################################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import genfromtxt
from mudpy import forward

fout = '/Users/tnye/FakeQuakes/Mentawai2010_fine/data/model_info/mentawai_fine.fault'
strike = 324
dip = 7.5
rise_time = 8
epicenter = [99.805960, -3.348720, 9.079]
nstrike = 78
dx_dip = 2
dx_strike = 2
num_updip = 19
num_downdip = 17
forward.makefault(fout,strike,dip,nstrike,dx_dip,dx_strike,epicenter,num_updip,num_downdip,rise_time)

o = genfromtxt('/Users/tnye/FakeQuakes/Mentawai2010/data/model_info/Mentawai_model.fault')
f = genfromtxt('/Users/tnye/FakeQuakes/Mentawai2010_fine/data/model_info/mentawai_fine.fault')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(f[:,1], f[:,2], -f[:,3], c='blue')
ax.scatter(o[:,1], o[:,2], -o[:,3], c='red')