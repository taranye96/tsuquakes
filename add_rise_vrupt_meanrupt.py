#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:59:33 2023

@author: tnye
"""

import numpy as np
from numpy import load,save,genfromtxt,log10,cos,sin,deg2rad,savetxt,zeros,where
from time import gmtime, strftime
from numpy.random import shuffle
from mudpy import fakequakes
from obspy import UTCDateTime
from obspy.taup import TauPyModel
import geopy.distance
import warnings
from numpy.random import rand,randn,randint
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from numpy import genfromtxt,zeros,arctan2,sin,r_,where,log10,isnan,argmin,setxor1d,exp

target_Mw = 7.84
rise_time_depths=[10,15]
shear_wave_fraction_shallow = 0.49
shear_wave_fraction_deep = 0.8
hypocenter = [100.14, -3.49, 8.82]

fault_name = 'mentawai_5km'

#Read fault and prepare output variable
whole_fault=genfromtxt(f'/Users/tnye/FakeQuakes/files/fault_info/depth_adjusted/{fault_name}.fault')

#Get structure model
vel_mod_file='/Users/tnye/FakeQuakes/files/velmods/mentawai_v1.mod'
vel = np.genfromtxt(vel_mod_file)

#Get TauPyModel
velmod = TauPyModel(model='/Users/tnye/FakeQuakes/files/velmods/mentawai_v1')

#Prepare output
fault_out = np.genfromtxt(f'/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/{fault_name}.rupt')

#Slip pattern sucessfully made, moving on.
#Rigidities
foo,mu=fakequakes.get_mean_slip(target_Mw,whole_fault,vel_mod_file)
fault_out[:,13]=mu

#Calculate moment and magnitude of fake slip pattern
slip = np.sqrt(fault_out[:,8]**2 + fault_out[:,9]**2)
M0=sum(slip*fault_out[:,10]*fault_out[:,11]*mu)
Mw=(2./3)*(log10(M0)-9.1)

#Get stochastic rake vector
stoc_rake=fakequakes.get_stochastic_rake(90,len(slip))

#Calculate and scale rise times
rise_times=fakequakes.get_rise_times(M0,slip,whole_fault,rise_time_depths,stoc_rake,'MH2017')

#Place rise_times in output variable
fault_out[:,7]=rise_times

# Get rupture onset
t_onset=zeros(len(whole_fault))
length2fault=zeros(len(whole_fault))
#Perturb all subfault depths a tiny amount by some random number so that they NEVER lie on a layer interface
z_perturb=(rand(len(whole_fault))-0.5)*1e-6
whole_fault[:,3]=whole_fault[:,3]+z_perturb
# Convert from thickness to depth to top of layer
depth_to_top=r_[0,vel[:,0].cumsum()[0:-1]]
#Get rupture speed shear-wave multipliers
rupture_multiplier=zeros(len(vel))
# Shallow 
i=where(depth_to_top<=rise_time_depths[0])[0]
rupture_multiplier[i]=shear_wave_fraction_shallow
# Deep 
i=where(depth_to_top>=rise_time_depths[1])[0]
rupture_multiplier[i]=shear_wave_fraction_deep
# Transition 
i=where((depth_to_top<rise_time_depths[1]) & (depth_to_top>rise_time_depths[0]))[0]
slope=(shear_wave_fraction_deep-shear_wave_fraction_shallow)/(rise_time_depths[1]-rise_time_depths[0])
intercept=shear_wave_fraction_deep-slope*rise_time_depths[1]
rupture_multiplier[i]=slope*depth_to_top[i]+intercept
for kfault in range(len(whole_fault)): 
    
    D,az,baz=gps2dist_azimuth(hypocenter[1],hypocenter[0],whole_fault[kfault,2],whole_fault[kfault,1])
    D=D/1000
    #Start and stop depths
    if whole_fault[kfault,3]<=hypocenter[2]:
        zshallow=whole_fault[kfault,3]
        zdeep=hypocenter[2]
    else:
        zdeep=whole_fault[kfault,3]
        zshallow=hypocenter[2]
    #Get angle between depths
    theta=arctan2(zdeep-zshallow,D)
    # get hypotenuse distance on all layers
    delta_ray=vel[:,0]/sin(theta)
    # Calculate distance in each layer
    depth1=0
    depth2=vel[0,0]
    length_ray=zeros(len(vel))
    for klayer in range(len(vel)):
        if zshallow>depth1 and zdeep<depth2: #both points in same layer
            length_ray[klayer]=abs(zshallow-zdeep)/sin(theta) 
        elif zshallow>depth1 and zshallow<depth2: #This is the top
            length_ray[klayer]=abs(depth2-zshallow)/sin(theta)
        elif zdeep>depth1 and zdeep<depth2: #This is the bottom
            length_ray[klayer]=abs(depth1-zdeep)/sin(theta)
        elif depth1>zshallow and depth2<zdeep: #Use full layer thickness for ray path length
            length_ray[klayer]=delta_ray[klayer]
        else: #Some other layer, do nothing
            pass
        #Update reference depths
        if klayer<len(vel)-1: #last layer:
            depth1=depth2
            depth2=depth2+vel[klayer+1,0]
        else:
            depth1=depth2
            depth2=1e6
    
    #Now multiply ray path length times rupture velocity
    ray_times=length_ray/(vel[:,1]*rupture_multiplier)
    t_onset[kfault]=ray_times.sum()  
    length2fault[kfault]=(ray_times*vel[:,1]*rupture_multiplier).sum()

#Now perturb onset times according to Graves-Pitarka eq 5 and 6 (assumes 1:1 corelation with slip)
delta_t0=((M0*1e7)**(1./3))*1.8e-9

#GP 2015 extra perturbation to destroy the 1:1 correlation with slip
rand_numb=randn()
sigma_rise_time=0.2
delta_t0=((M0*1e7)**(1./3))*1.8e-9
delta_t=delta_t0*exp(sigma_rise_time*rand_numb)

# #Now apply total perturbation
# slip_average=slip.mean()
# i=where(slip>0.05*slip_average)[0] #perturbation is applied only to subfaults with significant slip
# perturbation=(log10(slip)-log10(slip_average))/(log10(slip.max())-log10(slip_average))
# t_onset_final=t_onset.copy()
# t_onset_final[i]=t_onset[i]-delta_t[i]*perturbation[i]

# #Check for negative times
# i=where(t_onset_final<0)[0]
# t_onset_final[i]=t_onset[i]


fault_out[:,12]=t_onset
fault_out = np.c_[fault_out, length2fault/t_onset]

#Calculate location of moment centroid
centroid_lon,centroid_lat,centroid_z=fakequakes.get_centroid(fault_out)

#Calculate average risetime
rise = fault_out[:,7]
avg_rise = np.mean(rise)

#Write to file
outfile=f'/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/{fault_name}.rupt'
savetxt(outfile,fault_out,fmt='%d\t%10.6f\t%10.6f\t%8.4f\t%7.2f\t%7.2f\t%4.1f\t%.9e\t%.4e\t%.4e\t%10.2f\t%10.2f\t%.9e\t%.6e\t%.6e',header='No,lon,lat,z(km),strike,dip,rise,dura,ss-slip(m),ds-slip(m),ss_len(m),ds_len(m),rupt_time(s),rigidity(Pa),velocity(km/s)')
