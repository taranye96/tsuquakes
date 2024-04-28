#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:25:28 2022

@author: dmelgarm
"""

###############################################################################
# Script used to make a PGA file for simulations using various Q-attenuation
# parameters.
###############################################################################


from pyproj import Geod        
from glob import glob
import numpy as np
from obspy import read

home_dir = '/Users/tnye/FakeQuakes/FQ_status/new_Q_model/new_fault_model/m7.85/q_param_test/all_stns'
project_dir = f'{home_dir}/shallowest_150'

#list of ruptures and waveforms
waveforms_folders = np.sort(glob(f'{project_dir}/output/waveforms/mentawai*'))
ruptures = np.sort(glob(f'{project_dir}/output/ruptures/mentawai.*.rupt'))

#stations list
gf_file = '/Users/tnye/tsuquakes/files/random_stns.gflist'
stations = np.genfromtxt(gf_file,usecols=[0],dtype='U')
lon = np.genfromtxt(gf_file,usecols=[1])
lat = np.genfromtxt(gf_file,usecols=[2])

for k in range(len(ruptures)):

    #get rupture
    rupt = np.genfromtxt(ruptures[k])
    print(waveforms_folders[k])
    #keep only those with slip
    i = np.where(rupt[:,12]>0)[0]
    rupt = rupt[i,:]
    
    #get Rrupt
    #projection obnject
    p = Geod(ellps='WGS84')
    
    #lon will have as many rows as Vs30 points and as many columns as subfautls in rupture
    Nsubfaults = len(rupt)
    Nsta = len(lon)
    lon_surface = np.tile(lon,(Nsubfaults,1)).T
    lat_surface = np.tile(lat,(Nsubfaults,1)).T
    lon_subfaults = np.tile(rupt[:,1],(Nsta,1))-360
    lat_subfaults = np.tile(rupt[:,2],(Nsta,1))
    az,baz,dist = p.inv(lon_surface,lat_surface,lon_subfaults,lat_subfaults)
    dist = dist/1000
    
    #get 3D distance
    z = np.tile(rupt[:,3],(Nsta,1))
    xyz_dist = (dist**2 + z**2)**0.5
    Rrupt = xyz_dist.min(axis=1)
    
    #get scalar moment
    M0 = sum(rupt[:,10]*rupt[:,11]*rupt[:,13]*((rupt[:,8]**2+rupt[:,9]**2)**0.5))
    Mw = (2./3)*(np.log10(M0)-9.1)
    
    #get PGA for waveforms
    pga = np.zeros(Nsta)
    pga2 = np.zeros(Nsta)
    for ksta in range(len(stations)):
        
        # n = read(waveforms_folders[ksta]+'/'+stations[ksta]+'.bb.HNN.sac')
        # e = read(waveforms_folders[ksta]+'/'+stations[ksta]+'.bb.HNE.sac')
        n = read(waveforms_folders[k]+'/'+stations[ksta]+'.HNN.mpi.sac')
        e = read(waveforms_folders[k]+'/'+stations[ksta]+'.HNE.mpi.sac')
        nmax = np.max(np.abs(n[0].data))
        emax = np.max(np.abs(e[0].data))
        pga[ksta]=np.max([nmax,emax])
        
        # n = read(waveforms_folders[k]+'/'+stations[ksta]+'.bb.HNN.sac')
        # e = read(waveforms_folders[k]+'/'+stations[ksta]+'.bb.HNE.sac')
        # nmax = np.max(np.abs(n[0].data))
        # emax = np.max(np.abs(e[0].data))
        # pga2[ksta]=np.max([nmax,emax])
        pga2=pga
    
    geo_file = waveforms_folders[k]+'/_ground_motions.txt'
    
    #write to file
    out = np.c_[lon,lat,Rrupt,pga,pga2]
    np.savetxt(geo_file,out,fmt='%.5f,%.5f,%d,%.3f,%.3f',header='lon,lat,Rrupt(km),pga HF (m/s/s),pga BB (m/s/s)')