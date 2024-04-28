#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:15:02 2021

@author: tnye
"""

def view(fault):
    
    import matplotlib
    from numpy import genfromtxt,unique,where,zeros
    import matplotlib.pyplot as plt
    
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 16}
    
    matplotlib.rc('font', **font)
    
    
    f=genfromtxt(fault)
    num=f[:,0]
    # all_ss=f[:,8]
    # all_ds=f[:,9]
   
    #Get other parameters
    lon=f[0:len(num),1]
    lat=f[0:len(num),2]
    strike=f[0:len(num),4]
    
    #Plot
    plt.figure(figsize=(5.2,10))
    plt.scatter(lon,lat,marker='o')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.grid()
    plt.show()
    

fault = '/Users/tnye/Roses/FakeQuakes/files/mentawai.fault'
# fault = '/Users/tnye/Roses/FakeQuakes/files/mentawai_coarse.fault'

view(fault)
