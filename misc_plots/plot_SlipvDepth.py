#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:14:30 2021

@author: tnye
"""

###############################################################################
# Script that makes a plot comparing the depth of the peak slip between the
# Han Yue mean model and my synthetic models. 
###############################################################################

# Imports
import numpy as np
from glob import glob
from os import path, makedirs
import matplotlib.pyplot as plt
# from pylab import plot, show, savefig, xlim, figure, \
#                 hold, ylim, legend, boxplot, setp, axes

project = 'newq_m7.8'
files = sorted(glob(f'/Users/tnye/FakeQuakes/parameters/standard/{project}/disp/output/ruptures/*.rupt'))[:4]
# m785_files = sorted(glob('/Users/tnye/FakeQuakes/parameters/test/new_fault/disp/output/ruptures/*.rupt'))[4:]
han_yue = np.genfromtxt('/Users/tnye/tsuquakes/files/mentawai_fine2.rupt')

if not path.exists(f'/Users/tnye/FakeQuakes/parameters/standard/{project}/plots/misc'):
    makedirs(f'/Users/tnye/FakeQuakes/parameters/standard/{project}/plots/misc')

depth_list = []
slip_list = []

# depths = [4,5,6,7,8,9,10,11,12,13,14]
depths = [1,2,3,4,5,6,7,8,9,10,11]
data = [[] for _ in range(10)]

for file in files:
    
    # Read in rupt file
    rupt = np.genfromtxt(file)
    
    # depths = np.unique(rupt[:,3])
    
    for i in range(len(depths)):
        if i > 0:
            ind = np.where((rupt[:,3]>depths[i-1]) & (rupt[:,3]<depths[i]))[0]
        
            # Get total slip
            ss_slip = rupt[:,8][ind]
            ds_slip = rupt[:,9][ind]
            slip = np.sqrt(ss_slip**2 + ds_slip**2)
        
            depth_list.append(depths[i-1])
            data[i-1].append(np.max(slip))
            slip_list.append(np.max(slip))

# for file in m784_files:
    
#     # Read in rupt file
#     rupt = np.genfromtxt(file)
    
#     # depths = np.unique(rupt[:,3])
    
#     for i in range(len(depths)):
#         if i > 0:
#             ind = np.where((rupt[:,3]>depths[i-1]) & (rupt[:,3]<depths[i]))[0]
        
#             # Get total slip
#             ss_slip = rupt[:,8][ind]
#             ds_slip = rupt[:,9][ind]
#             slip = np.sqrt(ss_slip**2 + ds_slip**2)
#             data2[i-1].append(np.max(slip))


han_yue_depth = []
han_yue_slip = []
for i in range(len(depths)):
        if i > 0:
            ind = np.where((han_yue[:,3]>depths[i-1]) & (han_yue[:,3]<depths[i]))[0]
        
            # Get total slip
            ss_slip = han_yue[:,8][ind]
            ds_slip = han_yue[:,9][ind]
            slip = np.sqrt(ss_slip**2 + ds_slip**2)
        
            han_yue_depth.append(depths[i-1])
            han_yue_slip.append(np.max(slip))


fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.scatter(depth_list, slip_list, s=30, label='Synthetic')
ax.boxplot(data)
ax.scatter(han_yue_depth, han_yue_slip, s=30, label='Han Yue')
ax.set_xlabel('Depth (km)')
ax.set_ylabel('Peak Slip (m)')
plt.legend()
plt.show()
plt.savefig('/Users/tnye/tsuquakes/plots/misc/PeakSlip_vs_Depth.png', dpi=300)
plt.close()
    



# # function for setting the colors of the box plots pairs
# def setBoxColors(bp):
#     setp(bp['boxes'][0], color='blue')
#     setp(bp['caps'][0], color='blue')
#     setp(bp['caps'][1], color='blue')
#     setp(bp['whiskers'][0], color='blue')
#     setp(bp['whiskers'][1], color='blue')
#     setp(bp['fliers'][0], color='blue')
#     setp(bp['fliers'][1], color='blue')
#     setp(bp['medians'][0], color='blue')

#     setp(bp['boxes'][1], color='red')
#     setp(bp['caps'][2], color='red')
#     setp(bp['caps'][3], color='red')
#     setp(bp['whiskers'][2], color='red')
#     setp(bp['whiskers'][3], color='red')
#     setp(bp['fliers'][2], color='red')
#     setp(bp['fliers'][3], color='red')
#     setp(bp['medians'][1], color='red')

# # Some fake data to plot
# A= [[1, 2, 5,],  [7, 2]]
# B = [[5, 7, 2, 2, 5], [7, 2, 5]]
# C = [[3,2,5,7], [6, 7, 3]]

# fig = figure()
# ax = axes()
# hold(True)

# # first boxplot pair
# bp = boxplot(A, positions = [1, 2], widths = 0.6)
# setBoxColors(bp)

# # second boxplot pair
# bp = boxplot(B, positions = [4, 5], widths = 0.6)
# setBoxColors(bp)

# # thrid boxplot pair
# bp = boxplot(C, positions = [7, 8], widths = 0.6)
# setBoxColors(bp)

# # set axes limits and labels
# xlim(0,9)
# ylim(0,9)
# ax.set_xticklabels(['A', 'B', 'C'])
# ax.set_xticks([1.5, 4.5, 7.5])

# # draw temporary red and blue lines and use them to create a legend
# hB, = plot([1,1],'b-')
# hR, = plot([1,1],'r-')
# legend((hB, hR),('Apples', 'Oranges'))
# hB.set_visible(False)
# hR.set_visible(False)

# savefig('boxcompare.png')
# show()
