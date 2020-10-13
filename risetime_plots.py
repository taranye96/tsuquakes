#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:53:43 2020

@author: tnye
"""

###############################################################################
# Script that makes histogram plots of rise time for FakeQuakes runs with
# normal rise time (rise time not multiplied by anything)
###############################################################################

# Imports
from numpy import genfromtxt
import matplotlib.pyplot as plt

#%% First .rupt file

# path to file
rupt1_path = '/Users/tnye/FakeQuakes/parameters/standard/std/disp/output/ruptures/std.000000.rupt'

# open .rupt file and select out risetime col
f1 = open(rupt1_path, 'r')
header1 = f1.readline().rstrip('\n')
rupt1 = genfromtxt(rupt1_path)
rise1 = rupt1[:,7]

# plot histogram 
plt.hist(rise1)
plt.axvline(rise1.mean(), color='k', linestyle='dashed', linewidth=1)
plt.xlabel('rise time (s)')
plt.ylabel('# of subfaults')
plt.title('Rise time distribution for std.000000')

# save fig
plt.savefig('/Users/tnye/tsuquakes/plots/risetime_dist/std.000000.hist.png', dpi=300)
plt.close()

#%% Second .rupt file

# path to file
rupt2_path = '/Users/tnye/FakeQuakes/parameters/standard/std/disp/output/ruptures/std.000001.rupt'

# open .rupt file and select out risetime col
f2 = open(rupt2_path, 'r')
header2 = f2.readline().rstrip('\n')
rupt2 = genfromtxt(rupt2_path)
rise2 = rupt2[:,7]

# plot histogram 
plt.hist(rise2)
plt.axvline(rise2.mean(), color='k', linestyle='dashed', linewidth=1)
plt.xlabel('rise time (s)')
plt.ylabel('# of subfaults')
plt.title('Rise time distribution for std.000001')

# save fig
plt.savefig('/Users/tnye/tsuquakes/plots/risetime_dist/std.000001.hist.png', dpi=300)
plt.close()

#%% Third .rupt file

# path to file
rupt3_path = '/Users/tnye/FakeQuakes/parameters/standard/std/disp/output/ruptures/std.000002.rupt'

# open .rupt file and select out risetime col
f3 = open(rupt3_path, 'r')
header3 = f3.readline().rstrip('\n')
rupt3 = genfromtxt(rupt3_path)
rise3 = rupt3[:,7]

# plot histogram 
plt.hist(rise3)
plt.axvline(rise3.mean(), color='k', linestyle='dashed', linewidth=1)
plt.xlabel('rise time (s)')
plt.ylabel('# of subfaults')
plt.title('Rise time distribution for std.000002')

# save fig
plt.savefig('/Users/tnye/tsuquakes/plots/risetime_dist/std.000002.hist.png', dpi=300)
plt.close()