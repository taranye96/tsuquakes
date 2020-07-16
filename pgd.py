#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:23:42 2019

@author: tnye
"""

#####    Make PGD flatfile with the event flatfiles   ######

# Imports
import numpy as np
import pandas as pd

##### parameters and files   ####

## Refined flatfile (for reference):
refined_accel_flatfile = '/Users/tnye/tsuquakes/data/refined_subset.csv'

## Output PGD flatfile:
output_pgd_flatfile = '/Users/tnye/tsuquakes/flatfiles/pgd_flatfile.csv'

#################################


## Read in the refined file:
refined_accel_data = pd.read_csv(refined_accel_flatfile)

## Get file with all the pgd data
pgd_file = '/Users/tnye/tsuquakes/data/Mentawai2010/Mentawai2010.obs.pgd'
eventname = pgd_file.split('/')[-1].split('.')[0]


# Select out Mentawai data
event_sel = refined_accel_data.eventname == eventname
mentawai_data = refined_accel_data[event_sel]

## Get the pre-defined arrays for pgd dataset:
eventname = np.unique(mentawai_data.eventname)[0]
evlon = np.unique(mentawai_data.hypolon)[0]
evlat = np.unique(mentawai_data.hypolat)[0]
evdepth = np.unique(mentawai_data.hypodepth)[0]
station = np.unique(mentawai_data.station)
mw = np.unique(mentawai_data.mw)[0]
m0 = np.unique(mentawai_data.m0)[0]
mw_pgd = np.unique(mentawai_data.mw_pgd)[0]
m0_pgd = np.unique(mentawai_data.m0_pgd)[0]

## Find where in the glob file this event exists:
## Get the pgd and hyodist:
pgd_data = np.genfromtxt(pgd_file,skip_header=0)
hypdist = pgd_data[:,0]
pgd = pgd_data[:,1]

## convert pgd from cm to m:
pgd = pgd * 1e-2

#############################
## Make a dict with these:
pgd_dict = {'eventname':eventname, 'hyplon':evlon, 'hyplat':evlat, 'hypdepth':evdepth,
            'mw':mw, 'm0':m0, 'mw_pgd':mw_pgd, 'm0_pgd':m0_pgd, 'hypdist':hypdist,
            'pgd_meters':pgd}


## Put into a dataframe:
pgd_df = pd.DataFrame(pgd_dict)

## Sample the pgd data frame so it's only out to 500km distance:
pgd_df = pgd_df[pgd_df.hypdist <= 500]

## Save to a file:
pgd_df.to_csv(output_pgd_flatfile,index=False)
