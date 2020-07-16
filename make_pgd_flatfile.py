#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:46:19 2018

@author: vjs
"""

#####    Make PGD flatfile with the event flatfiles   ######

import numpy as np
import pandas as pd
from glob import glob



##### parameters and files   ####
## path to pgd files:
pgd_directory = '/Users/vjs/tsueqs/flatfiles/PGD_flatfiles'

## Events file:
events_path = '/Users/vjs/tsueqs/data/events.csv'

## Refined flatfile (for reference):
refined_accel_flatfile = '/Users/vjs/tsueqs/flatfiles/refined_subset.csv'

## Output PGD flatfile:
output_pgd_flatfile = '/Users/vjs/tsueqs/flatfiles/pgd_flatfile.csv'

## Output PGD reference GMPE file:
output_refpgd_flatfile = '/Users/vjs/tsueqs/flatfiles/reference_pgd.csv'

## reference rhypo and M:
ref_rhypo = np.linspace(5,550,60)
ref_M = np.array([7.,8.,9.])

### GMPE coefficients, from Ruhl et al. 2018 - [A, B, C]:
#pgd_coefficients = np.array([-5.919, 1.009, -0.145])     ## Ruhl 2018
pgd_coefficients = np.array([-4.434, 1.047, -0.138])      ## Melgar 2015


#################################

## Read in the events file:
events_data = pd.read_csv(events_path)

## Read in the refined file:
refined_accel_data = pd.read_csv(refined_accel_flatfile)

## Glob file with all the pgd flatfiles:
pgd_glob = np.array(glob(pgd_directory + '/*'))
glob_events = np.array([])

## Get the even tnames from the globbed files:
for i_file in range(len(pgd_glob)):
    filename_i = pgd_glob[i_file]
    eventname_i = filename_i.split('/')[-1].split('.')[0]
    glob_events = np.append(glob_events,eventname_i)

## Unioque events in the refined file:
refined_events, uniqueind = np.unique(refined_accel_data.eventname.values,return_index=True)
refined_lon = refined_accel_data.hypolon[uniqueind].values
refined_lat = refined_accel_data.hypolat[uniqueind].values
refined_depth = refined_accel_data.hypodepth[uniqueind].values
refined_mw = refined_accel_data.mw[uniqueind].values
refined_m0 = refined_accel_data.m0[uniqueind].values
refined_mw_pgd = refined_accel_data.mw_pgd[uniqueind].values
refined_m0_pgd = refined_accel_data.m0_pgd[uniqueind].values

## Get the pre-dfined arrays for pgd dataset:
eventname = np.array([])
evlon = np.array([])
evlat = np.array([])
evdepth = np.array([])
mw = np.array([])
m0 = np.array([])
mw_pgd = np.array([])
m0_pgd = np.array([])
pgd = np.array([])
rhypo = np.array([])
pgd = np.array([])

## For each event pgd file, read it in:
for i_event in range(len(refined_events)):
    event_i = refined_events[i_event]
    evlon_i = refined_lon[i_event]
    evlat_i = refined_lat[i_event]
    evdepth_i = refined_depth[i_event]
    mw_i = refined_mw[i_event]
    m0_i = refined_m0[i_event]
    mw_pgd_i = refined_mw_pgd[i_event]
    m0_pgd_i = refined_m0_pgd[i_event]
    
    ## Find where in the glob file this event exists:
    i_glob_index = np.where(glob_events == event_i)[0]
    if len(i_glob_index) > 0:
        ## Get the pgd and rhypo:
        i_pgddata = np.genfromtxt(pgd_glob[i_glob_index[0]],skip_header=0)
        #i_pgddata = pd.read_csv(pgd_glob[i_glob_index[0]],names=['rhypo','pgd'],skiprows=[0],delimiter='\t')
        i_rhypo = i_pgddata[:,0]
        i_pgd = i_pgddata[:,1]
        
        ## Append these on to the overall arrays:
        eventname = np.append(eventname,np.full(len(i_rhypo),event_i))
        evlon = np.append(evlon,np.full_like(i_rhypo,evlon_i))
        evlat = np.append(evlat,np.full_like(i_rhypo,evlat_i))
        evdepth = np.append(evdepth,np.full_like(i_rhypo,evdepth_i))
        mw = np.append(mw,np.full_like(i_rhypo,mw_i))
        m0 = np.append(m0,np.full_like(i_rhypo,m0_i))
        mw_pgd = np.append(mw_pgd,np.full_like(i_rhypo,mw_pgd_i))
        m0_pgd = np.append(m0_pgd,np.full_like(i_rhypo,m0_pgd_i))
        
        ## convert pgd to m from cm:
        i_pgd = i_pgd * 1e-2
        
        ## Append:
        rhypo = np.append(rhypo,i_rhypo)
        pgd = np.append(pgd,(i_pgd))
 

### Now get PGD GMPE predictions - first for reference data:
##  PGD in m is: log(PGD) = A + BM + CMlog(R)
ref_pgd = np.zeros((len(ref_rhypo),len(ref_M)))
for i_refM in range(len(ref_M)):
    i_pgdlog10 = pgd_coefficients[0] + pgd_coefficients[1]*ref_M[i_refM] + pgd_coefficients[2]*ref_M[i_refM]*np.log10(ref_rhypo)
    #i_pgd = 10**i_pgdlog10
    i_pgd = (10**(i_pgdlog10)) * 1e-2
    
    ref_pgd[:,i_refM] = i_pgd
    
## Now for each individual recording...
pgd_melgar2015_log10 = pgd_coefficients[0] + pgd_coefficients[1]*mw + pgd_coefficients[2]*mw*np.log10(rhypo)
#pgd_ruhl2018 = 10**(pgd_ruhl2018_log10)
pgd_melgar2015 = (10**(pgd_melgar2015_log10)) * 1e-2

## Get the residual - in ln space, observed / predicted
residual_melgar2015 = np.log(pgd/pgd_melgar2015)

    
#############################
## Make a dict with these:
pgd_dict = {'eventname':eventname, 'hypolon':evlon, 'hypolat':evlat, 'hypdepth':evdepth,
            'mw':mw, 'm0':m0, 'mw_pgd':mw_pgd, 'm0_pgd':m0_pgd, 'rhypo':rhypo, 'pgd_meters':pgd,
            'pgd_melgar2015':pgd_melgar2015, 'residual_melgar2015':residual_melgar2015}

ref_pgdgmpe_dict = {'ref_rhypo':ref_rhypo, 'ref_pgd_melgar2015_M7':ref_pgd[:,0], 
                    'ref_pgd_melgar2015_M8':ref_pgd[:,1],
                    'ref_pgd_melgar2015_M9':ref_pgd[:,2]}


## Put into a dataframe:
pgd_df = pd.DataFrame(pgd_dict)
ref_pgd_df = pd.DataFrame(ref_pgdgmpe_dict)

## Sample the pgd data frame so it's only out to 500km distance:
pgd_df = pgd_df[pgd_df.rhypo <= 500]

## Svae to a file:
pgd_df.to_csv(output_pgd_flatfile,index=False)
ref_pgd_df.to_csv(output_refpgd_flatfile,index=False)