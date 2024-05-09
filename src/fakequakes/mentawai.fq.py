'''
Parameter file for fakequakes run, with Christine's Hayward discretization

By Diego Melgar (University of Oregon)
Revised by Tara Nye (PhD student at UO)

'''

###############################################################################
# Script that runs through FakeQuakes steps.  Details to fill out the script
# are found here: https://github.com/taranye96/tsuquakes/wiki/FakeQuakes-Script. 
###############################################################################

# Imports
from mudpy import fakequakes,runslip,forward
import numpy as np
from obspy.core import UTCDateTime


########                            GLOBALS                             ########
home='/Users/tnye/FakeQuakes/simulations/'
project_name='hypo_test'
run_name='mentawai'
################################################################################


##############             What do you want to do??           ##################
init=0
make_ruptures=1
make_GFs=0
make_synthetics=0
make_waveforms=0
make_hf_waveforms=0
match_filter=0
# Things that only need to be done once
load_distances=1
G_from_file=0
###############################################################################


#############                 Run-time parameters            ##################
ncpus=4
hot_start=0
model_name='mentawai_v1.mod'   # Velocity model
#velmod_file=home+project_name+'/structure/iquique'
moho_depth_in_km=30.0    #is it ok to have changed this to 30
#fault_name='iquique_gfz.fault'    # Fault geometry
fault_name='mentawai_fine2.fault'
slab_name=None    # Slab 1.0 Ascii file, set to None for simple geometry
mesh_name=None    # GMSH output file, set to None for simple geometry
distances_name='mentawai_fine2' # Name of dist matrix
rupture_list='ruptures.list'
#rupture_list='testruptures.list'
UTM_zone='47M'
scaling_law='T' # T for thrust, S for strike-slip, N for normal

#Station information
GF_list='sm_close.gflist'
G_name='sm'

Nrealizations=2 # Number of fake ruptures to generate per magnitude bin
target_Mw=np.array([7.8])#,7.7]) # Of what approximate magnitudes
max_slip=40 #Maximum sip (m) allowed in the model
max_slip_rule=False

# Displacement and velocity waveform parameters
NFFT=1024 ; dt=0.5    # is this the delta in tr.stats  ### NFFT has to be in powers of 
#fk-parameters
dk=0.1 ; pmin=0 ; pmax=1 ; kmax=20
custom_stf=None

#High frequency waveform parameters
hf_dt=0.01
duration=500
Pwave=True

#Match filter parameters
zero_phase=True
order=4
fcorner_low=0.998
fcorner_high=0.1

# Correlation function parameters
hurst=0.75 # Melgar and Hayes 2019 found Hurst exponent is probably closer to 0.4?
Ldip='auto' # Correlation length scaling, 'auto' uses Mai & Beroza 2002
Lstrike='auto' # MH2019 uses Melgar & Hayes 2019
lognormal=True # Keep this as true
slip_standard_deviation=0.9 # Keep this at 0.9

# Rupture parameters
time_epi=UTCDateTime('2010-10-25T14:42:12Z')
hypocenter=[100.14, -3.49, 8.82] #closest subfault in the finer .rupt to USGS hypo (lowest t_rupt in model)
source_time_function='dreger' # options are 'triangle' or 'cosine' or 'dreger'
stf_falloff_rate=4 #Only affects Dreger STF, choose 4-8 are reasonable values
num_modes=2000 # The more modes, the better you can model the high frequency stuff
stress_parameter=50 #measured in bars
high_stress_depth=30 # SMGA must be below this depth (measured in km)
rake=90 # average rake
rise_time = 'MH2017'
rise_time_depths=[10,15] #Transition depths for rise time scaling (if slip shallower than first index, rise times are twice as long as calculated)
mean_slip_name=home+project_name+'/forward_models/mentawai_fine2.rupt'
shear_wave_fraction_shallow=0.49
shear_wave_fraction_deep=0.8
kappa=None

force_area=True
force_magnitude=True
force_hypocenter=True
###############################################################################



#Initalize project folders
if init==1:
    fakequakes.init(home,project_name)
    
#Generate rupture models
if make_ruptures==1: 
    fakequakes.generate_ruptures(home,project_name,run_name,fault_name,slab_name,mesh_name,load_distances,
        distances_name,UTM_zone,target_Mw,model_name,hurst,Ldip,Lstrike,num_modes,Nrealizations,rake,rise_time,
        rise_time_depths,time_epi,max_slip,source_time_function,lognormal,slip_standard_deviation,scaling_law,
        ncpus,mean_slip_name=mean_slip_name,force_magnitude=force_magnitude,force_area=force_area,
        hypocenter=hypocenter,force_hypocenter=force_hypocenter,shear_wave_fraction_shallow=shear_wave_fraction_shallow,
        shear_wave_fraction_deep=shear_wave_fraction_deep,max_slip_rule=max_slip_rule)  
                
# Prepare waveforms and synthetics       
if make_GFs==1 or make_synthetics==1:
    runslip.inversionGFs(home,project_name,GF_list,None,fault_name,model_name,
        dt,None,NFFT,None,make_GFs,make_synthetics,dk,pmin,
        pmax,kmax,0,time_epi,hot_start,ncpus,custom_stf,impulse=True) 

#Make low frequency waveforms
if make_waveforms==1:
    forward.waveforms_fakequakes(home,project_name,fault_name,rupture_list,GF_list,
                model_name,run_name,dt,NFFT,G_from_file,G_name,source_time_function,
                stf_falloff_rate,hot_start=hot_start)

#Make semistochastic HF waveforms         
if make_hf_waveforms==1:
    forward.hf_waveforms(home,project_name,fault_name,rupture_list,GF_list,
                model_name,run_name,dt,NFFT,G_from_file,G_name,rise_time_depths,
                moho_depth_in_km,ncpus,source_time_function=source_time_function,
                duration=duration,stf_falloff_rate=stf_falloff_rate,hf_dt=hf_dt,
                Pwave=Pwave,hot_start=hot_start,stress_parameter=stress_parameter,
                high_stress_depth=high_stress_depth,kappa=kappa)

# Combine LF and HF waveforms with match filter                              
if match_filter==1:
    forward.match_filter(home,project_name,fault_name,rupture_list,GF_list,
            zero_phase,order,fcorner_low,fcorner_high)
                                                                                                               
