#!/bin/bash

#########
sub_len=2.5 #in km
out=mentawai_${sub_len}km

outdir=/Users/tnye/tsuquakes/code/GMT/output
#########

# (depth adjsuted) ORIGINAL .RUPT FILE
rupt_file=/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/han_yue_depth_adjusted.txt
awk '{print($2,$3,$4)}' ${rupt_file} > ${outdir}/hanyue_depths.tmp #make tmp file of lon,lat,dep
awk '{print($2,$3,$9)}' ${rupt_file} > ${outdir}/hanyue_strikeslip.tmp #make tmp file of lon,lat,ss
awk '{print($2,$3,$10)}' ${rupt_file} > ${outdir}/hanyue_dipslip.tmp # make tmp file of lon,lat,ds


# get min/max lon/lat from .rupt
rupt_info=`gmt info ${rupt_file} -C`
min_lon=`echo ${rupt_info} | awk '{print($3)}'`
max_lon=`echo ${rupt_info} | awk '{print($4)}'`
min_lat=`echo ${rupt_info} | awk '{print($5)}'`
max_lat=`echo ${rupt_info} | awk '{print($6)}'`

# LOCATION OF NEW SUBFAULTS
fine_fault=/Users/tnye/FakeQuakes/files/fault_info/mentawai_${sub_len}km.fault

awk '{print($2,$3)}' ${fine_fault} > ${outdir}/locations_${sub_len}km.xy #make tmp file of lon,lat for finer fault grid

# Create interpolated surface
gmt surface ${outdir}/hanyue_strikeslip.tmp -R${min_lon}/${max_lon}/${min_lat}/${max_lat} -I1k -Gss_grd.nc -T0.25
gmt grdtrack ${outdir}/locations_${sub_len}km.xy -Gss_grd.nc > ${outdir}/strikeslip_${sub_len}km.xyz
gmt surface ${outdir}/hanyue_dipslip.tmp -R${min_lon}/${max_lon}/${min_lat}/${max_lat} -I1k -Gds_grd.nc -T0.25
gmt grdtrack ${outdir}/locations_${sub_len}km.xy -Gds_grd.nc > ${outdir}/dipslip_${sub_len}km.xyz
gmt surface ${outdir}/hanyue_depths.tmp -R${min_lon}/${max_lon}/${min_lat}/${max_lat} -I1k -Gdepths_grd.nc -T0.25
gmt grdtrack ${outdir}/locations_${sub_len}km.xy -Gdepths_grd.nc > ${outdir}/depths_${sub_len}km.xyz

paste -d '\t' ${outdir}/strikeslip_${sub_len}km.xyz ${outdir}/dipslip_${sub_len}km.xyz > ${outdir}/slip_${sub_len}km.xyz

#### CREATE NEW FINER RUPT FILE:
final_rupt_file=/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/${out}.rupt
echo "Removing Preexisting version of final rupt file"
rm ${final_rupt_file}
T=`wc -l ${fine_fault} | awk '{print($1)}'`
subfault=0
for (( N=1; N<=$T; N++ )); do
	if [ `gmt math -Q ${N} 500 MOD = ` -eq 0 ]; then echo "...subfault ${N}"; fi
	subfault=`gmt math -Q ${subfault} 1 ADD = `
	lon=`head -$N ${outdir}/depths_${sub_len}km.xyz | tail -1 | awk '{print($1)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.6f = `
	lat=`head -$N ${outdir}/depths_${sub_len}km.xyz | tail -1 | awk '{print($2)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.6f = `
	dep=`head -$N ${outdir}/depths_${sub_len}km.xyz | tail -1 | awk '{print($3)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.4f = `
	strike=324.0
	dip=7.5
	tri=0.5
	rise=8.0
	ss_slip=`head -$N ${outdir}/strikeslip_${sub_len}km.xyz | tail -1 | awk '{print($3)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.4f = `
	ds_slip=`head -$N ${outdir}/dipslip_${sub_len}km.xyz | tail -1 | awk '{print($3)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.4f = `
	len=$(echo "$sub_len*1000" | bc)
	wid=$(echo "$sub_len*1000" | bc)
	# len=$((${sub_len} * 1000))
	# wid=$((${sub_len} * 1000))
	# echo "subfault_len=$subfault_len"
	# echo "len=$(echo "$subfault_len*1000" | bc)"
	# echo "wid=$(echo "$subfault_len*1000" | bc)"
	timing=0
	mu=0
	echo "${subfault}	${lon}	${lat}	${dep}	${strike}	${dip}	${tri}	${rise}	${ss_slip}	${ds_slip}	${len}	${wid} ${timing} ${mu}" >> ${final_rupt_file}
done
echo Total subfaults: ${subfault}
# edit ${final_rupt_file}
