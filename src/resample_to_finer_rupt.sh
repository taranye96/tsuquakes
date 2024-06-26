#!/bin/bash

# Paths to finer .fault file and to outpath of finer .rupt file
fine_fault=$1
fine_rupt=$2

# Original .rupt file (adjusted for the 3km water layer subtracted)
rupt_file=/Users/tnye/FakeQuakes/files/model_info/depth_adjusted/han_yue_depth_adjusted.rupt

awk '{print($2,$3,$4)}' ${rupt_file} > depths.tmp #make tmp file of lon,lat,dep
awk '{print($2,$3,$9)}' ${rupt_file} > strikeslip.tmp #make tmp file of lon,lat,ss
awk '{print($2,$3,$10)}' ${rupt_file} > dipslip.tmp # make tmp file of lon,lat,ds

awk '{print($2,$3)}' ${fine_fault} > finelocations.xy #make tmp file of lon,lat for finer fault grid


# get min/max lon/lat from .rupt
rupt_info=`gmt info ${rupt_file} -C`
min_lon=`echo ${rupt_info} | awk '{print($3)}'`
max_lon=`echo ${rupt_info} | awk '{print($4)}'`
min_lat=`echo ${rupt_info} | awk '{print($5)}'`
max_lat=`echo ${rupt_info} | awk '{print($6)}'`

echo $max_lat

# # Create interpolated surface
# gmt surface strikeslip.tmp -R${min_lon}/${max_lon}/${min_lat}/${max_lat} -I2k -Gss_grd.nc -T0.25
# gmt grdtrack finelocations.xy -Gss_grd.nc > strikeslip.xyz
# gmt surface dipslip.tmp -R${min_lon}/${max_lon}/${min_lat}/${max_lat} -I2k -Gds_grd.nc -T0.25
# gmt grdtrack finelocations.xy -Gds_grd.nc > dipslip.xyz
# gmt surface depths.tmp -R${min_lon}/${max_lon}/${min_lat}/${max_lat} -I2k -Gdepths_grd.nc -T0.25
# gmt grdtrack finelocations.xy -Gdepths_grd.nc > depths.xyz
#
# paste -d '\t' strikeslip.xyz dipslip.xyz > slip.xyz
#
#### CREATE NEW FINER RUPT FILE:
# echo "Removing Preexisting version of final rupt file"
# rm ${fine_rupt}
T=`wc -l ${fine_fault} | awk '{print($1)}'`
subfault=0
for (( N=1; N<=$T; N++ )); do
	if [ `gmt math -Q ${N} 500 MOD = ` -eq 0 ]; then echo "...subfault ${N}"; fi
	subfault=`gmt math -Q ${subfault} 1 ADD = `
	lon=`head -$N depths.xyz | tail -1 | awk '{print($1)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.6f = `
	lat=`head -$N depths.xyz | tail -1 | awk '{print($2)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.6f = `
	dep=`head -$N depths.xyz | tail -1 | awk '{print($3)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.4f = `
	strike=`head -$N ${fine_fault} | tail -1 | awk '{print($5)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.1f = `
	dip=`head -$N ${fine_fault} | tail -1 | awk '{print($6)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.1f = `
	tri=`head -$N ${fine_fault} | tail -1 | awk '{print($7)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.1f = `
	rise=`head -$N ${fine_fault} | tail -1 | awk '{print($8)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.1f = `
	ss_slip=`head -$N strikeslip.xyz | tail -1 | awk '{print($3)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%E = `
	ds_slip=`head -$N dipslip.xyz | tail -1 | awk '{print($3)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%E = `
	len=`head -$N ${fine_fault} | tail -1 | awk '{print($9)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.1f = `
	wid=`head -$N ${fine_fault} | tail -1 | awk '{print($10)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.1f = `
	timing=0
	mu=0
	echo "${subfault}	${lon}	${lat}	${dep}	${strike}	${dip}	${tri}	${rise}	${ss_slip}	${ds_slip}	${len}	${wid} ${timing} ${mu}" >> ${fine_rupt}
done
echo Total subfaults: ${subfault}
edit ${fine_rupt}
