#########
# out=mentawai_medium_fault
#########

# ORIGINAL .RUPT FILE
rupt_file=/Users/tnye/FakeQuakes/Mentawai2010/forward_models/Mentawai_HanYuemodel.rupt
awk '{print($2,$3,$4)}' ${rupt_file} > depths.tmp #make tmp file of lon,lat,dep
awk '{print($2,$3,$9)}' ${rupt_file} > strikeslip.tmp #make tmp file of lon,lat,ss
awk '{print($2,$3,$10)}' ${rupt_file} > dipslip.tmp # make tmp file of lon,lat,ds


# get min/max lon/lat from .rupt
rupt_info=`gmt info ${rupt_file} -C`
min_lon=`echo ${rupt_info} | awk '{print($3)}'`
max_lon=`echo ${rupt_info} | awk '{print($4)}'`
min_lat=`echo ${rupt_info} | awk '{print($5)}'`
max_lat=`echo ${rupt_info} | awk '{print($6)}'`

# LOCATION OF NEW SUBFAULTS
fine_fault=/Users/tnye/FakeQuakes/Mentawai2010_fine/data/model_info/mentawai_medium.fault

awk '{print($2,$3)}' ${fine_fault} > finelocations.xy #make tmp file of lon,lat for finer fault grid

# Create interpolated surface
gmt surface strikeslip.tmp -R${min_lon}/${max_lon}/${min_lat}/${max_lat} -I1k -Gss_grd.nc -T0.25
gmt grdtrack finelocations.xy -Gss_grd.nc > strikeslip.xyz
gmt surface dipslip.tmp -R${min_lon}/${max_lon}/${min_lat}/${max_lat} -I1k -Gds_grd.nc -T0.25
gmt grdtrack finelocations.xy -Gds_grd.nc > dipslip.xyz
gmt surface depths.tmp -R${min_lon}/${max_lon}/${min_lat}/${max_lat} -I1k -Gdepths_grd.nc -T0.25
gmt grdtrack finelocations.xy -Gdepths_grd.nc > depths.xyz

paste -d '\t' strikeslip.xyz dipslip.xyz > slip.xyz

#### CREATE NEW FINER RUPT FILE:
final_rupt_file=/Users/tnye/FakeQuakes/Mentawai2010_fine/forward_models/${out}.rupt
echo "Removing Preexisting version of final rupt file"
rm ${final_rupt_file}
T=`wc -l ${fine_fault} | awk '{print($1)}'`
subfault=0
for (( N=1; N<=$T; N++ )); do
	if [ `gmt math -Q ${N} 500 MOD = ` -eq 0 ]; then echo "...subfault ${N}"; fi
	subfault=`gmt math -Q ${subfault} 1 ADD = `
	lon=`head -$N depths.xyz | tail -1 | awk '{print($1)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.6f = `
	lat=`head -$N depths.xyz | tail -1 | awk '{print($2)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.6f = `
	dep=`head -$N depths.xyz | tail -1 | awk '{print($3)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.4f = `
	strike=324.0
	dip=7.5
	tri=0.5
	rise=8.0
	ss_slip=`head -$N strikeslip.xyz | tail -1 | awk '{print($3)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.4f = `
	ds_slip=`head -$N dipslip.xyz | tail -1 | awk '{print($3)}' | gmt math -Q STDIN 1 MUL --FORMAT_FLOAT_OUT=%.4f = `
	len=2000
	wid=2000
	timing=0
	mu=0
	echo "${subfault}	${lon}	${lat}	${dep}	${strike}	${dip}	${tri}	${rise}	${ss_slip}	${ds_slip}	${len}	${wid} ${timing} ${mu}" >> ${final_rupt_file}
done
echo Total subfaults: ${subfault}
edit ${final_rupt_file}
