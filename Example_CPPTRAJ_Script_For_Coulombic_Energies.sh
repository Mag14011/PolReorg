#!/bin/bash

D=1274
A=1277
Dprmtop="/workhorse/OmcS_EField/r${D}/omcs_r${D}.prmtop"
  Dtraj="/workhorse/OmcS_EField/r${D}/prod_${D}centered.nc"
Aprmtop="/workhorse/OmcS_EField/r${A}/omcs_r${A}.prmtop"
  Atraj="/workhorse/OmcS_EField/r${A}/prod_${A}centered.nc"

echo "
 For the forward reaction:
   Donor               = ${D}
   Acceptor            = ${A}

   Donor Topology      = ${Dprmtop}
   Donor Trajectory    = ${Dtraj}

   Acceptor Topology   = ${Aprmtop}
   Acceptor Trajectory = ${Atraj}
"

rm Reorg.cpptraj 2> /dev/null
cat >> Reorg.cpptraj <<-EOF

#==================================================================================
kb= 0.00008617333262145
T= 300
#==================================================================================
parm   ${Dprmtop}
trajin ${Dtraj} 1 last 1 parmindex 0

esander I${D} out I${D}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

strip !(:HEH&:${D}) outprefix DAsubsysA${D}
esander DADAsubsysA${D} out DAsubsysA${D}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

unstrip

strip !(:HEH&:${A}) outprefix DAsubsysA${A}
esander DADAsubsysA${A} out DAsubsysA${A}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

unstrip

sysA = I${D}[elec]+I${D}[elec14]
subAA = DADAsubsysA${D}[elec]+DADAsubsysA${D}[elec14]
subAB = DADAsubsysA${A}[elec]+DADAsubsysA${A}[elec14]
PairIntA = (sysA - subAA - subAB)
writedata If.dat sysA subAAb subAB PairIntA
run
#----------------------------------------------------------------------------------

clear trajin 

#----------------------------------------------------------------------------------
parm   ${Aprmtop}
trajin ${Dtraj} 1 last 1 parmindex 1

esander F${A} out F${A}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

strip !(:HEH&:${D}) outprefix DAsubsysB${D}
esander DADAsubsysB${D} out DAsubsysB${D}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

unstrip

strip !(:HEH&:${A}) outprefix DAsubsysB${A}
esander DADAsubsysB${A} out DAsubsysB${A}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

unstrip

sysB = F${A}[elec]+F${A}[elec14]
subBA = DADAsubsysB${D}[elec]+DADAsubsysB${D}[elec14]
subBB = DADAsubsysB${A}[elec]+DADAsubsysB${A}[elec14]
PairIntB = (sysB - subBA - subBB)
writedata Ib.dat sysB subBA subBB PairIntB
run

VEGf = (PairIntB - PairIntA) * 0.0434
writedata VEGf.dat PairIntA PairIntB VEGf
#==================================================================================

clear trajin 

#==================================================================================
trajin ${Atraj} 1 last 1 parmindex 1

esander I${A} out I${A}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

strip !(:HEH&:${D}) outprefix DAsubsysC${D}
esander DADAsubsysC${D} out DAsubsysC${D}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

unstrip

strip !(:HEH&:${A}) outprefix DAsubsysC${A}
esander DADAsubsysC${A} out DAsubsysC${A}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

unstrip

sysC = I${A}[elec]+I${A}[elec14] 
subCA = DADAsubsysC${D}[elec]+DADAsubsysC${D}[elec14]
subCB = DADAsubsysC${A}[elec]+DADAsubsysC${A}[elec14]
PairIntC = (sysC - subCA - subCB)
writedata Ff.dat sysC subCA subCB PairIntC
run
#----------------------------------------------------------------------------------

clear trajin 

#----------------------------------------------------------------------------------
trajin ${Atraj} 1 last 1 parmindex 0

esander F${D} out F${D}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

strip !(:HEH&:${D}) outprefix DAsubsysD${D}
esander DADAsubsysD${D} out DAsubsysD${D}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

unstrip

strip !(:HEH&:${A}) outprefix DAsubsysD${A}
esander DADAsubsysD${A} out DAsubsysD${A}.dat cut 10.0 igb 0 ntb 1 ntf 2 ntc 2
run

unstrip

sysD = F${D}[elec]+F${D}[elec14]
subDA = DADAsubsysD${D}[elec]+DADAsubsysD${D}[elec14]
subDB = DADAsubsysD${A}[elec]+DADAsubsysD${A}[elec14]
PairIntD = (sysD - subDA - subDB)
writedata Fb.dat sysD subDA subDB PairIntD
run

VEGb = (PairIntC - PairIntD) * 0.0434 
writedata VEGb.dat PairIntC PairIntD VEGb
#==================================================================================

avgVEGf   = avg(VEGf)
avgVEGb   = avg(VEGb)
lambdast  = (avgVEGf - avgVEGb)/2

varVEGf   = stdev(VEGf)^2
varVEGb   = stdev(VEGb)^2
lambdavarf = varVEGf/(2*kb*T)
lambdavarb = varVEGb/(2*kb*T)

xg = (lambdavarf + lambdavarb)/(2*lambdast)
 
printdata avgVEGf avgVEGb varVEGf varVEGb
printdata lambdast lambdavarf lambdavarb xg
writedata Reorg${D},${A}.dat lambdast lambdavarf lambdavarb xg
run

quit
EOF

echo -n " Submit for MPI run? "
read SubChoice

if [ $SubChoice == "YES" ] || [ $SubChoice == "Yes" ] || [ $SubChoice == "yes" ] || [ $SubChoice == "Y" ] || [ $SubChoice == "y" ];then
  echo -n "   How many procs? "
  read nproc

  mpirun -np ${nproc} cpptraj.MPI -i Reorg.cpptraj > Reorg.log &
else
  echo -n " Not submitted."
fi


