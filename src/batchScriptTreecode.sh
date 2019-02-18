####  PBS preamble

#PBS -N treecodeTesting
#PBS -M njvaughn@umich.edu
#PBS -m a

#PBS -A krasny_fluxg
#PBS -l qos=flux
#PBS -q fluxg

 
#PBS -l nodes=1:gpus=1:titanv,mem=8gb
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -V

####  End PBS preamble

if [ -s "$PBS_NODEFILE" ] ; then
    echo "Running on"
    uniq -c $PBS_NODEFILE
fi

if [ -d "$PBS_O_WORKDIR" ] ; then
    cd $PBS_O_WORKDIR
    echo "Running from $PBS_O_WORKDIR"
fi

cd /home/njvaughn/hybrid-gpu-treecode/src


#module purge
#module load cuda cupti pgi openmpi/1.10.2/pgi/16.4 mkl/2017.0.1

pgititan
nvidia-smi
export PGI_ACC_TIME=0


SOURCES=/scratch/krasny_fluxg/njvaughn/examplesOxygenAtom/S636608.bin
TARGETS=/scratch/krasny_fluxg/njvaughn/examplesOxygenAtom/T636608.bin
NUMSOURCES=636608
NUMTARGETS=636608
TREETYPE=1

SFLAG=1
PFLAG=0
DFLAG=0

#OUTFILE=out636608_yukawa0p5_versus_Coulomb.csv
OUTFILE=out636608_titanTimingTests_coulombSS.csv



#../bin/tree.exe   $SOURCES $TARGETS $DIRECTSUM $OUTFILE $NUMSOURCES $NUMTARGETS 0.7 8 $TREETYPE 8000 $KAPPA $POTENTIALTYPE $SFLAG $PFLAG $DFLAG 8000
#../bin/tree.exe   $SOURCES $TARGETS $DIRECTSUM $OUTFILE $NUMSOURCES $NUMTARGETS 0.8 8 $TREETYPE 8000 $KAPPA $POTENTIALTYPE $SFLAG $PFLAG $DFLAG 8000

#../binTitan/tree.exe   $SOURCES $TARGETS $DIRECTSUM $OUTFILE $NUMSOURCES $NUMTARGETS 0.7 8 $TREETYPE 8000 $KAPPA $POTENTIALTYPE $SFLAG $PFLAG $DFLAG 8000
#../binTitan/tree.exe   $SOURCES $TARGETS $DIRECTSUM $OUTFILE $NUMSOURCES $NUMTARGETS 0.8 8 $TREETYPE 8000 $KAPPA $POTENTIALTYPE $SFLAG $PFLAG $DFLAG 8000

DIRECTSUM=/scratch/krasny_fluxg/njvaughn/examplesOxygenAtom/ex_st636608_coulomb_titan.bin
KAPPA=0.5
POTENTIALTYPE=2

for BATCHSIZE in 8000
do
	for MAXPARNODE in 8000 16000
	  do
		for ORDER in 5 6 7 8 9
		  do 
		     for THETA in 0.6 0.7 0.8 0.9
		     	do
		     		../binTitan/tree.exe   $SOURCES $TARGETS $DIRECTSUM $OUTFILE $NUMSOURCES $NUMTARGETS $THETA $ORDER $TREETYPE $MAXPARNODE $KAPPA $POTENTIALTYPE $SFLAG $PFLAG $DFLAG $BATCHSIZE
		     done
		 done
	done
done
