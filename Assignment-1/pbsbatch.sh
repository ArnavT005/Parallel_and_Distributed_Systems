#!/bin/sh
### Set the job name (for your reference)
#PBS -N A1
### Set the project name, your department code by default
#PBS -P col380.cs1190424
### Request email when job begins and ends, don't change anything on the below line 
#PBS -m bea
### Specify email address to use for notification, don't change anything on the below line
#PBS -M $USER@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=1:ncpus=24
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=00:30:00
#PBS -l software=PYTHON

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
cd $SCRATCH
module load compiler/gcc/9.1/openmpi/4.0.2

./main input_100000.txt 1
./main input_100000.txt 2
./main input_100000.txt 4
./main input_100000.txt 6
./main input_100000.txt 8
./main input_100000.txt 10
./main input_100000.txt 12
./main input_100000.txt 14
./main input_100000.txt 16
./main input_100000.txt 18
./main input_100000.txt 20
./main input_100000.txt 22
./main input_100000.txt 24

./main input_1000000.txt 1
./main input_1000000.txt 2
./main input_1000000.txt 4
./main input_1000000.txt 6
./main input_1000000.txt 8
./main input_1000000.txt 10
./main input_1000000.txt 12
./main input_1000000.txt 14
./main input_1000000.txt 16
./main input_1000000.txt 18
./main input_1000000.txt 20
./main input_1000000.txt 22
./main input_1000000.txt 24

./main input_10000000.txt 1
./main input_10000000.txt 2
./main input_10000000.txt 4
./main input_10000000.txt 6
./main input_10000000.txt 8
./main input_10000000.txt 10
./main input_10000000.txt 12
./main input_10000000.txt 14
./main input_10000000.txt 16
./main input_10000000.txt 18
./main input_10000000.txt 20
./main input_10000000.txt 22
./main input_10000000.txt 24

./main input_100000000.txt 1
./main input_100000000.txt 2
./main input_100000000.txt 4
./main input_100000000.txt 6
./main input_100000000.txt 8
./main input_100000000.txt 10
./main input_100000000.txt 12
./main input_100000000.txt 14
./main input_100000000.txt 16
./main input_100000000.txt 18
./main input_100000000.txt 20
./main input_100000000.txt 22
./main input_100000000.txt 24

exit