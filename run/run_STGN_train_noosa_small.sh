#!/bin/bash -l

#PBS -N STGN_train_noosa_small

#PBS -l walltime=47:59:59
#PBS -l mem=32g
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -l gputype=A100

#PBS -o ~/SLSQ_Project/STGN-SLSQ/out/STGN_train_noosa_small_stdout.out
#PBS -e ~/SLSQ_Project/STGN-SLSQ/out/STGN_train_noosa_small_stderr.out

source ~/environments/environment_python3.8.6.sh
source ~/venv/default_3.8.6/bin/activate
cd ~/SLSQ_Project/STGN-SLSQ/

python process_data.py --HPC --exp_name noosa
python train.py --HPC --exp_name noosa --load_all --adaptive --agg --epochs 10
python test.py --HPC --exp_name noosa --load_all --adaptive --agg
