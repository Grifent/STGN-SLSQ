#!/bin/bash -l

#PBS -N STGN_train_large

#PBS -l walltime=47:59:59
#PBS -l mem=32g
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -l gputype=A100

#PBS -o out/STGN_train_large_stdout.out
#PBS -e out/STGN_train_large_stderr.out

source ~/environments/environment_python3.8.6.sh
source ~/venv/default_3.8.6/bin/activate
cd ~/SLSQ_Project/STGN-SLSQ/

python process_data.py --HPC --exp_name large_person_type --max_len 5 --min_len 3
python train.py --HPC --exp_name large_person_type --max_len 5 --min_len 3 --load_all --adaptive --agg --epochs 100 --shape 720 1280
python test.py --HPC --exp_name large_person_type --load_all --adaptive --agg --shape 720 1280
python misc/cleanup.py
