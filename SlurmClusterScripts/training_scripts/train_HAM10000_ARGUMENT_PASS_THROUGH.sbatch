#!/bin/bash -l
#SBATCH --ntasks=1                 # Number of tasks (see below)
#SBATCH --partition=gpu-2080ti
#SBATCH --constraint=nvidia_v515
#SBATCH --cpus-per-task=72         # Number of CPU cores per task # max 72 on gpu-2080ti
#SBATCH --nodes=1                  # Ensure that all cores are on one machine
#SBATCH --time=0-20:00             # Runtime in D-HH:MM
#SBATCH --gres=gpu:8               # optionally type and number of gpus
#SBATCH --mem=340G                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/hennig/hmx148/slurm_logs/train_%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/hennig/hmx148/slurm_logs/train_%j.err    # File to which STDERR will be written - make sure this is not on $HOME

# insert your commands here
echo $SHELL

echo "Hostname: "
hostname

echo "  "

cd $WORK/MastersThesisCode/laplace-redux/
conda activate \$WORK/.conda/envs/MALaplaceRedux

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/qb/work/hennig/hmx148/.conda/envs/MALaplaceRedux/lib 

echo "RUNNING COMMAND: python ./baselines/vanilla/train_ham10000.py $@"

python ./baselines/vanilla/train_ham10000.py "$@"