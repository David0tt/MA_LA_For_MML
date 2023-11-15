#!/bin/bash -l
#SBATCH --ntasks=1                 # Number of tasks (see below)
#SBATCH --partition=gpu-2080ti
#SBATCH --cpus-per-task=16          # Number of CPU cores per task # max 72 on gpu-2080ti
#SBATCH --nodes=1                  # Ensure that all cores are on one machine
#SBATCH --time=0-20:00             # Runtime in D-HH:MM
#SBATCH --gres=gpu:1               # optionally type and number of gpus
#SBATCH --mem=50G                  # Memory pool for all cores (see also --mem-per-cpu) # 200G is only for SkinLesion Runs, for others 50G would suffice
#SBATCH --output=/mnt/qb/work/hennig/hmx148/slurm_logs/uq_apt_%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/hennig/hmx148/slurm_logs/uq_apt_%j.err    # File to which STDERR will be written - make sure this is not on $HOME

# insert your commands here
echo $SHELL

echo "Hostname: "
hostname

echo "  "

cd $WORK/MastersThesisCode/laplace-redux/
conda activate \$WORK/.conda/envs/MALaplaceRedux

echo "RUNNING COMMAND: python uq.py $@"

python uq.py "$@"