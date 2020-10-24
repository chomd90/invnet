#!/bin/sh
#
#SBATCH --verbose
#SBATCH -p gpu
#SBATCH --job-name=pearl_cheetah
#SBATCH --output=pearl_cheetah%j.out
#SBATCH --error=pearl_cheetah%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=km3888@nyu.edu

module load anaconda3
module load cuda/9.0
source activate rl


python /gpfsnyu/home/km3888/oyster/launch_experiment.py 0 /gpfsnyu/home/km3888/oyster/configs/cheetah-dir.json
