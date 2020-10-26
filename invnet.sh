#!/bin/sh
#
#SBATCH --verbose
#SBATCH -p gpu
#SBATCH --job-name=invnet
#SBATCH --output=out/invnet%j.out
#SBATCH --error=out/invnet%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=km3888@nyu.edu

module load anaconda3
module load cuda/9.0
source activate venvs


python /home/km3888/graph_invnet --output_path './output_dir'
