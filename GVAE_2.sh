#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --job-name=GVAE2
#SBATCH --account=pr-jreid03-1-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=royhe@student.ubc.ca
#SBATCH --output=output_GVAE_2.txt
#SBATCH --error=output_GVAE_2.txt

cd $SLURM_SUBMIT_DIR
module load conda
conda init bash
module load gcc
module load openmpi
module load git
source ~/.bashrc
conda activate gvae_2
python train.py
conda deactivate
