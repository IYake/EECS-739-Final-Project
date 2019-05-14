#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --mem=4GB
#SBATCH --reservation=eecs739
#SBATCH --gres="gpu:p100:1"
#SBATCH -t 00:20:00
#SBATCH -J gpu_test
#SBATCH -o output.out
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Setup environment
module load CUDA/8.0.44

# Run gpucode 
nvcc -o GaussNewton GaussNewton.cu
./GaussNewton