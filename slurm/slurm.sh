#!/bin/bash
#SBATCH --job-name=APE_test_1    ## job name
#SBATCH --nodes=2            ## 索取 2 節點
#SBATCH --mem=16384
#SBATCH --ntasks-per-node=4      ## 每個節點運行 8 srun tasks
#SBATCH --cpus-per-task=4        ## 每個 srun task 索取 4 CPUs
#SBATCH --gres=gpu:8        ## 每個節點索取 8 GPUs
#SBATCH --account="MST112195"    ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gp1d        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、gp4d(最長跑4天)

# module purge
# module load singularity 
module load miniconda3
conda info --envs
huggingface-cli whoami
conda activate /home/kodmas2023/miniconda3/envs/ape
python ../experiments/run_instruction_induction.py --task=informal_to_formal
# python ../experiments/run_truthful_qa.py

# SIF=/work/TWCC_cntr/pytorch_20.09-py3_horovod.sif
# SINGULARITY="singularity run --nv $SIF"

# pytorch horovod benchmark script from
# wget https://raw.githubusercontent.com/horovod/horovod/v0.20.3/examples/pytorch/pytorch_synthetic_benchmark.py
#HOROVOD="python experiments/run_instruction_induction.py --task=antonyms"

# enable NCCL log
# export NCCL_DEBUG=INFO

# srun $SINGULARITY $HOROVOD
