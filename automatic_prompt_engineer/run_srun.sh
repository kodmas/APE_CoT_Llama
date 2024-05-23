#!/bin/bash
#SBATCH --job-name=Test          ## job name
#SBATCH --nodes=2                ## 索取 2 節點
#SBATCH --ntasks-per-node=8      ## 每個節點運行 8 srun tasks
#SBATCH --cpus-per-task=4        ## 每個 srun task 索取 4 CPUs
#SBATCH --gres=gpu:8             ## 每個節點索取 8 GPUs
#SBATCH --account="MST112195"    ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gtest        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、gp4d(最長跑4天)

module purge
module load singularity 

SIF=/work/TWCC_cntr/pytorch_20.09-py3_horovod.sif
SINGULARITY="singularity run --nv $SIF"

CONDA_INIT="conda init bash"

# pytorch horovod benchmark script from
# wget https://raw.githubusercontent.com/horovod/horovod/v0.20.3/examples/pytorch/pytorch_synthetic_benchmark.py
HOROVOD="conda activate ape" 
PYTHON="python ../experiments/run_instruction_induction.py"

# enable NCCL log
export NCCL_DEBUG=INFO

srun $SINGULARITY 
srun $CONDA_INIT
echo "conda init successfully" 
srun $HOROVOD
echo "conda activate successfully" 
srun $PYTHON
echo "Finish"
