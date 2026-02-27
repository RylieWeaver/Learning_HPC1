#!/bin/bash
#SBATCH -A LRN036
#SBATCH -J TrainingAlphaGenomeModel
#SBATCH -o log/ddp-%j.o
#SBATCH -e log/ddp-%j.e
#SBATCH -t 02:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 2


# Modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

# Activate Environment (these are currently hard-coded paths)
source /lustre/orion/lrn036/world-shared/rylieweaver/Environments/LearningDistributed/dist-env/bin/activate
export PYTHONPATH=/home/r9w/Scratch/Distributed/Orbit/Learning/Learning_HPC1:$PYTHONPATH

# Distributed Env Vars
export MASTER_ADDR=$(hostname -i)      # This will be different per node, so is passed via command-line arg
export MASTER_PORT=3442                # Also passed via command-line arg for consistency
export NCCL_SOCKET_IFNAME=hsn0

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# World Configuration
export SLURM_NNODES=${SLURM_NNODES}
export NGPUS_PER_NODE=8
export WORLD_SIZE=$(($SLURM_NNODES * $NGPUS_PER_NODE))
GPUS_PER_TASK=1                 # May need to be changed when expanding past simple DDP
NTASKS=$(($WORLD_SIZE / $GPUS_PER_TASK))

# Proxies
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# Other Env Vars
export OMP_NUM_THREADS=1        # Number of threads for CPU operations (can be changed)
WANDB_API_KEY=$1                # WandB API Key passed as first argument (don't hard-code for security reasons)
export WANDB_API_KEY            # If the API Key is set, WandB logging will be enabled
# export NCCL_DEBUG=INFO          # Uncomment when debugging


# Run Script
echo "Running Python script"
FASTA_NUM=$((${SLURM_PROCID} % 32))  # Each process gets a different FASTA file based on its rank (reduces reading bottlenecks)
FASTA_PATH="/lustre/orion/lrn036/world-shared/rylieweaver/Poplar/file_copy${FASTA_NUM}.fa"
cd /home/r9w/Scratch/Distributed/Orbit/Learning/Learning_HPC1/examples/alphagenome_model/
srun -N ${SLURM_NNODES} --ntasks-per-node ${NGPUS_PER_NODE} -c 7 --gpus-per-task=$GPUS_PER_TASK --gpu-bind=closest \
    python3 -W ignore -u train_distributed.py --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}
