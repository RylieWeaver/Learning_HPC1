# Environment Setup

## DGX
Install uv if not already: curl -LsSf https://astral.sh/uv/install.sh | sh
uv init LearningDistributed --bare --python 3.12 && cd LearningDistributed
uv venv dist-env --python 3.12 --native-tls && source dist-env/bin/activate
uv pip install torch numpy einops

## Frontier
Install uv if not already: curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p /lustre/orion/<proj_id>/world-shared/<user_id>/Environments
cd /lustre/orion/<proj_id>/world-shared/<user_id>/Environments
export UV_CACHE_DIR=/lustre/orion/<proj_id>/world-shared/<user_id>/Environments

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

uv init LearningDistributed --bare --python 3.13 && cd LearningDistributed
uv venv dist-env --python 3.13 --native-tls && source dist-env/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
uv pip install numpy einops


# Running Examples
cd */Learning_HPC1/
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd examples/
python train.py --context_len 2048 --model_dim 1024
torchrun --standalone --nproc_per_node=4 test_dpsp.py  --data_parallel_size 2  --sequence_parallel_size 2
torchrun --standalone --nproc_per_node=4 train_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2 --context_len 2048 --model_dim 1024
sbatch script.sh
