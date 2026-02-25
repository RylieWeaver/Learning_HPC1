# Environment Setup
Install uv if not already: curl -LsSf https://astral.sh/uv/install.sh | sh
uv init LearningDistributed --bare --python 3.12 && cd LearningDistributed
uv venv dist-env --python 3.12 --native-tls && source dist-env/bin/activate
uv pip install torch numpy einops

# Running Examples
cd */Learning_HPC1/
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd examples/
python train.py --context_len 2048 --model_dim 1024
torchrun --standalone --nproc_per_node=4 test_dpsp.py  --data_parallel_size 2  --sequence_parallel_size 2
torchrun --standalone --nproc_per_node=4 train_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2 --context_len 2048 --model_dim 1024
