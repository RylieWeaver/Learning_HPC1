# General
import argparse
from pathlib import Path

# Torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Learning HPC1
from learning_hpc.train import TrainerConfig, Trainer
from learning_hpc.model import DNATransformerConfig, DNATransformer
from learning_hpc.data import DNADataset, create_random_dna_string
from learning_hpc.distributed import init_parallel_state, is_rank0



# Commands:
# - torchrun --standalone --nproc_per_node=8 train_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2 --tensor_parallel_size 2
# - nohup torchrun --standalone --nproc_per_node=8 train_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2 --tensor_parallel_size 2 > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'train_distributed.py'


# Notes:
# - We forego a lot of things for simplicity here, including train/val/test splits, pathing, and
#   model/optimizer hyperparameters. The focus here is on 3D distributed training.



if __name__ == "__main__":
    # Setup (just reading args here for flexibility in calling the script)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(Path(__file__).parent.resolve()), help="Base directory for data and checkpoints")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for DNA sequences")
    parser.add_argument("--model_dim", type=int, default=2048, help="Model dimension")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--data_parallel_size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--sequence_parallel_size", type=int, default=2, help="Sequence parallel size")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--resume_from_step", type=int, default=None, help="Step number to resume training from checkpoint")
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    chunk_size = args.chunk_size
    model_dim = args.model_dim
    steps = args.steps
    learning_rate = args.learning_rate
    dp_size = args.data_parallel_size
    sp_size = args.sequence_parallel_size
    tp_size = args.tensor_parallel_size
    resume_from_step = args.resume_from_step

    # Distributed setup
    parallel_state = init_parallel_state(
        dp_size=dp_size,
        sp_size=sp_size,
        tp_size=tp_size,
        model_type="FSDP",  # NOTE: Make sure this matches how we wrap the model below!!!
    )
    torch.cuda.set_device(parallel_state.local_rank)
    device = torch.device(f"cuda:{parallel_state.local_rank}")

    # Get dataset and loader
    # NOTE: Dataset can be inspected with print(DNADataset.dna_string)
    data_path = base_dir / "dna.txt"
    # NOTE: If changing the context size, I recommend deleting and re-creating the random string. 
    #       Otherwise Feel free to treat is a black box, but for more details:
    """
    We must have n_bases >= chunk size to ensure that we don't have to deal with padding, which would bloat the 
    code and detract from the learning purpose of this example. Additionally, if n_bases is too high, the user 
    may not see meaningful improvement over training, which detracts from seeing tangible speedups from parallelism.
    Thus, we set n_bases to be just slightly larger than chunk_size (1% increase).
    """
    # Only have one of the parallel processes create the random dna string
    if is_rank0() and not data_path.exists():
        create_random_dna_string(data_path, n_bases=int(1.01 * chunk_size), seed=parallel_state.rank + 42)
    # Make sure that the file is created before other ranks try to read it (dist.barrier() must be reached by all ranks before continuing)
    dist.barrier()
    # The dataset and loader are created on all processes
    dataset = DNADataset(path=data_path, chunk_size=chunk_size, seed=parallel_state.rank + 42)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Train from scratch if no resume step is provided
    if not resume_from_step:
        ## Define the model
        model_cfg = DNATransformerConfig(
            vocab_size=4,  # DNA has nucleotides: [A, C, G, T]
            max_seq_len=chunk_size,
            dim=model_dim,
            num_heads=8,  # NOTE: This must be divisible by tp_size
            num_layers=6,
        )
        model = DNATransformer(model_cfg, parallel_state).to(device)
        ## Trainer configuration
        trainer_cfg = TrainerConfig(
            log_every=1,
            eval_every=100,
            eval_batches=10,
            batches_per_step=1,
            learning_rate=learning_rate,
            checkpoint_dir=f"{base_dir}/checkpoints",
            save_every=1000,
        )
        trainer = Trainer(trainer_cfg, model, device, parallel_state)
    # Otherwise, train from checkpoint
    else:
        ckpt_dir = f"{base_dir}/checkpoints/step_{resume_from_step}"
        trainer = Trainer.load(ckpt_dir, device, parallel_state=parallel_state)

    # Train the model
    trainer.set_loader(loader)
    # trainer.model = DDP(trainer.model, process_group=parallel_state.dpsp_group, device_ids=[device])  # Alternative DDP wrapping can be uncommented
    trainer.model = FSDP(trainer.model, process_group=parallel_state.dpsp_group, device_id=device)
    trainer._init_optimizer()  # Opt must be initialized after wrapping the model for FSDP
    trainer.train(steps=steps)

    # Cleanup
    dist.destroy_process_group()
