# General
import argparse
from pathlib import Path

# Torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Learning HPC1
from learning_hpc.train import TrainerConfig, Trainer
from learning_hpc.model import DNATransformerConfig, DNATransformer
from learning_hpc.data import DNADataset, create_random_dna_string
from learning_hpc.distributed import init_parallel_state, is_rank0, resolve_device
from learning_hpc.utils import set_all_random_seeds



# Commands:
# - torchrun --standalone --nproc_per_node=4 train_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2
# - nohup torchrun --standalone --nproc_per_node=4 train_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2 > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'train_distributed.py'


# Notes:
# - We forego a lot of things for simplicity here, including train/val/test splits, pathing, and
#   model/optimizer hyperparameters. The focus here is on distributed training.



if __name__ == "__main__":
    # Setup (just reading args here for flexibility in calling the script)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(Path(__file__).parent.resolve()), help="Base directory for data and checkpoints")
    parser.add_argument("--context_len", type=int, default=2048, help="Context length for model")
    parser.add_argument("--model_dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--data_parallel_size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--sequence_parallel_size", type=int, default=2, help="Sequence parallel size")
    parser.add_argument("--resume_from_step", type=int, default=None, help="Step number to resume training from checkpoint")
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    context_len = args.context_len
    model_dim = args.model_dim
    steps = args.steps
    learning_rate = args.learning_rate
    dp_size = args.data_parallel_size
    sp_size = args.sequence_parallel_size
    resume_from_step = args.resume_from_step

    # Distributed setup
    parallel_state = init_parallel_state(
        dp_size=dp_size,
        sp_size=sp_size,
    )
    device = resolve_device(parallel_state=parallel_state)
    set_all_random_seeds(42 + parallel_state.dp_rank)

    # Get dataset and loader
    data_path = base_dir / "dna.txt"
    """
    We must have n_bases >= context_len to ensure that we don't have to deal with padding, which would bloat the 
    code and detract from the learning purpose of this example. Additionally, if n_bases is too high, the user 
    may not see meaningful improvement over training, which detracts from seeing tangible speedups from parallelism.
    Thus, we set n_bases to be just slightly larger than context_len (1% increase).
    """
    # Only have one of the parallel processes create the random dna string
    if is_rank0():
        create_random_dna_string(data_path, n_bases=int(1.01 * context_len), seed=parallel_state.rank + 42)
    # Make sure that the file is created before other ranks try to read it (dist.barrier() must be reached by all ranks before continuing)
    dist.barrier()
    # The dataset and loader are created on all processes
    # NOTE: Dataset can be inspected with print(DNADataset.dna_string)
    dataset = DNADataset(path=data_path, chunk_size=context_len, seed=parallel_state.rank + 42)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Train from scratch if no resume step is provided
    if not resume_from_step:
        ## Define the model
        model_cfg = DNATransformerConfig(
            vocab_size=4,  # DNA has nucleotides: [A, C, G, T]
            max_seq_len=context_len,
            dim=model_dim,
            num_heads=8,
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
        trainer = Trainer(config=trainer_cfg, model=model, device=device, parallel_state=parallel_state)
    # Otherwise, train from checkpoint
    else:
        ckpt_dir = f"{base_dir}/checkpoints/step_{resume_from_step}"
        trainer = Trainer.load(ckpt_dir, device, parallel_state=parallel_state)

    # Train the model
    trainer.set_loader(loader)
    trainer.model = DDP(trainer.model, process_group=parallel_state.world_group, device_ids=[device])
    trainer._init_optimizer()
    trainer.train(steps=steps)

    # Cleanup
    dist.destroy_process_group()
