# General
import argparse
from pathlib import Path

# Torch
import torch

# Learning HPC1
from learning_hpc.train import TrainerConfig, Trainer
from learning_hpc.model import DNATransformerConfig, DNATransformer
from learning_hpc.data import DNADataset, create_random_dna_string
from learning_hpc.distributed import resolve_device
from learning_hpc.utils import set_all_random_seeds



# Commands:
# - python train.py --context_len 2048 --model_dim 1024


# Notes:
# - We forego a lot of things for simplicity here, including train/val/test splits, pathing, and
#   model hyperparameters. The focus here is on distributed training.



if __name__ == "__main__":
    # Setup (just reading args here for flexibility in calling the script)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(Path(__file__).parent.resolve()), help="Base directory for data and checkpoints")
    parser.add_argument("--context_len", type=int, default=2048, help="Context length for model")
    parser.add_argument("--model_dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory to resume training from checkpoint")
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    context_len = args.context_len
    model_dim = args.model_dim
    steps = args.steps
    learning_rate = args.learning_rate
    resume_from = args.resume_from
    set_all_random_seeds(42)

    # Distributed setup
    device = resolve_device()

    # Get dataset and loader
    data_path = base_dir / "dna.txt"
    # NOTE: If changing the context size, I recommend deleting and re-creating the random string. 
    #       Otherwise Feel free to treat is a black box, but for more details:
    """
    We must have n_bases >= context_len to ensure that we don't have to deal with padding, which would bloat the 
    code and detract from the learning purpose of this example. Additionally, if n_bases is too high, the user 
    may not see meaningful improvement over training, which detracts from seeing the training actually improve.
    Thus, we set n_bases to be just slightly larger than context_len (1% increase).
    """
    create_random_dna_string(data_path, n_bases=int(1.01 * context_len), seed=42)
    dataset = DNADataset(path=data_path, chunk_size=context_len, seed=42)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Train from scratch if no resume step is provided
    if not resume_from:
        ## Define the model
        model_cfg = DNATransformerConfig(
            vocab_size=4,  # DNA has nucleotides: [A, C, G, T]
            max_seq_len=context_len,
            dim=model_dim,
            num_heads=8,
            num_layers=6,
        )
        model = DNATransformer(model_cfg).to(device)
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
        trainer = Trainer(config=trainer_cfg, model=model, device=device)
    # Otherwise, train from checkpoint
    else:
        ckpt_dir = f"{base_dir}/checkpoints/{resume_from}"
        trainer = Trainer.load(ckpt_dir, device)

    # Train the model
    trainer.set_loader(loader)
    trainer._init_optimizer()
    trainer.train(steps=steps)
