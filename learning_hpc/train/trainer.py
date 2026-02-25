# General
import json
from pathlib import Path
from typing import Optional, Union

# Torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Learning HPC1
from learning_hpc.model import DNATransformerConfig, DNATransformer
from learning_hpc.distributed import ParallelState, is_rank0, rank0_print, rank0_write, reduce_scalar, unwrap_model, resolve_device
from learning_hpc.data import move_to
from learning_hpc.utils import Config



class TrainerConfig(Config):
    def __init__(
            self,
            log_every: int = 1,
            eval_every: int = 100,
            eval_batches: int = 10,
            batches_per_step: int = 1,
            learning_rate: float = 1e-4,
            log_dir: Optional[Union[Path, str]] = None,
            checkpoint_dir: Optional[Union[Path, str]] = None,
            save_every: Optional[int] = None,
            **kwargs
    ):
        # Read args
        self.log_every = log_every
        self.eval_every = eval_every
        self.eval_batches = eval_batches
        self.batches_per_step = batches_per_step
        self.learning_rate = learning_rate
        self.log_dir = Path(log_dir) if log_dir else Path("log")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir and is_rank0():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every

    @staticmethod
    def load(path: Union[Path, str]) -> "TrainerConfig":
        path = Path(path) if path else None
        with path.open("r") as f:
            cfg = json.load(f)
        return TrainerConfig(**cfg)


class Trainer:
    def __init__(self, config, model, device: Optional[Union[torch.device, str]] = None, parallel_state: Optional[ParallelState] = None):
        # Read args
        self.cfg = config
        self.parallel_state = parallel_state if parallel_state else ParallelState()
        self.device = resolve_device(device, self.parallel_state)

        # Init objects
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.descriptors = ["Train", "Eval"]
        self._init_cumulative_metrics()

        # Init trainer state
        self.last_step = 0

    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-4)
    
    def _init_cumulative_metrics(self):
        """
        These are used to accumulate metrics over multiple batches before averaging and logging.
        """
        self.cumulative_metrics = {}
        for desc in self.descriptors:
            self.cumulative_metrics[desc] = {
                "loss": 0.0,
                "correct": 0,
                "count": 0
            }

    def _log_metrics(self, desc: str = ""):
        acc = self.cumulative_metrics[desc]["correct"] / self.cumulative_metrics[desc]["count"] if self.cumulative_metrics[desc]["count"] > 0 else 0.0
        message = (
            f"[Step] {self.last_step}: "
            f"{desc} Loss: {self.cumulative_metrics[desc]['loss']:.4f}, "
            f"{desc} Accuracy: {acc:.4f}"
        )
        rank0_write(self.cfg.log_dir / "log.txt", message)
        rank0_print(message)

    def set_loader(self, loader):
        self.loader = iter(loader)
        self.cfg.batch_size = loader.batch_size

    def _run_batch(self, token_ids):
        # Get size of sp group:
        if self.parallel_state.sp_size > 1:
            sp_group = self.parallel_state.sp_group
            sp_src = dist.get_global_rank(sp_group, 0)
            dist.broadcast(token_ids, src=sp_src, group=sp_group)
        logits, labels = self.model(token_ids)
        return logits, labels

    def _shape_data(self, logits, labels):
        """
        Reshape data for cross_entropy loss.

        logits:  [B, S, V]  -->  [B*S, V]
        labels: [B, S]      -->  [B*S]
        """
        B, S, V = logits.size()
        logits = logits.view(B * S, V)
        labels = labels.view(B * S)
        return logits, labels

    def _accuracy_counter(self, logits, labels, ignore_index=-100, dim=-1):
        """
        The purpose of this function is to compute the number of correct predictions
        and the total count of predictions for accuracy calculation.

        The plumbing with ignore index is not important to what this repo is trying 
        to teach (parallelism). It's essentially there to ignore pads when counting 
        accuracy. There shouldn't be any pads in the toy data anyway, but the plumbing 
        is left so that it doesn't raise questions with the acc calculation.
        """
        predicted_class = torch.argmax(logits, dim=dim)
        mask = labels != ignore_index
        count = mask.sum().item()
        correct = (predicted_class[mask] == labels[mask]).sum().item()
        return correct, count

    def _compute_metrics(self, logits, labels, weight=1.0):
        # Reshape for loss/accuracy computation
        logits, labels = self._shape_data(logits, labels)
        # Loss computation
        loss = self.criterion(logits, labels) * weight  # Normalize loss if accumulating over minibatches
        # Accuracy computation
        correct, count = self._accuracy_counter(logits, labels)
        return loss, correct, count

    def _inc_metrics(self, loss, correct, count, desc):
        self.cumulative_metrics[desc]["loss"] += loss.item()
        self.cumulative_metrics[desc]["correct"] += correct
        self.cumulative_metrics[desc]["count"] += count

    def _reduce_metrics(self, desc):
        """
        This reduces metrics over sequence and data parallelism.

        Reduction could be sum or average depending on the metric.

        Note that either/both parallelism dimensions may be size 1, 
        in which case the reduction is a no-op for that dimension.
        """
        self.cumulative_metrics[desc]["loss"] = reduce_scalar(
            self.cumulative_metrics[desc]["loss"], device=self.device, 
            group=self.parallel_state.world_group
        )
        self.cumulative_metrics[desc]["correct"] = reduce_scalar(
            self.cumulative_metrics[desc]["correct"], device=self.device, 
            group=self.parallel_state.world_group, average=False
        )
        self.cumulative_metrics[desc]["count"] = reduce_scalar(
            self.cumulative_metrics[desc]["count"], device=self.device, 
            group=self.parallel_state.world_group, average=False
        )

    def _average_loss(self, batches, desc):
        """
        We only average loss over batches here because
        accuracy = (correct / count), so dividing both correct
        and count by batches would cancel out.
        """
        self.cumulative_metrics[desc]["loss"] = self.cumulative_metrics[desc]["loss"] / batches

    def _run_eval(self):
        # Setup
        self.model.eval()
        d = self.descriptors[1]  # Eval descriptor for easy reference
        steps = self.cfg.eval_batches
        self._init_cumulative_metrics()  # Reset metrics

        # Accumulate metrics
        with torch.no_grad():
            for _ in range(steps):
                token_ids = next(self.loader)
                token_ids = move_to(token_ids, self.device)
                logits, labels = self._run_batch(token_ids)
                batch_loss, batch_correct, batch_count = (
                    self._compute_metrics(logits, labels)
                )
                self._inc_metrics(batch_loss, batch_correct, batch_count, d)

        # Average, Log, and Reset
        self._reduce_metrics(d)  # Reduce metrics over parallel processes
        self._average_loss(self.cfg.eval_batches, d)  # Average loss over batches
        self._log_metrics(desc=d)
        self._init_cumulative_metrics()  # Reset metrics
    
    def train(self, steps=10000):
        # Setup
        end_step = self.last_step + steps

        # Initial evaluation before training
        if self.last_step == 0:
            self._run_eval()
            self.model.train()  # Switch back to train

        # Training loop
        while self.last_step < end_step:
            self.optimizer.zero_grad(set_to_none=True)
            # Accumulate gradients
            for _ in range(self.cfg.batches_per_step):
                # Forward
                token_ids = next(self.loader)           # [B, S] (int)
                token_ids = move_to(token_ids, self.device)
                logits, labels = self._run_batch(token_ids)     # [B, S, V]
                # Loss and backward
                loss, correct, count = self._compute_metrics(logits, labels, weight=1.0/self.cfg.batches_per_step)
                self._inc_metrics(loss, correct, count, self.descriptors[0])
                loss.backward()
            # Optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.last_step += 1

            # Reached log step
            if self.last_step % self.cfg.log_every == 0:
                d = self.descriptors[0]  # Train descriptor for easy reference
                self._reduce_metrics(d)  # Reduce metrics over parallel processes
                self._average_loss(self.cfg.log_every, d)  # Average loss over batches
                self._log_metrics(desc=d)
                self._init_cumulative_metrics()  # Reset metrics

            # Reached eval step
            # NOTE: We don't do best-model checkpointing to stay simple
            if self.last_step % self.cfg.eval_every == 0:
                self._run_eval()
                self.model.train()  # Switch back to train

            # Reached checkpoint step
            if self.cfg.checkpoint_dir and self.cfg.save_every and self.last_step % self.cfg.save_every == 0:
                self.save_checkpoint(f"step_{self.last_step}")

    def _load_state_dict(self, state_dict: dict):
        self.last_step = state_dict["step"]

    def save_checkpoint(self, ckpt_dir: str):
        # Setup
        model = unwrap_model(self.model)  # Removes DDP wrapper if present
        save_dir = self.cfg.checkpoint_dir / ckpt_dir

        # Save (only on rank 0)
        if is_rank0():
            save_dir.mkdir(parents=True, exist_ok=True)
            # Model
            model.cfg.save(save_dir / "model_config.json")
            torch.save(model.state_dict(), save_dir / f"model.pt")
            # Trainer
            self.cfg.save(save_dir / "trainer_config.json")
            torch.save({"step": self.last_step}, save_dir / "trainer.pt")
            # Optimizer
            torch.save({"optimizer": self.optimizer.state_dict()}, save_dir / f"optimizer.pt")

    @staticmethod
    def load_checkpoint(dir: Union[Path, str], device: torch.device, parallel_state: ParallelState = None) -> "Trainer":
        # Setup
        dir = Path(dir)
        parallel_state = parallel_state if parallel_state else ParallelState()

        # Model
        model_dict_path = dir / "model_config.json"
        with model_dict_path.open("r") as f:
            model_dict = json.load(f)
        # NOTE: We only support DNATransformer here for simplicity
        model_cfg = DNATransformerConfig(**model_dict)
        model = DNATransformer(model_cfg, parallel_state).to(device)
        state_dict = torch.load(dir / f"model.pt", weights_only=True)
        model.load_state_dict(state_dict)

        # Trainer
        trainer_dict_path = dir / "trainer_config.json"
        trainer_cfg = TrainerConfig.load(trainer_dict_path)
        trainer = Trainer(config=trainer_cfg, model=model, device=device, parallel_state=parallel_state)
        trainer_state = torch.load(dir / "trainer.pt")
        trainer.last_step = trainer_state["step"]
        
        # Optimizer
        trainer._init_optimizer()
        optimizer_state = torch.load(dir / "optimizer.pt")
        trainer.optimizer.load_state_dict(optimizer_state["optimizer"])
        return trainer
