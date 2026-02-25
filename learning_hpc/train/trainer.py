# General
import json
from pathlib import Path
from typing import Optional, Union

# Torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Learning HPC1
from learning_hpc.model import DNATransformerConfig, DNATransformer
from learning_hpc.distributed import ParallelState, is_rank0, rank0_print, rank0_write, reduce_scalar, unwrap_model
from learning_hpc.data import move_to
from learning_hpc.utils import Config



def resolve_device(device: Optional[Union[torch.device, str]], parallel_state: ParallelState) -> torch.device:
    # NOTE: Only allow CPU training if not doing any type of parallelism
    if parallel_state and torch.cuda.is_available():
        local_rank = parallel_state.local_rank
        return torch.device(f"cuda:{local_rank}")
    elif isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return torch.device("cpu")


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
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every

    @staticmethod
    def load(path: Union[Path, str]) -> "TrainerConfig":
        path = Path(path) if path else None
        with path.open("r") as f:
            cfg = json.load(f)
        return TrainerConfig(**cfg)


class Trainer:
    def __init__(self, config, model, device: Optional[Union[torch.device, str]] = None, parallel_state: ParallelState = None):
        # Read args
        self.cfg = config
        self.device = resolve_device(device, parallel_state)
        self.parallel_state = parallel_state if parallel_state else ParallelState()

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
        token_ids = move_to(token_ids, self.device)
        # Get size of sptp group:
        if dist.get_world_size(group=self.parallel_state.sptp_group) > 1:
            sptp_group = self.parallel_state.sptp_group
            sptp_src = dist.get_global_rank(sptp_group, 0)
            dist.broadcast(token_ids, src=sptp_src, group=sptp_group)
        logits = self.model(token_ids)
        return logits

    def _shape_data(self, preds, labels):
        """
        Reshape data for metric computations.

        preds:  [B, S, V]  -->  [B*S, V]
        labels: [B, S]     -->  [B*S]
        """
        B, S, V = preds.size()
        preds = preds.view(B * S, V)
        labels = labels.view(B * S)
        return preds, labels

    def _accuracy_counter(self, preds, labels, ignore_index=-100, dim=-1):
        """
        The purpose of this function is to compute the number of correct predictions
        and the total count of predictions for accuracy calculation.

        The plumbing with ignore index is not important to what this repo is trying 
        to teach (3D parallelism). It's essentially there to ignore pads when counting 
        accuracy. There shouldn't be any pads in the toy data anyway, but the plumbing 
        is left so that it doesn't raise questions with the acc calculation.
        """
        predicted_class = torch.argmax(preds, dim=dim)
        mask = labels != ignore_index
        count = mask.sum().item()
        correct = (predicted_class[mask] == labels[mask]).sum().item()
        return correct, count

    def get_sp_preds_and_labels(self, logits, token_ids):
        """
        This function gets the subset of predictions and labels
        that are owned by the current sequence parallel rank.

        The sp_rank already only holds a subsequence of the 
        """
        # Setup
        B, S_sub, V = logits.size()
        _, S = token_ids.size()
        sp_size = self.parallel_state.sp_size
        sp_rank = self.parallel_state.sp_rank
        
        # Make offsets
        if self.parallel_state.sp_size == 1:
            return logits[:, :-1, :], token_ids[:, 1:]  # Simple MLM shift
        else:
            subseq_len = S_sub
            seq_offset = self.parallel_state.sp_rank * subseq_len
            start_idx = seq_offset
            end_idx = seq_offset + subseq_len

        # Get preds (only need to cut off on the last sp_rank for causal MLM)
        if sp_rank == sp_size - 1:
            sp_preds = logits[:, :-1, :]  # drop last pred
            sp_labels = token_ids[:, start_idx+1:end_idx]  # there is no label for the last pred
        else:
            sp_preds = logits
            sp_labels = token_ids[:, start_idx+1:end_idx+1]
        return sp_preds, sp_labels

    def _compute_metrics(self, logits, token_ids, weight=1.0):
        # Shifted preds/labels for causal MLM
        preds, labels = self.get_sp_preds_and_labels(logits, token_ids)
        # Reshape for loss/accuracy computation
        preds, labels = self._shape_data(preds, labels)
        # Loss computation
        loss = self.criterion(preds, labels) * weight  # Normalize loss if accumulating over minibatches
        # Accuracy computation
        correct, count = self._accuracy_counter(preds, labels)
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
            group=self.parallel_state.dpsp_group
        )
        self.cumulative_metrics[desc]["correct"] = reduce_scalar(
            self.cumulative_metrics[desc]["correct"], device=self.device, 
            group=self.parallel_state.dpsp_group, average=False
        )
        self.cumulative_metrics[desc]["count"] = reduce_scalar(
            self.cumulative_metrics[desc]["count"], device=self.device, 
            group=self.parallel_state.dpsp_group, average=False
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
                token_ids = next(self.loader).to(self.device)
                logits = self._run_batch(token_ids)
                batch_loss, batch_correct, batch_count = (
                    self._compute_metrics(logits, token_ids)
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
                token_ids = next(self.loader).to(self.device)       # [B, S] (int)
                logits = self._run_batch(token_ids)                 # [B, S, V]
                # Loss and backward
                loss, correct, count = self._compute_metrics(logits, token_ids, weight=1.0/self.cfg.batches_per_step)
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
                self._save_checkpoint(f"step_{self.last_step}")

    def _load_state_dict(self, state_dict: dict):
        self.last_step = state_dict["step"]

    # def _save_checkpoint(self, ckpt_dir: str):
    #     # Setup
    #     model = unwrap_model(self.model)  # Removes FSDP or DDP wrapper if present
    #     save_dir = self.cfg.checkpoint_dir / ckpt_dir
    #     if is_rank0():
    #         save_dir.mkdir(parents=True, exist_ok=True)
    #     dist.barrier()  # Ensure rank0 has finished creating save_dir before other ranks potentially use it

    #     # Purely Rank 0 saving (configs, trainer state)
    #     if is_rank0():
    #         # Trainer
    #         self.cfg.save(save_dir / "trainer_config.json")
    #         torch.save({"step": self.last_step}, save_dir / "trainer.pt")
    #         # Model Config
    #         model.cfg.save(save_dir / "model_config.json")

    #     # Potentially Sharded Saving (model and optimizer states)
    #     if isinstance(self.model, FSDP):
    #         # Setup
    #         dp_rank = self.parallel_state.dp_rank
    #         # Model
    #         torch.save(model.state_dict(), save_dir / f"model_rank_{dp_rank}.pt")
    #         # Optimizer
    #         torch.save({"optimizer": self.optimizer.state_dict()}, save_dir / f"optimizer_rank_{dp_rank}.pt")
    #     # NOTE: Saving is the same for DDP vs. non-distributed models
    #     elif is_rank0():
    #         # Model
    #         torch.save(model.state_dict(), save_dir / "model.pt")
    #         # Optimizer
    #         torch.save({"optimizer": self.optimizer.state_dict()}, save_dir / "optimizer.pt")

    def _save_checkpoint(self, ckpt_dir: str):
        # Setup
        model = unwrap_model(self.model)  # Removes FSDP or DDP wrapper if present
        save_dir = self.cfg.checkpoint_dir / ckpt_dir
        if is_rank0():
            save_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()  # Ensure rank0 has finished creating save_dir before other ranks potentially use it

        # Purely Rank 0 saving (configs, trainer state)
        if is_rank0():
            # Trainer
            self.cfg.save(save_dir / "trainer_config.json")
            torch.save({"step": self.last_step}, save_dir / "trainer.pt")
            # Model Config
            model.cfg.save(save_dir / "model_config.json")

        # Potentially Sharded Saving (model and optimizer states)
        dpsp_rank = dist.get_rank(group=self.parallel_state.dpsp_group)
        if dpsp_rank == 0:
            tp_rank = self.parallel_state.tp_rank
            # Model
            torch.save(model.state_dict(), save_dir / f"model_{tp_rank}.pt")
            # Optimizer
            torch.save({"optimizer": self.optimizer.state_dict()}, save_dir / f"optimizer_{tp_rank}.pt")

    # @staticmethod
    # def load_checkpoint(dir: Union[Path, str], device: torch.device, parallel_state: ParallelState = None) -> "Trainer":
    #     # Setup
    #     dir = Path(dir)
    #     parallel_state = parallel_state if parallel_state else ParallelState()
    #     model_type = parallel_state.model_type  # e.g. "default", "DDP", "FSDP"

    #     # Model config
    #     model_dict_path = dir / "model_config.json"
    #     with model_dict_path.open("r") as f:
    #         model_dict = json.load(f)
    #     # NOTE: We only support DNATransformer here for simplicity
    #     model_cfg = DNATransformerConfig(**model_dict)
    #     model = DNATransformer(model_cfg, parallel_state).to(device)

    #     # Model (must be before optimizer load for FSDP)
    #     if model_type == "FSDP":
    #         dp_rank = parallel_state.dp_rank
    #         state_dict = torch.load(dir / f"model_rank_{dp_rank}.pt", weights_only=True)
    #         model.load_state_dict(state_dict)
    #         model = FSDP(model, process_group=parallel_state.dpsp_group, device_id=device)
    #     elif model_type == "DDP":
    #         state_dict = torch.load(dir / f"model.pt", weights_only=True)
    #         model.load_state_dict(state_dict)
    #         model = DDP(model, process_group=parallel_state.dpsp_group, device_ids=device)
    #     else:
    #         state_dict = torch.load(dir / f"model.pt", weights_only=True)
    #         model.load_state_dict(state_dict)

    #     # Trainer (model load within trainer instantiation)
    #     trainer_dict_path = dir / "trainer_config.json"
    #     trainer_cfg = TrainerConfig.load(trainer_dict_path)
    #     trainer = Trainer(trainer_cfg, model, device, parallel_state)
    #     trainer_state = torch.load(dir / "trainer.pt")
    #     trainer.last_step = trainer_state["step"]
        
    #     # Optimizer
    #     trainer._init_optimizer()
    #     if model_type == "FSDP":
    #         dp_rank = parallel_state.dp_rank
    #         optimizer_state = torch.load(dir / f"optimizer_rank_{dp_rank}.pt")
    #     else:
    #         optimizer_state = torch.load(dir / "optimizer.pt")
    #     trainer.optimizer.load_state_dict(optimizer_state["optimizer"])
    #     return trainer

    @staticmethod
    def load_checkpoint(dir: Union[Path, str], device: torch.device, parallel_state: ParallelState = None) -> "Trainer":
        # Setup
        dir = Path(dir)
        parallel_state = parallel_state if parallel_state else ParallelState()
        tp_rank = parallel_state.tp_rank

        # Model config
        model_dict_path = dir / "model_config.json"
        with model_dict_path.open("r") as f:
            model_dict = json.load(f)
        # NOTE: We only support DNATransformer here for simplicity
        model_cfg = DNATransformerConfig(**model_dict)
        model = DNATransformer(model_cfg, parallel_state).to(device)

        # Model
        state_dict = torch.load(dir / f"model_{tp_rank}.pt", weights_only=True)
        model.load_state_dict(state_dict)

        # Trainer (model load within trainer instantiation)
        trainer_dict_path = dir / "trainer_config.json"
        trainer_cfg = TrainerConfig.load(trainer_dict_path)
        trainer = Trainer(trainer_cfg, model, device, parallel_state)
        trainer_state = torch.load(dir / "trainer.pt")
        trainer.last_step = trainer_state["step"]
        
        # Optimizer
        trainer._init_optimizer()
        optimizer_state = torch.load(dir / f"optimizer_{tp_rank}.pt")
        trainer.optimizer.load_state_dict(optimizer_state["optimizer"])
        return trainer
