import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

dp, sp, tp = 2, 3, 5

setup()
mesh = init_device_mesh("cuda", (dp, sp, tp), mesh_dim_names=("dp","sp","tp"))

tp_mesh = mesh["tp"]                  # TP collectives live here
ddp_group = mesh[("dp","sp")].get_group()  # avg grads over DP×SP (fix tp coordinate)

model = build_model().cuda()

# TP: row/col parallelize selected linears
plan = {
    "mlp.fc1": ColwiseParallel(),  # shard output features
    "mlp.fc2": RowwiseParallel(),  # shard input features
    # "attn.qkv": ColwiseParallel(),
    # "attn.out": RowwiseParallel(),
}
model = parallelize_module(model, tp_mesh, plan)

# DP×SP gradient averaging (NOT TP)
model = DDP(model, device_ids=[torch.cuda.current_device()], process_group=ddp_group)

optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = model(x).sum()
loss.backward()
optim.step()


