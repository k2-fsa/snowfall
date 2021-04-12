import os
import torch
from torch import distributed as dist


def setup_dist(rank, world_size, master_port = None):
    os.environ['MASTER_ADDR'] = 'localhost'
<<<<<<< HEAD
    os.environ['MASTER_PORT'] = '12355'
=======
    os.environ['MASTER_PORT'] = ('12354' if master_port is None
                                 else str(master_port))
>>>>>>> 8f51e68... Make master port command-line configurable
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_dist():
    dist.destroy_process_group()
