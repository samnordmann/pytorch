import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_pg(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_pg():
    dist.destroy_process_group()

print("GPUs detected: ", torch.cuda.device_count())  # Should return 8

model_name = "llama-3.1-8b"  # Update with the exact model identifier.

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency.
    device_map="auto"  # Automatically map layers to available GPUs.
)




# Example setup for 8 GPUs
world_size = 8
rank = ...  # Set rank per process (0 to 7)

setup_pg(rank, world_size)

# Place model on rank-specific GPU
device = torch.device(f'cuda:{rank}')
model = model.to(device)

# Wrap with DDP
ddp_model = DDP(model, device_ids=[rank])
