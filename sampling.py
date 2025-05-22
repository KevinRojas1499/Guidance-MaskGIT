import os
from PIL import Image
import torchvision.transforms as T
import torch
import click
import torch.distributed as dist
import os
from tqdm import tqdm
from Utils.utils import load_args_from_file
from Utils.viz import show_images_grid
from huggingface_hub import hf_hub_download

from Trainer.cls_trainer import MaskGIT
from Sampler.diffusion_sampler import TauLeapingSampler, TauLeapingUnlocking, SimpleGuidance
from my_stuff.guidance_schedules import get_guidance_schedule

def setup_distributed():
    """Initialize distributed training setup."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size()
    
    return 0, 1

def unprocess(img):
    if img.dim() == 4: 
        img = img.squeeze(0)
    if img.min() < 0:
        img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    return img


@click.command()
@click.option('--config', default="Config/base_cls2img.yaml", help='Path to config file')
@click.option('--sampler', type=click.Choice(['tau', 'unlocking', 'simple']), 
              default='tau', help='Sampling method to use')
@click.option('--guid_sched', type=click.Choice(['constant', 'interval', 'linear-ramp-up']), 
              default='constant', help='Guidance Schedule to use')
@click.option('--left_guid', default=0., help='Left endpoint for guidance interval')
@click.option('--right_guid', default=1., help='Right endpoint for guidance interval')
@click.option('--num_samples', default=50000, help='Number of samples to generate')
@click.option('--batch_size', default=125, help='Per GPU Batch size for generation')
@click.option('--steps', default=50, help='Number of sampling steps')
@click.option('--seed', default=0, help='Random seed')
@click.option('--w', default=0., type=float, help='Guidance weight parameter')
@click.option('--out_dir', type=str, help='Directory to save the samples')
def sample(config, sampler, guid_sched, left_guid, right_guid, num_samples, batch_size, steps, seed, w, out_dir):
    rank, world_size = setup_distributed()
    torch.manual_seed(seed + rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    args = load_args_from_file(config)
    args.device = device
    args.global_rank = rank
    args.world_size = world_size

    if rank == 0:
        hf_hub_download(repo_id="FoundationVision/LlamaGen", 
                        filename="vq_ds16_c2i.pt", 
                        local_dir="./saved_networks/")

        hf_hub_download(repo_id="llvictorll/Halton-Maskgit", 
                        filename="ImageNet_384_large.pth", 
                        local_dir="./saved_networks/")

    dist.barrier()
    
    guid_schedule = get_guidance_schedule(guid_sched, w, left=left_guid, right=right_guid)
    model = MaskGIT(args)
    local_num_samples = num_samples // world_size + 1

    if sampler == 'tau':
        sampler_obj = TauLeapingSampler(model, guidance_schedule=guid_schedule)
    elif sampler == 'unlocking':
        sampler_obj = TauLeapingUnlocking(model, guidance_schedule=guid_schedule)
    elif sampler == 'simple':
        sampler_obj = SimpleGuidance(model, guidance_schedule=guid_schedule)
    

    out_dir = os.path.join(out_dir, f'Rank-{rank}')
    os.makedirs(out_dir, exist_ok=True)
    
    to_pil = T.ToPILImage()
    
    for i in tqdm(range(0, local_num_samples, batch_size)):
        current_batch_size = min(batch_size, local_num_samples - i)
        
        batch_labels = torch.randint(0, 1000, (current_batch_size, ), device=device)
        _, batch_images = sampler_obj.sample(current_batch_size, steps, labels=batch_labels, w=w)
        
        for j, img in enumerate(batch_images):
            img_idx = i * batch_size + j
            img_idx_str = f'{img_idx:08d}'
            folder = os.path.join(out_dir, img_idx_str[:4])
            os.makedirs(folder, exist_ok=True)
            img = unprocess(img)
            
            pil_img = to_pil(img)
            pil_img.save(os.path.join(folder, f'{img_idx}.png'))
            
    
    dist.destroy_process_group()


if __name__ == '__main__':
    sample()