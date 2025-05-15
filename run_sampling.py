import torch
from Utils.utils import load_args_from_file
from Utils.viz import show_images_grid
from huggingface_hub import hf_hub_download

from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler
from Sampler.confidence_sampler import ConfidenceSampler
from Sampler.diffusion_sampler import TauLeapingSampler, TauLeapingSamplerWrong
from my_stuff.guidance_schedules import get_guidance_schedule

config_path = "Config/base_cls2img.yaml"        # Path to your config file
args = load_args_from_file(config_path)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download the VQGAN from LlamaGen 
hf_hub_download(repo_id="FoundationVision/LlamaGen", 
                filename="vq_ds16_c2i.pt", 
                local_dir="./saved_networks/")

# Download the MaskGIT
hf_hub_download(repo_id="llvictorll/Halton-Maskgit", 
                filename="ImageNet_384_large.pth", 
                local_dir="./saved_networks/")

torch.manual_seed(0)
# Initialisation of the model
model = MaskGIT(args)

# select your scheduler
sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=0,
                        sched_pow=2, step=32, randomize=True, top_k=-1)

sampler = ConfidenceSampler(sm_temp=1, w=2, step=32)

w = 2
guid_schedule = get_guidance_schedule('constant', w)
sampler = TauLeapingSampler(model, guid_schedule)
# sampler = TauLeapingSamplerWrong(model, guid_schedule)


# [goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner]
labels = torch.tensor([1, 7, 282, 604, 724, 179, 751, 404], device='cuda')

gen_images = sampler.sample(8, 50, labels=labels, w=2)[1]
# gen_images = sampler(trainer=model, nb_sample=8, labels=labels, verbose=True)[0]
show_images_grid(gen_images)