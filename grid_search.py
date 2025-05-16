import click
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fid_score import calculate_fid_given_paths

@click.group()
def grid_search():
    pass

@grid_search.command()
@click.option('--num_process', type=int, default=8)
@click.option('--schedule', type=click.Choice(['interval', 'linear-ramp-up']), default='interval')
@click.option('--guidance_scale', type=float, default=2.)
@click.option('--ref_path', type=str, default='samples_cfg/guid_0/ref/fid_stats.npy')
@click.option('--out_dir', type=str, default='grid_search')
def noisy_guidance(num_process, schedule, guidance_scale, ref_path, out_dir):
    final_times = np.linspace(1.0, 0.0, 11)
    final_times = np.array([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, .0])
    # final_times = np.array([1.]) #, .9, .8, .7, .6, .5, .4, .3, .2, .1, .0])
    guidance_final_time = tqdm(final_times)
    schedules = ['linear-ramp-up', 'interval']
    for sched in schedules:
        for final_time in guidance_final_time:
            guidance_final_time.set_description(f'Guidance-final-time: {final_time}')
            fids = []
            folder_name = f'{out_dir}/{sched}/right_{final_time}'

            os.system(f'torchrun --master-port 29502 --nproc-per-node {num_process} sampling.py  \
                    --out_dir {folder_name} \
                    --w {guidance_scale} \
                    --guid_sched {sched} \
                    --num_samples 10000 \
                    --left_guid 0.0 \
                    --right_guid {final_time}')
            
            # fid = calculate_fid_given_paths(
            #     path=folder_name,
            #     ref_path=ref_path,
            #     save_path=folder_name,
            #     res=256
            # )
            # print(f'FID: {fid}')
            # fids.append(fid.item())
            
            # np.save(f'{out_dir}/fid_{final_time}.npy', fids)
        # fig,ax = plt.subplots()
        # ax.plot(final_times, fids)
        # fig.savefig(f'{out_dir}/guid_{final_time}.png')
        # plt.close(fig)
            
            
    
if __name__ == "__main__":
    grid_search()
