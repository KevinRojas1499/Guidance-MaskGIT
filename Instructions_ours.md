Please use the instructions from the original repo found in the README.md file to set up

## Sampling

You can sample using the following command:

```{bash}
torchrun --nproc-per-node 8 sampling.py --sampler tau --guid_sched constant --out_dir samples_dir
```

Other options for the sampler are `tau`, `unlocking` and `simple` to change the guiding mechanism. 

## FID evaluation

You can evaluate FID between two image folder using the following command:

```{bash}
torchrun fid_score.py --path path1 --ref_path path2 --save_path path_to_save_stats 
```
This will print the FID value and save the reference statistics of each folder for faster evaluation in the future.