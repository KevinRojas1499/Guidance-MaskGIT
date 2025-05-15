import numpy as np
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import abc
from tqdm import tqdm
from my_stuff.graph_lib import Absorbing

class BaseSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self):
        pass
   
class TauLeapingSampler(BaseSampler):
    def __init__(self, maskgit, guidance_schedule):
        self.maskgit = maskgit
        self.N = self.maskgit.args.codebook_size + 1 # 1025
        # print(maskgit.args)
        self.patch_size = maskgit.input_size # 16 or 32
        self.D = self.patch_size ** 2 # 16*16 or 32*32
        self.device = self.maskgit.args.device
        self.noise = Absorbing(self.N)
        self.guid_sched = guidance_schedule

    @torch.no_grad()
    @torch.amp.autocast(device_type='cuda')
    def qrate_fn(self, x:torch.tensor, t:torch.tensor, labels:torch.LongTensor, 
                 w:float=3, sm_temp:float=1):
        """
        x: [B, 16*16], t: [B], labels: [B]
        output: [B, 16*16, 1025]
        """
        drop = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        x_ = x.view(-1, self.patch_size, self.patch_size)
        if w != 0:
            logit = self.maskgit.vit(torch.cat([x_.clone(), x_.clone()], dim=0), # image token, [B, 16, 16]
                                    torch.cat([labels, labels], dim=0), # condition class, B
                                    torch.cat([~drop, drop], dim=0)) # drop label, B
            logit_c, logit_u = torch.chunk(logit, 2, dim=0)
            _w = self.guid_sched(t, w)[0].item()
            # Classifier Free Guidance, 
            # guidance strength is linearly increased from 0 to w
            logit = (1 + _w) * logit_c - _w * logit_u
        else:
            logit = self.maskgit.vit(x_.clone(), labels, drop_label=~drop)
        prob = torch.softmax(logit * sm_temp, -1)
        prob[x!=self.N-1] = 0
        prob.scatter_(-1, x[..., None], torch.zeros_like(prob[..., :1])) # make prob[b,d,x[b][d]] = 0

        rate = self.noise.sigma(t) / torch.expm1(self.noise.sigma_int(t))
        return rate[..., None, None] * prob

    @torch.no_grad()
    def sample(self, batch_num, num_steps, labels = None, max_t=1, eps=.01, w=3, sm_temp=1):
        """
        return: codes: [B, 16*16] or [B, 32*32]; decoded images: [B, 3, 256, 256] or [B, 3, 512, 512]
        """
        self.maskgit.vit.eval()
        if labels is None:
            # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
            if batch_num >= 10:
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999] * (batch_num // 10)
            else:
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999][:batch_num]
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        arange_vals = torch.arange(self.N, device=self.device).reshape(1, 1, self.N)
        ones = torch.ones((batch_num,), device=self.device)
        x = torch.full((batch_num, self.D), self.N-1, dtype=torch.int64, device=self.device)
        ts = np.linspace(max_t, eps, num_steps)

        for idx in tqdm(range(num_steps-1), leave=False):
            # ts[idx] (larger) -> ts[idx+1] (smaller)
            t=ts[idx]
            t_ten = ones * t
            h = ts[idx] - ts[idx + 1]
            qmat = self.qrate_fn(x, t_ten, labels=labels, w=w, sm_temp=sm_temp) # [B, D, N] TODO
            diffs = arange_vals - x.unsqueeze(-1)
            # [B, D, N], state change, diffs[b,d,n] = n - x[b,d]
            jump_nums = torch.distributions.poisson.Poisson(h * qmat).sample()
            jump_nums[jump_nums.sum(dim = -1) > 1] = 0 # TODO reject multiple jumps to the same token
            overall_jump = torch.sum(jump_nums * diffs, dim=-1)
            x = torch.clamp(x + overall_jump, min=0, max=self.N-1).to(dtype=torch.int64)

            del jump_nums, overall_jump, qmat, diffs
    
        # At final time eps, replace mask with the most probable token
        eps_ten = torch.full((batch_num,), eps).to(self.device)
        qmat = self.qrate_fn(x, eps_ten, labels=labels, w=w, sm_temp=sm_temp)
        masked = x == self.N - 1
        x[masked]=torch.argmax(qmat[:,:,:-1], dim=-1)[masked] # exclude the mask token

        return x, self.maskgit.ae.decode_code(x.view(batch_num, self.patch_size, self.patch_size))
    
class TauLeapingSamplerWrong(BaseSampler):
    def __init__(self, maskgit, guidance_schedule):
        self.maskgit = maskgit
        self.N = self.maskgit.args.codebook_size + 1 # 1025
        # print(maskgit.args)
        self.patch_size = maskgit.input_size # 16 or 32
        self.D = self.patch_size ** 2 # 16*16 or 32*32
        self.device = self.maskgit.args.device
        self.noise = Absorbing(self.N)
        self.guid_sched = guidance_schedule
    
    def get_prob(self, logits, x, sm_temp):
        prob = torch.softmax(logits * sm_temp, -1)
        prob[x!=self.N-1]=0
        prob.scatter_(-1, x[..., None], torch.zeros_like(prob[..., :1])) # make prob[b,d,x[b][d]] = 0
        return prob

    @torch.amp.autocast(device_type='cuda')
    def qrate_fn(self, x:torch.tensor, t:torch.tensor, labels:torch.LongTensor, 
                 w:float=3, sm_temp:float=1):
        """
        x: [B, 16*16], t: [B], labels: [B]
        output: [B, 16*16, 1025]
        """
        drop = torch.ones(x.shape[0], dtype=torch.bool).to(x.device)
        x_ = x.view(-1, self.patch_size, self.patch_size)
        if w != 0:
            logit = self.maskgit.vit(torch.cat([x_.clone(), x_.clone()], dim=0), # image token, [B, 16, 16]
                                    torch.cat([labels, labels], dim=0), # condition class, B
                                    torch.cat([~drop, drop], dim=0)) # drop label, B
            logit_c, logit_u = torch.chunk(logit, 2, dim=0)
            prob_c = self.get_prob(logit_c, x, sm_temp=sm_temp)
            prob_u = self.get_prob(logit_u, x, sm_temp=sm_temp)
            _w = self.guid_sched(t, w)[0].item()
            # Classifier Free Guidance, 
            log_prob_c = torch.log(prob_c + 1e-10) 
            log_prob_u = torch.log(prob_u + 1e-10)
            log_prob = (1+_w) * log_prob_c - _w * log_prob_u
            prob = torch.exp(log_prob) # Notice that this is unnormalized
        else:
            logit = self.maskgit.vit(x_.clone(), labels, drop_label=~drop)

            prob = self.get_prob(logit, x, sm_temp=sm_temp)
        rate = self.noise.sigma(t) / torch.expm1(self.noise.sigma_int(t))
        return rate[..., None, None] * prob

    @torch.no_grad()
    def sample(self, batch_num, num_steps, labels = None, max_t=1, eps=.01, w=3, sm_temp=1):
        """
        return: codes: [B, 16*16] or [B, 32*32]; decoded images: [B, 3, 256, 256] or [B, 3, 512, 512]
        """
        self.maskgit.vit.eval()
        if labels is None:  # Default classes generated
            # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
            if batch_num >= 10:
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999] * (batch_num // 10)
            else:
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999][:batch_num]
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        x = torch.full((batch_num, self.D), self.N-1, dtype=torch.int64).to(self.device) # [B, D]
        ts = np.linspace(max_t, eps, num_steps) # from max_t (prior) to eps (~data)

        for idx in tqdm(range(num_steps-1)):
            # ts[idx] (larger) -> ts[idx+1] (smaller)
            t=ts[idx]; h = ts[idx] - ts[idx + 1]
            t_ten = torch.full((batch_num,), t).to(self.device)
            qmat = self.qrate_fn(x, t_ten, labels=labels, w=w, sm_temp=sm_temp) # [B, D, N] TODO
            diffs = torch.arange(self.N).view(1, 1, self.N).to(self.device) - x.view(batch_num, self.D, 1)
            # [B, D, N], state change, diffs[b,d,n] = n - x[b,d]
            jump_nums = torch.distributions.poisson.Poisson(h * qmat).sample().to(self.device) # [B, D, N]
            jump_nums[jump_nums.sum(dim = -1) > 1] = 0 # TODO reject multiple jumps to the same token
            overall_jump = torch.sum(jump_nums * diffs, dim=-1).to(self.device)
            x = torch.clamp(x + overall_jump, min=0, max=self.N-1).to(torch.int64) # [B, D]
            
            del jump_nums, overall_jump, qmat, diffs
    
        # At final time eps, replace mask with the most probable token
        eps_ten = torch.full((batch_num,), eps).to(self.device)
        qmat = self.qrate_fn(x, eps_ten, labels=labels, w=w, sm_temp=sm_temp)
        masked = x == self.N - 1
        x[masked]=torch.argmax(qmat[:,:,:-1], dim=-1)[masked] # exclude the mask token

        return x, self.maskgit.ae.decode_code(x.view(batch_num, self.patch_size, self.patch_size))

class SimpleGuidance(BaseSampler):
    def __init__(self, maskgit, guidance_schedule):
        self.maskgit = maskgit
        self.N = self.maskgit.args.codebook_size + 1 # 1025
        # print(maskgit.args)
        self.patch_size = maskgit.input_size # 16 or 32
        self.D = self.patch_size ** 2 # 16*16 or 32*32
        self.device = self.maskgit.args.device
        self.noise = Absorbing(self.N)
        self.guid_sched = guidance_schedule


    @torch.no_grad()
    @torch.amp.autocast(device_type='cuda')
    def qrate_fn(self, x:torch.tensor, t:torch.tensor, labels:torch.LongTensor, 
                 w:float=3, sm_temp:float=1):
        """
        x: [B, 16*16], t: [B], labels: [B]
        output: [B, 16*16, 1025]
        """
        drop = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        x_ = x.view(-1, self.patch_size, self.patch_size)
        if w != 0:
            logit = self.maskgit.vit(torch.cat([x_.clone(), x_.clone()], dim=0), # image token, [B, 16, 16]
                                    torch.cat([labels, labels], dim=0), # condition class, B
                                    torch.cat([~drop, drop], dim=0)) # drop label, B
            logit_c, logit_u = torch.chunk(logit, 2, dim=0)
            _w = self.guid_sched(t, w)[0].item()
            # Classifier Free Guidance, 
            # guidance strength is linearly increased from 0 to w
            logit = (1 + _w) * logit_c - _w * logit_u
        else:
            logit = self.maskgit.vit(x_.clone(), labels, drop_label=~drop)
        prob = torch.softmax(logit * sm_temp, -1)
        prob[x!=self.N-1] = 0
        prob.scatter_(-1, x[..., None], torch.zeros_like(prob[..., :1])) # make prob[b,d,x[b][d]] = 0

        rate = self.noise.sigma(t) / torch.expm1(self.noise.sigma_int(t))
        return rate[..., None, None] * prob


    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log()).to(categorical_probs.dtype)
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    @torch.no_grad() 
    @torch.amp.autocast(device_type='cuda')
    def _cfg_denoise(
        self,
        labels: torch.tensor,
        w: float,
        x: torch.tensor,
        t: torch.tensor,
        move_chance_t: torch.tensor,
        move_chance_s: torch.tensor,
    ):
        drop = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        x_ = x.view(-1, self.patch_size, self.patch_size)
        if w != 0:
            logit = self.maskgit.vit(torch.cat([x_.clone(), x_.clone()], dim=0), # image token, [B, 16, 16]
                                    torch.cat([labels, labels], dim=0), # condition class, B
                                    torch.cat([~drop, drop], dim=0)) # drop label, B
            logit_c, logit_u = torch.chunk(logit, 2, dim=0)
            _w = w   # * (1 - t)[...,None,None]
            # Classifier Free Guidance, 
            logit = (1 + _w) * logit_c - _w * logit_u
        else:
            logit = self.maskgit.vit(x_.clone(), labels, drop_label=~drop)

        x_theta = logit.softmax(dim=-1)
        q_xs = x_theta * (move_chance_t - move_chance_s)
        q_xs[:, :, self.N - 1] = move_chance_s[:, :, 0]
        q_xs /= move_chance_t

        # Sample from posterior
        xs = self._sample_categorical(q_xs)
        xs = torch.where(x != self.N -1, x, xs)

        return xs

    @torch.no_grad()
    def sample(self, batch_num, num_steps, labels = None, max_t=1, eps=.01, w=3, sm_temp=1):
        """
        return: codes: [B, 16*16] or [B, 32*32]; decoded images: [B, 3, 256, 256] or [B, 3, 512, 512]
        """
        self.maskgit.vit.eval()
        if labels is None:
            # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
            if batch_num >= 10:
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999] * (batch_num // 10)
            else:
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999][:batch_num]
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        ones = torch.ones((batch_num,), device=self.device)
        x = torch.full((batch_num, self.D), self.N-1, dtype=torch.int64, device=self.device)
        ts = np.linspace(max_t, eps, num_steps)

        for idx in tqdm(range(num_steps-1), leave=False):
            # ts[idx] (larger) -> ts[idx+1] (smaller)
            t=ts[idx]
            t_ten = ones * t
            h = ts[idx] - ts[idx + 1]
            sigma_t = self.noise.sigma_int(t_ten)
            sigma_s = self.noise.sigma_int(t_ten-h)
            move_chance_t = 1 - torch.exp(-sigma_t)
            move_chance_s = 1 - torch.exp(-sigma_s)
            move_chance_t = move_chance_t[:, None, None]
            move_chance_s = move_chance_s[:, None, None]


            x = self._cfg_denoise(labels, w, x, t, move_chance_t, move_chance_s)
            
    
        # At final time eps, replace mask with the most probable token
        eps_ten = eps * ones
        qmat = self.qrate_fn(x, eps_ten, labels=labels, w=w, sm_temp=sm_temp)
        masked = x == self.N - 1
        x[masked]=torch.argmax(qmat[:,:,:-1], dim=-1)[masked] # exclude the mask token
        x = x.clamp(0, self.N-2)

        return x, self.maskgit.ae.decode_code(x.view(batch_num, self.patch_size, self.patch_size))

# class RK2Sampler(BaseSampler):
#     '''
#     Second-order Runge–Kutta method. theta = 1/2: midpoint; theta = 1: Heun; theta = 2/3: Ralston.

#     See [link](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#:~:text=In%20this%20family,is%20Ralston%27s%20method)
#     '''
#     def __init__(self, maskgit:MaskGIT, noise:Noise, theta:float=.5, sample_midpt:bool=True):
#         self.maskgit = maskgit
#         self.N = self.maskgit.codebook_size + 1 # 1025
#         self.patch_size = self.maskgit.args.path_size # 16 or 32
#         self.D = self.patch_size ** 2 # 16*16 or 32*32
#         self.device = self.maskgit.args.device
#         self.noise = noise
#         # assert isinstance(theta, float) and .5<=theta<=1 
#         assert isinstance(theta, float) and 0<theta<=1 
#         self.theta = theta
#         self.sample_midpt = sample_midpt 
#         # if True, sample from Poissons; otherwise, use expectation (will generate nonsense)

#     @torch.cuda.amp.autocast()
#     def qrate_fn(self, x:torch.tensor, t:torch.tensor, labels:torch.LongTensor, 
#                  w:float=3, sm_temp:float=1):
#         """
#         x: [B, 16*16], t: [B], labels: [B]
#         output: [B, 16*16, 1025]
#         """
#         drop = torch.ones(x.shape[0], dtype=torch.bool).to(x.device)
#         x_ = x.view(-1, self.patch_size, self.patch_size)
#         if w != 0:
#             # Model Prediction
#             logit = self.maskgit.vit(torch.cat([x_.clone(), x_.clone()], dim=0), # image token, [B, 16, 16]
#                                     torch.cat([labels, labels], dim=0), # condition class, B
#                                     torch.cat([~drop, drop], dim=0)) # drop label, B
#             # drop label: the first half is False (cond sampling), the second half is True (uncond sampling)
#             # logit: [2*B, 16*16, 1025]
#             logit_c, logit_u = torch.chunk(logit, 2, dim=0)
#             # split a tensor into the specified number of chunks. 
#             _w = w * (1 - t)[...,None,None]
#             # Classifier Free Guidance, 
#             # guidance strength is linearly increased from 0 to w
#             logit = (1 + _w) * logit_c - _w * logit_u
#         else:
#             logit = self.maskgit.vit(x_.clone(), labels, drop_label=~drop)
#         # logit: [B, 16*16, 1025]
#         prob = torch.softmax(logit * sm_temp, -1) # [B, 16*16, 1025]
#         prob[x!=self.N-1]=0;# prob[:,:,-1]=0 # TODO
#         prob.scatter_(-1, x[..., None], torch.zeros_like(prob[..., :1])) # make prob[b,d,x[b][d]] = 0
#         # prob[b][d][n] != 0 only if x[b][d] == mask and n != mask

#         rate = self.noise.rate_noise(t) / torch.expm1(self.noise.total_noise(t))
#         return rate[..., None, None] * prob

#     @torch.no_grad()
#     def sample(self, batch_num, num_steps, labels = None, max_t=1, eps=.01, w=3, sm_temp=1):
#         """
#         return: codes: [B, 16*16] or [B, 32*32]; decoded images: [B, 3, 256, 256] or [B, 3, 512, 512]
#         """
#         self.maskgit.vit.eval()
#         if labels is None:  # Default classes generated
#             # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
#             if batch_num >= 10:
#                 labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999] * (batch_num // 10)
#             else:
#                 labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999][:batch_num]
#         labels = torch.tensor(labels, dtype=torch.long).to(self.device)

#         x = torch.full((batch_num, self.D), self.N-1, dtype=torch.int64).to(self.device) # [B, D]
#         ts = np.linspace(max_t, eps, num_steps) # from max_t (prior) to eps (~data)
#         # print('ts:', ts)
#         # for idx in tqdm(range(num_steps-1), leave=True):
#         for idx in range(num_steps-1):
#             # ts[idx] (larger) -> ts[idx+1] (smaller)
#             t=ts[idx]; h = ts[idx] - ts[idx + 1]
#             t_ten = torch.full((batch_num,), t).to(self.device)
#             qmat = self.qrate_fn(x, t_ten, labels=labels, w=w, sm_temp=sm_temp) # [B, D, N] TODO
#             diffs = torch.arange(self.N).view(1, 1, self.N).to(self.device) - x.view(batch_num, self.D, 1)
#             # [B, D, N], state change, diffs[b,d,n] = n - x[b,d]
#             if self.sample_midpt:
#                 jump_nums = torch.distributions.poisson.Poisson(self.theta * h * qmat).sample().to(self.device) # [B, D, N]
#                 jump_nums[jump_nums.sum(dim = -1) > 1] = 0 # TODO reject multiple jumps to the same token
#             else:
#                 jump_nums = self.theta * h * qmat # [B, D, N]
#             overall_jump = torch.sum(jump_nums * diffs, dim=-1).to(dtype=torch.int64,device=self.device)
#             x_mid = torch.clamp(x + overall_jump, min=0, max=self.N-1).to(torch.int64) # [B, D]
#             t_theta_ten = t_ten - self.theta * h
#             qmat_midstep = self.qrate_fn(x_mid, t_theta_ten, labels=labels, w=w, sm_temp=sm_temp)
#             final_qmat = (1 - 1/(2*self.theta)) * qmat + 1/(2*self.theta) * qmat_midstep
#             final_qmat = torch.clamp(final_qmat, min=0)
#             final_jump_nums = torch.distributions.poisson.Poisson(h * final_qmat).sample().to(self.device)
#             final_jump_nums[final_jump_nums.sum(dim = -1) > 1] = 0
#             final_diffs = torch.arange(self.N).view(1, 1, self.N).to(self.device) - x.view(batch_num, self.D, 1)
#             final_overall_jump = torch.sum(final_jump_nums * final_diffs, dim=-1).to(self.device)
#             x = torch.clamp(x + final_overall_jump, min=0, max=self.N-1).to(torch.int64)

#         # At final time eps, replace mask with the most probable token
#         eps_ten = torch.full((batch_num,), eps).to(self.device)
#         qmat = self.qrate_fn(x, eps_ten, labels=labels, w=w, sm_temp=sm_temp)
#         masked = x == self.N - 1
#         x[masked]=torch.argmax(qmat[:,:,:-1], dim=-1)[masked] # exclude the mask token

#         # clamp to 0~1023 and decode the final prediction
#         x = torch.clamp(x, min=0, max=self.N-2)
#         return x, self.maskgit.ae.decode_code(x.view(batch_num, self.patch_size, self.patch_size))


# class RK2TrapezoidSampler(BaseSampler):
#     '''
#     Theta-trapezoidal tau-leaping. See https://doi.org/10.1063/1.3609119
#     '''
#     def __init__(self, maskgit:MaskGIT, noise:Noise, theta:float=.5, sample_midpt:bool=True):
#         self.maskgit = maskgit
#         self.N = self.maskgit.codebook_size + 1 # 1025
#         self.patch_size = self.maskgit.args.path_size # 16 or 32
#         self.D = self.patch_size ** 2 # 16*16 or 32*32
#         self.device = self.maskgit.args.device
#         self.noise = noise
#         assert isinstance(theta, float) and 0<theta<=1
#         self.alpha1, self.alpha2 = 1/(2*theta), ((1-theta)**2+theta**2)/(2*theta) 
#         self.theta = theta
#         self.sample_midpt = sample_midpt 
#         # if True, sample from Poissons; otherwise, use expectation (will generate nonsense)

#     @torch.cuda.amp.autocast()
#     def qrate_fn(self, x:torch.tensor, t:torch.tensor, labels:torch.LongTensor, 
#                  w:float=3, sm_temp:float=1):
#         """
#         x: [B, 16*16], t: [B], labels: [B]
#         output: [B, 16*16, 1025]
#         """
#         drop = torch.ones(x.shape[0], dtype=torch.bool).to(x.device)
#         x_ = x.view(-1, self.patch_size, self.patch_size)
#         if w != 0:
#             # Model Prediction
#             logit = self.maskgit.vit(torch.cat([x_.clone(), x_.clone()], dim=0), # image token, [B, 16, 16]
#                                     torch.cat([labels, labels], dim=0), # condition class, B
#                                     torch.cat([~drop, drop], dim=0)) # drop label, B
#             # drop label: the first half is False (cond sampling), the second half is True (uncond sampling)
#             # logit: [2*B, 16*16, 1025]
#             logit_c, logit_u = torch.chunk(logit, 2, dim=0)
#             # split a tensor into the specified number of chunks. 
#             _w = w * (1 - t)[...,None,None]
#             # Classifier Free Guidance, 
#             # guidance strength is linearly increased from 0 to w
#             logit = (1 + _w) * logit_c - _w * logit_u
#         else:
#             logit = self.maskgit.vit(x_.clone(), labels, drop_label=~drop)
#         # logit: [B, 16*16, 1025]
#         prob = torch.softmax(logit * sm_temp, -1) # [B, 16*16, 1025]
#         prob[x!=self.N-1]=0;# prob[:,:,-1]=0 # TODO
#         prob.scatter_(-1, x[..., None], torch.zeros_like(prob[..., :1])) # make prob[b,d,x[b][d]] = 0
#         # prob[b][d][n] != 0 only if x[b][d] == mask and n != mask

#         rate = self.noise.rate_noise(t) / torch.expm1(self.noise.total_noise(t))
#         return rate[..., None, None] * prob

#     @torch.no_grad()
#     def sample(self, batch_num, num_steps, labels = None, max_t=1, eps=.01, w=3, sm_temp=1):
#         """
#         return: codes: [B, 16*16] or [B, 32*32]; decoded images: [B, 3, 256, 256] or [B, 3, 512, 512]
#         """
#         self.maskgit.vit.eval()
#         if labels is None:  # Default classes generated
#             # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
#             if batch_num >= 10:
#                 labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999] * (batch_num // 10)
#             else:
#                 labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999][:batch_num]
#         labels = torch.tensor(labels, dtype=torch.long).to(self.device)

#         x = torch.full((batch_num, self.D), self.N-1, dtype=torch.int64).to(self.device) # [B, D]
#         ts = np.linspace(max_t, eps, num_steps) # from max_t (prior) to eps (~data)
#         # print('ts:', ts)
#         # for idx in tqdm(range(num_steps-1), leave=True):
#         for idx in range(num_steps-1):
#             # ts[idx] (larger) -> ts[idx+1] (smaller)
#             t=ts[idx]; h = ts[idx] - ts[idx + 1]
#             t_ten = torch.full((batch_num,), t).to(self.device)
#             qmat = self.qrate_fn(x, t_ten, labels=labels, w=w, sm_temp=sm_temp) # [B, D, N] TODO
#             diffs = torch.arange(self.N).view(1, 1, self.N).to(self.device) - x.view(batch_num, self.D, 1)
#             # [B, D, N], state change, diffs[b,d,n] = n - x[b,d]
#             if self.sample_midpt:
#                 jump_nums = torch.distributions.poisson.Poisson(self.theta * h * qmat).sample().to(self.device) # [B, D, N]
#                 jump_nums[jump_nums.sum(dim = -1) > 1] = 0 # TODO reject multiple jumps to the same token
#             else:
#                 jump_nums = self.theta * h * qmat # [B, D, N]
#             overall_jump = torch.sum(jump_nums * diffs, dim=-1).to(dtype=torch.int64,device=self.device)
#             x_mid = torch.clamp(x + overall_jump, min=0, max=self.N-1).to(torch.int64) # [B, D]

#             t_theta_ten = t_ten - self.theta * h
#             qmat_midstep = self.qrate_fn(x_mid, t_theta_ten, labels=labels, w=w, sm_temp=sm_temp)
#             final_qmat = torch.clamp(self.alpha1 * qmat_midstep - self.alpha2 * qmat, min=0)
#             final_jump_nums = torch.distributions.poisson.Poisson(h * final_qmat).sample().to(self.device)
#             final_jump_nums[final_jump_nums.sum(dim = -1) > 1] = 0
#             final_diffs = torch.arange(self.N).view(1, 1, self.N).to(self.device) - x_mid.view(batch_num, self.D, 1)
#             final_overall_jump = torch.sum(final_jump_nums * final_diffs, dim=-1).to(self.device)
#             x = torch.clamp(x_mid + final_overall_jump, min=0, max=self.N-1).to(torch.int64)

#         # At final time eps, replace mask with the most probable token
#         eps_ten = torch.full((batch_num,), eps).to(self.device)
#         qmat = self.qrate_fn(x, eps_ten, labels=labels, w=w, sm_temp=sm_temp)
#         masked = x == self.N - 1
#         x[masked]=torch.argmax(qmat[:,:,:-1], dim=-1)[masked] # exclude the mask token

#         # clamp to 0~1023 and decode the final prediction
#         x = torch.clamp(x, min=0, max=self.N-2)
#         return x, self.maskgit.ae.decode_code(x.view(batch_num, self.patch_size, self.patch_size))


# class RK2TrapezoidNewSampler(BaseSampler):
#     '''
#     Theta-trapezoidal tau-leaping. See https://doi.org/10.1063/1.3609119
#     '''
#     def __init__(self, maskgit:MaskGIT, noise:Noise, theta:float=.5, sample_midpt:bool=True):
#         self.maskgit = maskgit
#         self.N = self.maskgit.codebook_size + 1 # 1025
#         self.patch_size = self.maskgit.args.path_size # 16 or 32
#         self.D = self.patch_size ** 2 # 16*16 or 32*32
#         self.device = self.maskgit.args.device
#         self.noise = noise
#         assert isinstance(theta, float) and 0<theta<=1
#         self.alpha1, self.alpha2 = 1/(2*theta), ((1-theta)**2+theta**2)/(2*theta) 
#         self.theta = theta
#         self.sample_midpt = sample_midpt 
#         # if True, sample from Poissons; otherwise, use expectation (will generate nonsense)

#     @torch.cuda.amp.autocast()
#     def qrate_fn(self, x:torch.tensor, t:torch.tensor, labels:torch.LongTensor, 
#                  w:float=3, sm_temp:float=1):
#         """
#         x: [B, 16*16], t: [B], labels: [B]
#         output: [B, 16*16, 1025]
#         """
#         drop = torch.ones(x.shape[0], dtype=torch.bool).to(x.device)
#         x_ = x.view(-1, self.patch_size, self.patch_size)
#         if w != 0:
#             # Model Prediction
#             logit = self.maskgit.vit(torch.cat([x_.clone(), x_.clone()], dim=0), # image token, [B, 16, 16]
#                                     torch.cat([labels, labels], dim=0), # condition class, B
#                                     torch.cat([~drop, drop], dim=0)) # drop label, B
#             # drop label: the first half is False (cond sampling), the second half is True (uncond sampling)
#             # logit: [2*B, 16*16, 1025]
#             logit_c, logit_u = torch.chunk(logit, 2, dim=0)
#             # split a tensor into the specified number of chunks. 
#             _w = w * (1 - t)[...,None,None]
#             # Classifier Free Guidance, 
#             # guidance strength is linearly increased from 0 to w
#             logit = (1 + _w) * logit_c - _w * logit_u
#         else:
#             logit = self.maskgit.vit(x_.clone(), labels, drop_label=~drop)
#         # logit: [B, 16*16, 1025]
#         prob = torch.softmax(logit * sm_temp, -1) # [B, 16*16, 1025]
#         prob[x!=self.N-1]=0;# prob[:,:,-1]=0 # TODO
#         prob.scatter_(-1, x[..., None], torch.zeros_like(prob[..., :1])) # make prob[b,d,x[b][d]] = 0
#         # prob[b][d][n] != 0 only if x[b][d] == mask and n != mask

#         rate = self.noise.rate_noise(t) / torch.expm1(self.noise.total_noise(t))
#         return rate[..., None, None] * prob

#     def compute_shifted_rate(self, qmat, x_mid, x):
#         shifts = x_mid - x ## (B, D)
#         shifted_indices = shifts.unsqueeze(-1) + torch.arange(self.N, device=x.device).view(1, 1, -1)
#         valid_mask = (shifted_indices >= 0) & (shifted_indices < self.N)
    
    
    
#     @torch.no_grad()
#     def sample(self, batch_num, num_steps, labels = None, max_t=1, eps=.01, w=3, sm_temp=1):
#         """
#         return: codes: [B, 16*16] or [B, 32*32]; decoded images: [B, 3, 256, 256] or [B, 3, 512, 512]
#         """
#         self.maskgit.vit.eval()
#         if labels is None:  # Default classes generated
#             # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
#             if batch_num >= 10:
#                 labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999] * (batch_num // 10)
#             else:
#                 labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, 999][:batch_num]
#         labels = torch.tensor(labels, dtype=torch.long).to(self.device)

#         x = torch.full((batch_num, self.D), self.N-1, dtype=torch.int64).to(self.device) # [B, D]
#         ts = np.linspace(max_t, eps, num_steps) # from max_t (prior) to eps (~data)
#         # print('ts:', ts)
#         # for idx in tqdm(range(num_steps-1), leave=True):
#         for idx in range(num_steps-1):
#             # ts[idx] (larger) -> ts[idx+1] (smaller)
#             t=ts[idx]; h = ts[idx] - ts[idx + 1]
#             t_ten = torch.full((batch_num,), t).to(self.device)
#             qmat = self.qrate_fn(x, t_ten, labels=labels, w=w, sm_temp=sm_temp) # [B, D, N] TODO
#             # qmat_old = qmat.clone()
#             diffs = torch.arange(self.N).view(1, 1, self.N).to(self.device) - x.view(batch_num, self.D, 1)
#             # [B, D, N], state change, diffs[b,d,n] = n - x[b,d]
#             if self.sample_midpt:
#                 jump_nums = torch.distributions.poisson.Poisson(self.theta * h * qmat).sample().to(self.device) # [B, D, N]
#                 jump_nums[jump_nums.sum(dim = -1) > 1] = 0 # TODO reject multiple jumps to the same token
#             else:
#                 jump_nums = self.theta * h * qmat # [B, D, N]
#             overall_jump = torch.sum(jump_nums * diffs, dim=-1).to(dtype=torch.int64,device=self.device)
#             x_mid = torch.clamp(x + overall_jump, min=0, max=self.N-1).to(torch.int64) # [B, D]

#             t_theta_ten = t_ten - self.theta * h
#             qmat_midstep = self.qrate_fn(x_mid, t_theta_ten, labels=labels, w=w, sm_temp=sm_temp)
            
#             ##################
#             # offset = (x_mid - x).unsqueeze(-1) + torch.arange(self.N, device=x.device).view(1, 1, -1)
#             # qmat_midstep = qmat_midstep.gather(2, offset.clamp(0, self.N-1))
#             # valid_mask = (offset >= 0) & (offset < self.N)
#             # qmat_midstep[~valid_mask] = 0
            
#             offset = (x - x_mid).unsqueeze(-1) + torch.arange(self.N, device=x.device).view(1, 1, -1)
#             qmat = qmat.gather(2, offset.clamp(0, self.N-1))
#             valid_mask = (offset >= 0) & (offset < self.N)
#             qmat[~valid_mask] = 0
            
#             # print(torch.linalg.norm(qmat - qmat_old))
#             ##################
            
#             final_qmat = torch.clamp(self.alpha1 * qmat_midstep - self.alpha2 * qmat, min=0)
#             # final_qmat_2 = torch.clamp(self.alpha1 * qmat_midstep - self.alpha2 * qmat_old, min=0)    
#             # print("final", torch.linalg.norm(final_qmat - final_qmat_2))
            
#             final_jump_nums = torch.distributions.poisson.Poisson(h * final_qmat).sample().to(self.device)
#             final_jump_nums[final_jump_nums.sum(dim = -1) > 1] = 0
#             final_diffs = torch.arange(self.N).view(1, 1, self.N).to(self.device) - x_mid.view(batch_num, self.D, 1)
#             final_overall_jump = torch.sum(final_jump_nums * final_diffs, dim=-1).to(self.device)
#             x = torch.clamp(x_mid + final_overall_jump, min=0, max=self.N-1).to(torch.int64)

#         # At final time eps, replace mask with the most probable token
#         eps_ten = torch.full((batch_num,), eps).to(self.device)
#         qmat = self.qrate_fn(x, eps_ten, labels=labels, w=w, sm_temp=sm_temp)
#         masked = x == self.N - 1
#         x[masked]=torch.argmax(qmat[:,:,:-1], dim=-1)[masked] # exclude the mask token

#         # clamp to 0~1023 and decode the final prediction
#         x = torch.clamp(x, min=0, max=self.N-2)
#         return x, self.maskgit.ae.decode_code(x.view(batch_num, self.patch_size, self.patch_size))