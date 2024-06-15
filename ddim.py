import torch, gc
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

class DDIMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120, cfg_scale: float = 7.5):
        self.generator = generator
        self.cfg_scale = cfg_scale
        
        self.T = num_training_steps
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = F.pad(self.alphas_bar[:-1],(1,0), value=1.)
        self.sigma = self.betas*((1. - self.alpha_bar_prev)/(1. - self.alphas_bar))
        
        #for training
        self.sqrt_alpha_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alphas_bar)

        #for DDIM sampling
        self.scheduling = 'uniform'
        self.tau = 20 # uniform - time step 을 몇 번 건너 뛸 건지 / exp - time step 을 몇 번 거쳐서 생성할 건지

        self.eta = 0 
        
    def gather_and_expand(self, coeff, t, xshape):
        # [1000] / [batch_size] / [batch_size, 4, 64, 48]

        # batch_size / (4, 64, 48)
        batch_size, *dims = xshape
        
        # T 개의 coeff 중에서, index가 t인 것들을 추출
        coeff_t = torch.gather(coeff, dim=0, index=t)
        
        # coeff_t를 각 batch에 곱할 수 있도록 reshape, 각 pixel마다 같은 값을 coefficient로 곱해주기 때문에 뒤의 차원이 (1,1,1)
        return coeff_t.view(batch_size, 1, 1, 1)

    def train(self, model, x_0, cloth_agnostic_mask,
              densepose, image_embeddings, warped_cloth_mask, do_cfg, use_attention_loss=False):
        # x_T: [batch_size, 4, Height / 8, Width / 8]
        # person_agnostic_mask, densepose, cloth, cloth_mask: [batch_size, 4, Height / 8, Width / 8]
        # cloth_embeddings, person_embeddings: [batch_size, 1037, 768]
        # Decoupled condition: cloth_embeddings -> SD, person_embeddings -> ControlNet
        
        # t: [batch_size]
        t = torch.randint(self.T, generator=self.generator, size=(x_0.shape[0],))
        
        for index, time in enumerate(t):
            if index == 0:
                temp = self.get_time_embedding(time)
            else:
                temp = torch.cat([temp, self.get_time_embedding(time)], dim = 0)
        
        # [batch_size, 160 * 2]
        time_embedding = temp
        
        # [batch_size, 64, 48]
        eps = torch.randn_like(x_0)
        
        # [batch_size, 4, 64, 48]
        x_t = self.gather_and_expand(self.sqrt_alpha_bar.to("cuda"), t.to("cuda"), x_0.shape) * x_0 \
        + self.gather_and_expand(self.sqrt_one_minus_alpha_bar.to("cuda"), t.to("cuda"), x_0.shape) * eps.to("cuda")
        
        if use_attention_loss:
            predicted_eps, cwg_loss, tv_loss, dcml_loss = model(x_t, cloth_agnostic_mask, densepose, image_embeddings, time_embedding.to("cuda"),
                                                    warped_cloth_mask=warped_cloth_mask, do_cfg=do_cfg, use_attention_loss=use_attention_loss)
            naive_loss = F.mse_loss(predicted_eps, eps.to("cuda"))
            
        else:
            loss = F.mse_loss(model(x_t, cloth_agnostic_mask, densepose, image_embeddings, time_embedding.to("cuda"),
                                    do_cfg=do_cfg, use_attention_loss=False), eps.to("cuda"))

        del x_t
        gc.collect()
        torch.cuda.empty_cache()
        
        if use_attention_loss:
            return naive_loss, cwg_loss, tv_loss, dcml_loss
        return loss
    
    def _get_process_scheduling(self, reverse = True):
        if self.scheduling == 'uniform':
            diffusion_process = list(range(0, len(self.alphas_bar), self.tau)) + [len(self.alphas_bar)-1]
        elif self.scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(len(self.alphas_bar)* 0.8), self.tau)** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [len(self.alphas_bar)-1]
        else:
            assert 'Not Implementation'
            
        
        diffusion_process = zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:])) if reverse else zip(diffusion_process[1:], diffusion_process[:-1])
        return diffusion_process

    def DDIM_sampling(self, model, x_T, cloth_agnostic_mask,
                      densepose, image_embeddings, do_cfg=True):
        # x_T: [batch_size, 4, Height / 8, Width / 8]
        # person_agnostic_mask, densepose, cloth, cloth_mask: [batch_size, 4, Height / 8, Width / 8]
        # cloth_embeddings, person_embeddings: [batch_size, 1037, 768]
        # Decoupled condition: cloth_embeddings -> SD, person_embeddings -> ControlNet
        
        diffusion_process = self._get_process_scheduling(reverse=True)
        batch_size = x_T.size()[0]
            
        x = x_T.clone()

        with tqdm(total=1000 // self.tau) as pbar:
            for prev_idx, idx in diffusion_process:
                # Scalar 999 -> 0
                time_step = idx
                
                # [batch_size], [batch_size]
                idx = torch.Tensor([idx for _ in range(x.size(0))]).long()
                prev_idx = torch.Tensor([prev_idx for _ in range(x.size(0))]).long()
                
                # time_embedding: [1, 160 * 2]
                time_embedding = self.get_time_embedding(time_step)
                
                model_output = model(x, cloth_agnostic_mask, densepose, image_embeddings, time_embedding.to("cuda")
                                     , is_train=False, do_cfg=do_cfg, use_attention_loss=False)
                
                if do_cfg:
                    # [batch_size, 4, Height / 8, Width / 8], [batch_size, 4, Height / 8, Width / 8]
                    output_cond, output_uncond = model_output.chunk(2)
                    eps = self.cfg_scale * (output_cond - output_uncond) + output_uncond
                else:
                    eps = model_output
                
                # [batch_size, 4, Height / 8, Width / 8]
                predicted_x0 = (x - torch.sqrt(1 - self.alphas_bar[idx].to("cuda").view(batch_size, 1, 1, 1)) * eps) / torch.sqrt(self.alphas_bar[idx].to("cuda").view(batch_size, 1, 1, 1))
                
                # [batch_size, 4, Height / 8, Width / 8]
                direction_pointing_to_xt = torch.sqrt(1 - self.alphas_bar[prev_idx].to("cuda")).view(batch_size, 1, 1, 1) * eps
                
                # [batch_size, 4 ,Height / 8, Width / 8]
                x = torch.sqrt(self.alphas_bar[prev_idx].to("cuda")).view(batch_size, 1, 1, 1) * predicted_x0 + direction_pointing_to_xt
                pbar.update(1)

        del x, direction_pointing_to_xt, model_output, eps
        gc.collect()
        torch.cuda.empty_cache()
        return predicted_x0
    
    def get_time_embedding(self, timestep):
        # Shape: (160,)
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
        # Shape: (1, 160)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
        # Shape: (1, 160 * 2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    