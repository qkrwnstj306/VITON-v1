import torch
import torch.nn as nn
from torchvision import transforms

from ddim import DDIMSampler

WIDTH = 384     
HEIGHT = 512   
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

class MyModel(nn.Module):
    
    def __init__(self, encoder ,decoder, diffusion, dinov2, mlp, args):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion = diffusion
        self.dinov2 = dinov2
        self.mlp = mlp
        
        self.args = args
        self.latent_shape = (self.args.batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        self.encoder.eval()
        self.decoder.eval()
        self.diffusion.eval()
        self.dinov2.eval()
        self.mlp.eval()
        
        self.generator = torch.Generator()
        self.generator.manual_seed(self.args.seed)
        self.sampler = DDIMSampler(self.generator, cfg_scale=self.args.cfg_scale)
        
        self.temperal_image_transforms = transforms.Compose([
        transforms.Resize((518,392), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        input_image = batch['input_image'].to('cpu') # [batch_size, 3, 512, 384]  
        cloth_agnostic_mask = batch['cloth_agnostic_mask'].to('cuda') # [batch_size, 3, 512, 384]
        densepose = batch['densepose'].to('cuda') # [batch_size, 3, 512, 384] 
        cloth = batch['cloth'].to('cuda') # [batch_size, 3, 512, 384]
        
        cloth_for_image_encoder = self.temperal_image_transforms(cloth)
        
        encoder_inputs = torch.cat((cloth_agnostic_mask, densepose), dim=0)
        
        encoder_noise = torch.randn((encoder_inputs.size(0), 4, LATENTS_HEIGHT, LATENTS_WIDTH)).to("cuda")
        
        cloth_agnostic_mask_latents, densepose_latents = \
            torch.chunk(self.encoder(encoder_inputs, encoder_noise), 2, dim=0)
            
        image_embeddings = self.dinov2(cloth_for_image_encoder)
        image_embeddings = self.mlp(image_embeddings)
        
        x_T = torch.randn(self.latent_shape, generator=self.generator).to("cuda")
        
        CFG = True if self.args.do_cfg else False
        
        x_0 = self.sampler.DDIM_sampling(self.diffusion, x_T, cloth_agnostic_mask_latents,
                                         densepose_latents, image_embeddings, do_cfg=CFG)
        
        predicted_images = self.decoder(x_0).to("cpu")
        cloth_agnostic_mask, densepose, cloth = cloth_agnostic_mask.to('cpu'), densepose.to('cpu'), cloth.to('cpu')
        
        return predicted_images, CFG