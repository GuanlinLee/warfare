# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random

import numpy as np

import torch
import torchvision.utils as tvu

from DiffPure.ddpm.unet_ddpm import Model
from DiffPure.ddpm.ema_helper import EMAHelper



def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *, model, logvar, betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()


        print("Loading model")
        model = Model(self.config)
        model = model.cuda()
        self.model = model

        if self.config.data.dataset == "LSUN":
            ckpt = "ema_lsun.ckpt"
        elif self.config.data.dataset == "CELEBA":
            ckpt = "celeba.pth"
        ckpt_path = os.path.abspath('.')
        ckpt_path = os.path.join(ckpt_path, "DiffPure/pretrained", ckpt)
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.abspath('..')
            ckpt_path = os.path.join(ckpt_path, "DiffPure/pretrained", ckpt)
        print("load model path is", ckpt_path)
        ckpt = torch.load(ckpt_path)
        if self.config.data.dataset == "CELEBA":
            model = torch.nn.DataParallel(model)
            model.load_state_dict(ckpt[0])
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(ckpt[-1])
            ema_helper.ema(model)
        else:
            model.load_state_dict(ckpt)
        model.eval()
        print("Model loaded")
        #exit(0)

    def image_editing_sample(self, img=None, bs_id=0, tag=None):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        with torch.no_grad():
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            out_dir = os.path.join(self.args.savepath, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim
            x0 = img
            """
            if bs_id < 2:
                os.makedirs(out_dir, exist_ok=True)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))
            """
            xs = []
            for it in range(self.args.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0).to(x0.device)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                """
                if bs_id < 2:
                    tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))
                """
                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=img.device)
                    x = image_editing_denoising_step_flexible_mask(x, t=t, model=self.model,
                                                                    logvar=self.logvar,
                                                                    betas=self.betas.to(img.device))
                    # added intermediate step vis
                    """
                    if (i - 49) % 50 == 0 and bs_id < 2:
                        tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'noise_t_{i}_{it}.png'))
                    """
                x0 = x
                """
                if bs_id < 2:
                    torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))
                """
                xs.append(x0)

            return torch.cat(xs, dim=0)

def keyname_change(para_dict):
    for k, v in para_dict.items():
        if 'time_embedding.linear_1' in k:
            k = k.replace('time_embedding.linear_1', 'temb.dense.0')
        if 'time_embedding.linear_2' in k:
            k = k.replace('time_embedding.linear_2', 'temb.dense.1')
        if 'down_blocks' in k:
            k = k.replace('down_blocks', 'down')
            if 'resnets' in k:
                k = k.replace('resnets', 'block')
            if 'time_emb_proj' in k:
                k = k.replace('time_emb_proj', 'temb_proj.weight')
            if 'downsamplers.0' in k:
                k = k.replace('downsamplers.0', 'downsample')
            if 'conv_shortcut' in k:
                k = k.replace('conv_shortcut', 'nin_shortcut')
            if 'attentions' in k:
                k = k.replace('attentions', 'attn')
            if 'group_norm' in k:
                k = k.replace('group_norm', 'norm')
            if 'query' in k:
                k = k.replace('query', 'q')
            if 'key' in k:
                k = k.replace('key', k)
            if 'value' in k:
                k = k.replace('value', 'v')
            if 'proj_attn' in k:
                k = k.replace('proj_attn', 'proj_out')
        if 'mid_block' in k:
            k = k.replace('mid_block', 'mid')
            if 'attentions.0' in k:
                k = k.replace('attentions.0', 'attn_1')
            if 'resnets.0' in k:
                k = k.replace('resnets.0', 'block_1')
            if 'resnets.1' in k:
                k = k.replace('resnets.1', 'block_2')
            if 'group_norm' in k:
                k = k.replace('group_norm', 'norm')
            if 'query' in k:
                k = k.replace('query', 'q')
            if 'key' in k:
                k = k.replace('key', k)
            if 'value' in k:
                k = k.replace('value', 'v')
            if 'proj_attn' in k:
                k = k.replace('proj_attn', 'proj_out')
            if 'time_emb_proj' in k:
                k = k.replace('time_emb_proj', 'temb_proj.weight')
        if 'up_blocks' in k:
            k = k.replace('up_blocks', 'up')
            if 'resnets' in k:
                k = k.replace('resnets', 'block')
            if 'time_emb_proj' in k:
                k = k.replace('time_emb_proj', 'temb_proj.weight')
            if 'upsamplers.0' in k:
                k = k.replace('upsamplers.0', 'downsample')
            if 'conv_shortcut' in k:
                k = k.replace('conv_shortcut', 'nin_shortcut')
            if 'attentions' in k:
                k = k.replace('attentions', 'attn')
            if 'group_norm' in k:
                k = k.replace('group_norm', 'norm')
            if 'query' in k:
                k = k.replace('query', 'q')
            if 'key' in k:
                k = k.replace('key', k)
            if 'value' in k:
                k = k.replace('value', 'v')
            if 'proj_attn' in k:
                k = k.replace('proj_attn', 'proj_out') 


