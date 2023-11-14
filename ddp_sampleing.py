import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from train_wrn_ebm import CCF
from fast_dit import DiT_models
import math
from piq import FID
from PIL import Image
from tqdm import tqdm
import numpy as np
from CustomDataset import CustomDataset
from torchvision.utils import make_grid, save_image

import time
from diffusion import create_diffusion
import argparse
from autoencoder import FrozenAutoencoderKL

import pdb
now = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000))
img_transformers = transforms.Compose(
    [transforms.ToTensor()]
)

class Model(nn.Module):
    def __init__(self, args, input_size, in_channels, num_class) -> None:
        super().__init__()
        self.backbone = DiT_models[args.base_model](input_size=input_size,
                                                    in_channels = in_channels)
        # self.backbone = wideresnet(depth=28, width=10, norm='layer', dropout_rate=0.0)
        self.embed_dim = self.backbone.embed_dim
        # self.head = DINOHead(in_dim=self.embed_dim, out_dim=num_class)
        # self.head = nn.Linear(self.embed_dim, num_class)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_class)
        )

    def forward(self, x, t, cls_only=True):
        
        z = self.backbone(x, t)
        out = self.head(z).squeeze()
        if cls_only:
            return out
        else:
            logits_energy = out.logsumexp(1)
            with torch.enable_grad():
                if self.training:
                    x_prime = torch.autograd.grad(logits_energy.sum(), [x], create_graph=True, retain_graph=True)[0]
                else:
                    x_prime = torch.autograd.grad(logits_energy.sum(), [x])[0]
                    x = x.detach()
            x_prime = x_prime * -1
        return x_prime



if __name__=='__main__':
    parser = argparse.ArgumentParser('Energy Based Model')
    parser.add_argument('--vae_weight', type=str, default='/remote-home/yfsong/Diffusion/dit-dino/weights/autoencoder_kl.pth')
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ckpts', type=str, default='/remote-home/yfsong/Diffusion/dit-dino/experiments/2023-11-11-08:38:23/ckpt_160.pt')
    ###############################################################
    parser.add_argument('--base_model', type=str, default='DiT-S/2')
    parser.add_argument("--sigma", type=float, default=3e-2)
    parser.add_argument('--input_size', type=int, default=32, help='Input Image shape')
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--num_class', type=int, default=100)
    #################################################################\
    parser.add_argument('--weight_classify', type=float, default=1.)
    parser.add_argument('--weight_energy', type=float, default=1.)
    #################################################################
    parser.add_argument('--debug', action='store_true')
    ###################################################################
    parser.add_argument('--ckpt_every', type=int, default=20)
    parser.add_argument('--img_every', type=int ,default=1000)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument('--wandb_name', type=str, default='Dits_JEM')
    #################################################################
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    ddconfig_f8 = dict(
        double_z=True,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0
    )
    vae_model = FrozenAutoencoderKL(ddconfig_f8, 4, args.vae_weight, 0.18215)
    device = 0
    diffusion = create_diffusion(timestep_respacing='')
    model = Model(args, input_size=args.input_size, in_channels=args.in_channels, num_class=args.num_class)
    model.load_state_dict(torch.load(args.ckpts))
    vae_model = vae_model.to(device)


    def cond_fn(x, t):    
        x = x.requires_grad_(True)
        grad = model(x, t, cls_only=False)
        return grad

    model = model.to(device)
    model.eval()
    vae_model.eval()
    for i in range(8):
        res = []
        shape = (args.test_batch_size, args.in_channels, args.input_size, args.input_size)
        x_q = diffusion.p_sample_loop(model, shape, cond_fn=cond_fn, model_kwargs={})
        test_q = vae_model.decode(x_q).cpu()
        # pdb.set_trace()
        test_q = torch.clamp(test_q, -1, 1)*0.5
        test_q = (test_q + 1)
        # sample = ((test_q + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        res.append(test_q)
    res = torch.cat(res, dim=0)
    img = make_grid(res)
    save_image(img, '/remote-home/yfsong/Diffusion/dit-dino/test_res/test.png')
