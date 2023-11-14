import torch
import torch.nn as nn
from models import DiT_models, DINOHead
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision import transforms
import torchvision
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
import time
from tqdm import tqdm
import argparse
import math
import logging
import os
import json
from diffusion import create_diffusion
import wideresnet
import utils
import wandb
from torch.cuda.amp import autocast as autocast
import pdb


torch.autograd.set_detect_anomaly(True)
now = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000))
os.environ['TORCH_HOME'] = '/remote-home/yfsong/.cache/torch/hub/checkpoints'

def get_data(args):
    transform_train = transforms.Compose(
            [
             transforms.Pad(4, padding_mode="reflect"),
             transforms.RandomCrop(args.input_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * torch.randn_like(x)]
        )
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * torch.randn_like(x)])
    
    
    dataset_train = datasets.CIFAR10(root=args.data_root, train=True, transform=transform_train)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
    dataset_test = datasets.CIFAR10(root=args.data_root, train=False, transform=transform_test)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test, shuffle=True)
    dataset_ebm = datasets.CIFAR10(root=args.data_root, train=True, transform=transform_train)
    sampler_ebm = torch.utils.data.DistributedSampler(dataset_ebm, shuffle=True)
    dloader_train = DataLoader(dataset_train, 
                               batch_size=args.batch_size, 
                               num_workers=args.num_workers,
                               sampler=sampler_ebm,
                               pin_memory=True,
                               drop_last=True)
    dloader_test =DataLoader(dataset_test, 
                               batch_size=args.batch_size, 
                               num_workers=args.num_workers,
                               sampler=sampler_test,
                               pin_memory=True,
                               drop_last=True)
    dloader_ebm = DataLoader(dataset_ebm, 
                               batch_size=args.batch_size, 
                               num_workers=args.num_workers,
                               sampler=sampler,
                               pin_memory=True,
                               drop_last=True)
    return dloader_train, cycle(dloader_ebm), dloader_test
    
class Model(nn.Module):
    def __init__(self, args, input_size, in_channels, num_class) -> None:
        super().__init__()
        # self.backbone = DiT_models[args.base_model](input_size=input_size,
                                                    # in_channels = in_channels)
        self.backbone = wideresnet(depth=28, width=10, norm='layer', dropout_rate=0.0)
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

def cycle(loader):
    while True:
        for data in loader:
            yield data

def eval_classification(model, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.cuda(), y_p_d.cuda()
        times = torch.from_numpy(np.random.choice(1, size=(x_p_d.shape[0], ))).long().to(device)
        logits = model(x_p_d, times, cls_only=True)
        loss = nn.CrossEntropyLoss(reduction='none', reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss

def checkpoint(f, tag, args):
    ckpt_dict = {
        "model_state_dict": f.module.state_dict(),
    }
    torch.save(ckpt_dict, os.path.join(args.save_dir, tag))

def uploadImg(x):
    x = torch.clamp(x, -1, 1)
    x = torchvision.utils.make_grid(x, nrow=int(math.sqrt(int(x.shape[0]))), normalize=True)
    ndarr = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr

def main(args):
    dist.init_process_group("nccl")
    dataloader_train, dataloader_ebm, dataloader_test = get_data(args)
    rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    scaler = torch.cuda.amp.GradScaler()
    if rank==0:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(f'{args.save_dir}/params.txt', 'w') as f:
            json.dump(args.__dict__, f)
    model = Model(args, input_size=args.input_size, in_channels=args.in_channels, num_class=args.num_class)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing='')
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    model.train()

    cur_iter = 0
    
    
    for epoch in range(args.epochs):
        for i, (x_train, y_train) in tqdm(enumerate(dataloader_train)):
            L = 0.
            x_train, y_train = x_train.cuda(), y_train.cuda()
            t_train = torch.randint(0, diffusion.num_timesteps, (x_train.shape[0], ), dtype=torch.long, device=device)
            x_ebm, y_ebm = dataloader_ebm.__next__()
            x_ebm, y_ebm = x_ebm.to(device).detach(), y_ebm.to(device)

            with autocast():
                t_clean = torch.from_numpy(np.random.choice(1, size=(x_train.shape[0], ))).long().to(device)
                noise = torch.randn_like(x_train, requires_grad=True)
                x_train_noise = diffusion.q_sample(x_train, t_train, noise)
                timesteps = torch.cat((t_clean, t_train), dim=0)
                x = torch.cat((x_train, x_train_noise), dim=0).detach()
                y = torch.cat((y_train, y_train.clone()), dim=0)
                sqrt_alphas_cumprod = torch.from_numpy(diffusion.sqrt_alphas_cumprod).to(device)[timesteps].float()
                logits = model(x, timesteps, cls_only=True)
                loss_classify = nn.CrossEntropyLoss(reduction='none')(logits, y)
                L = L + args.weight_classify*(loss_classify*sqrt_alphas_cumprod).mean()
                ###############################################################
                t_ebm = torch.randint(0, diffusion.num_timesteps, (x_ebm.shape[0], ), device=device)
                noise_ebm = torch.randn_like(x_ebm, requires_grad=True)
                x_ebm_noise = diffusion.q_sample(x_ebm, t_ebm, noise_ebm)
                x_grad = model(x_ebm_noise, t=t_ebm, cls_only=False)
                loss_energy = utils.mean_flat((noise_ebm-x_grad)**2)
                L = L + (args.weight_energy*loss_energy).mean()
            optim.zero_grad()
            scaler.scale(L).backward()
            scaler.step(optim)
            scaler.update()
            if cur_iter % args.print_every==0 and rank==0:
                print('Loss:{} Class Loss: {} EBM Loss: {}'.format(L.item(), (loss_classify*sqrt_alphas_cumprod).mean().item(), loss_energy.mean().item()))
                if not args.debug:
                    wandb.log({'Loss': L.item(),
                            'Class Loss':(loss_classify*sqrt_alphas_cumprod).mean().item(),
                            'EBM Loss':loss_energy.mean().item()})
            cur_iter += 1

            if cur_iter % args.img_every == 0 and rank == 0:
                model.eval()
                # with torch.no_grad():
                shape = (args.batch_size, 3, args.input_size, args.input_size)
                x_q = diffusion.p_sample_loop(model, shape)
                if not args.debug:
                        wandb.log({'x_q':wandb.Image(uploadImg(x_q), caption='x_q_{}_{:>06d}'.format(epoch, i))})
                model.train()

        if (epoch) % args.ckpt_every == 0 and rank==0:
            checkpoint(model, f'ckpt_{epoch}.pt', args)

        if rank==0 and epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # test set
                correct, loss = eval_classification(model, dataloader_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
                if not args.debug:
                    wandb.log({'Test Loss': loss})
                    wandb.log({'Test Acc': correct})
            model.train()
        checkpoint(model, "last_ckpt.pt", args)

if __name__=='__main__':
    parser = argparse.ArgumentParser('Energy Based Model')
    parser.add_argument('--data_root', type=str, default='./data', help='Data dir')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--save_dir", type=str, default='./experiments/{}'.format(now))
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)  
    ###############################################################
    parser.add_argument('--base_model', type=str, default='DiT-S/2')
    parser.add_argument("--sigma", type=float, default=3e-2)
    parser.add_argument('--input_size', type=int, default=32, help='Input Image shape')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_class', type=int, default=10)
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
    if not args.debug and args.local_rank==0:
        wandb.init(
            project='JEM-5',
            name=args.wandb_name,
            config=vars(args)
        )
    main(args)




    