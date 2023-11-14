import torch
import torch.nn as nn
from fast_dit import DiT_models
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
from accelerate import Accelerator
from CustomDataset import CustomDataset
from autoencoder import FrozenAutoencoderKL
import pdb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.autograd.set_detect_anomaly(True)
now = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000))
os.environ['TORCH_HOME'] = '/remote-home/yfsong/.cache/torch/hub/checkpoints'

def get_data(args):
    # transform_train = transforms.Compose(
    #         [
    #          transforms.Pad(4, padding_mode="reflect"),
    #          transforms.RandomCrop(args.input_size),
    #          transforms.RandomHorizontalFlip(),
    #          transforms.ToTensor(),
    #          transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    #          lambda x: x + args.sigma * torch.randn_like(x)]
    #     )
    # transform_test = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    #      lambda x: x + args.sigma * torch.randn_like(x)])
    
    dataset_train = CustomDataset(data_root=args.data_root)
    dataset_test = CustomDataset(data_root=args.test_root)
    dataset_ebm = CustomDataset(data_root=args.data_root)
    # dataset_train = datasets.CIFAR10(root=args.data_root, train=True, transform=transform_train)
    # dataset_test = datasets.CIFAR10(root=args.data_root, train=False, transform=transform_test)
    # dataset_ebm = datasets.CIFAR10(root=args.data_root, train=True, transform=transform_train)

    dloader_train = DataLoader(dataset_train, 
                               batch_size=args.batch_size, 
                               num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=True)
    dloader_test =DataLoader(dataset_test, 
                               batch_size=args.batch_size, 
                               num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=True)
    dloader_ebm = DataLoader(dataset_ebm, 
                               batch_size=args.batch_size, 
                               num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=True)
    return dloader_train, cycle(dloader_ebm), dloader_test
    
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

def cycle(loader):
    while True:
        for data in loader:
            yield data

def eval_classification(model, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        y_p_d = y_p_d.squeeze(1)
        times = torch.from_numpy(np.random.choice(1, size=(x_p_d.shape[0], ))).long().to(device)
        logits = model(x_p_d, times, cls_only=True)
        loss = nn.CrossEntropyLoss(reduction='none', reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss

# def checkpoint(f, tag, args):
#     ckpt_dict = {
#         "model_state_dict": f.module.state_dict(),
#     }
#     torch.save(ckpt_dict, os.path.join(args.save_dir, tag))

def uploadImg(x):
    x = (x + 1) * 0.5
    x = torchvision.utils.make_grid(x, nrow=int(math.sqrt(int(x.shape[0]))), normalize=True)
    ndarr = x.mul(255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr

def main(args):
    # dist.init_process_group("nccl")
    
    dataloader_train, dataloader_ebm, dataloader_test = get_data(args)
    # rank = dist.get_rank()
    # seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(1234)
    # device = rank % torch.cuda.device_count()
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
    accelerator = Accelerator()
    if not args.debug and accelerator.is_main_process:
        wandb.init(
            project='JEM-5',
            name=args.wandb_name,
            config=vars(args)
        )
    device  = accelerator.device
    torch.cuda.set_device(device)
    # scaler = torch.cuda.amp.GradScaler()
    
    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(f'{args.save_dir}/params.txt', 'w') as f:
            json.dump(args.__dict__, f)
    model = Model(args, input_size=args.input_size, in_channels=args.in_channels, num_class=args.num_class)
    model = model.to(device)
    vae_model = vae_model.to(device)
    diffusion = create_diffusion(timestep_respacing='')
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    optim_sheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=50, T_mult=2, eta_min=1e-6)
    model.train()
    model, optim, dataloader_train, optim_sheduler = accelerator.prepare(model, optim, dataloader_train, optim_sheduler)
    # model = accelerator.unwrap_model(model, keep_fp32_wrapper=True)

    cur_iter = 0
    
    # accelerator.save(model.module.state_dict(), os.path.join(args.save_dir, 'ckpt_a.pt'))
    for epoch in range(args.epochs):
        if accelerator.is_main_process and not args.debug:
            wandb.log(
                {'lr':optim.param_groups[0]["lr"]}
            )
        
        for i, (x_train, y_train) in tqdm(enumerate(dataloader_train), disable=not accelerator.is_local_main_process):
            L = 0.
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.squeeze(1)
            t_train = torch.randint(0, diffusion.num_timesteps, (x_train.shape[0], ), device=device)
            x_ebm, y_ebm = dataloader_ebm.__next__()
            x_ebm, y_ebm = x_ebm.to(device).detach(), y_ebm.to(device)
            y_ebm = y_ebm.squeeze(1)
            with accelerator.autocast():
                t_clean = torch.zeros(size=(x_train.shape[0],), device=device, dtype=torch.long)
                noise = torch.randn_like(x_train, requires_grad=True)
                x_train_noise = diffusion.q_sample(x_train, t_train, noise)
                timesteps = torch.cat((t_clean, t_train), dim=0)
                x = torch.cat((x_train, x_train_noise), dim=0).detach()
                y = torch.cat((y_train, y_train.clone()), dim=0)
                sqrt_alphas_cumprod = torch.from_numpy(diffusion.sqrt_alphas_cumprod).to(device)[timesteps]
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
            accelerator.backward(L)
            optim.step()
            if cur_iter % args.print_every==0 and accelerator.is_main_process:
                print('epoch/step:{}/{}, Loss:{} Class Loss: {} EBM Loss: {}'.format(epoch, i, L.item(), (loss_classify*sqrt_alphas_cumprod).mean().item(), loss_energy.mean().item()))
                # print('epoch/step:{}/{}, Loss:{} EBM Loss: {}'.format(epoch, i, L.item(), loss_energy.mean().item()))
                
                if not args.debug:
                    wandb.log({'Loss': L.item(),
                            'Class Loss':(loss_classify*sqrt_alphas_cumprod).mean().item(),
                            'EBM Loss':loss_energy.mean().item()})
            cur_iter += 1

            if (cur_iter-1) % args.img_every == 0 and accelerator.is_main_process:
                model.eval()
                # with torch.no_grad():
                shape = (args.test_batch_size, args.in_channels, args.input_size, args.input_size)
                x_q = diffusion.p_sample_loop(model, shape)
                test_q = vae_model.decode(x_q)
                if not args.debug:
                        wandb.log({'x_q':wandb.Image(uploadImg(test_q), caption='x_q_{}_{:>06d}'.format(epoch, i))})
                model.train()
        optim_sheduler.step()

        accelerator.wait_for_everyone()
        if (epoch) % args.ckpt_every == 0 and accelerator.is_main_process:
            accelerator.save(model.module.state_dict(), os.path.join(args.save_dir, f'ckpt_{epoch}.pt'))

        if accelerator.is_main_process and epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # test set
                correct, loss = eval_classification(model, dataloader_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
                if not args.debug:
                    wandb.log({'Test Loss': loss})
                    wandb.log({'Test Acc': correct})
            model.train()
        accelerator.save(model.module.state_dict(), os.path.join(args.save_dir, "last_ckpt.pt"))

if __name__=='__main__':
    parser = argparse.ArgumentParser('Energy Based Model')
    parser.add_argument('--data_root', type=str, default='/remote-home/share/yfsong/imageNet/train', help='Data dir')
    parser.add_argument('--test_root', type=str, default='/remote-home/share/yfsong/imageNet/val', help='Test Data dir')
    parser.add_argument('--vae_weight', type=str, default='/remote-home/yfsong/Diffusion/dit-dino/weights/autoencoder_kl.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
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
    main(args)




    