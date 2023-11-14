
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
from train_jem import Model
import math
from piq import FID
from PIL import Image
from tqdm import tqdm
import numpy as np
from CustomDataset import CustomDataset

import time
from diffusion import create_diffusion
import argparse

import pdb
now = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000))
img_transformers = transforms.Compose(
    [transforms.ToTensor()]
)


np.random.seed(1234)

def load_model(args, checkpoints_path):
    model = Model(args, input_size=args.input_size, in_channels=args.in_channels, num_class=args.num_class)
    ckpt_dict = torch.load(checkpoints_path, map_location='cpu')
    # pdb.set_trace()
    model.load_state_dict(ckpt_dict["model_state_dict"])
    return model

# @torch.no_grad()
def TestFid(args, model):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    diffusion = create_diffusion(timestep_respacing='')
    # transform_test = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    # )
    fid = FID().to(device)
    model = DDP(model.to(device), device_ids=[rank])
    # test_dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=transform_test)
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, sampler=test_sampler)
    fake_features_list = []
    real_features_list = []
    for real_data, _ in tqdm(test_loader):
        # pdb.set_trace()
        torch.cuda.empty_cache()
        real_data = real_data.to(args.device)
        real_data = (torch.clamp(real_data, -1, 1)+1)/2
        fake_data = diffusion.p_sample_loop(model, real_data.shape)
        fake_data = (torch.clamp(fake_data, -1, 1)+1)/2
        real_feature = fid.compute_feats([{'images':real_data},]).detach().cpu()
        fake_feature = fid.compute_feats([{'images':fake_data},]).detach().cpu()
        real_features_list.append(real_feature)
        fake_features_list.append(fake_feature)
        torch.cuda.empty_cache()
    if args.local_rank==0:
        real_features = torch.cat(real_features_list, dim=0)
        fake_features = torch.cat(fake_features_list, dim=0)
        fid_score= fid(real_features, fake_features)
        print('FID:', fid_score)
    # return fid_score.item()

if __name__=='__main__':
    parser = argparse.ArgumentParser('Energy Based Model')
    parser.add_argument('--data_root', type=str, default='./data', help='Data dir')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--save_dir", type=str, default='./experiments/{}'.format(now))
    parser.add_argument('--lr', type=float, default=0.0001)
    ###############################################################
    parser.add_argument('--base_model', type=str, default='DiT-B/4')
    # parser.add_argument("--patch_size", type=int, default=4)
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
    parser.add_argument('--device', type=int, default=0)
    #################################################################
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # torch.cuda.set_device(args.device)
    model = load_model(args, '/remote-home/yfsong/Diffusion/dit-dino/experiments/2023-11-01-14:20:35/last_ckpt.pt')
    # pdb.set_trace()
    # model = model.to(args.device)
    fid = TestFid(args, model)
    # print(fid)