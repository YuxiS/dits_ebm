
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from train_wrn_ebm import CCF
import math
from piq import FID
from PIL import Image
from tqdm import tqdm
import numpy as np
import time
import argparse

import pdb
now = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000))
img_transformers = transforms.Compose(
    [transforms.ToTensor()]
)


np.random.seed(1234)


def load_model(checkpoints_path):
    model = CCF(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=10, sp_norm=args.sp_norm)
    ckpt_dict = torch.load(checkpoints_path)
    model.load_state_dict(ckpt_dict["model_state_dict"])
    replay_buffer = ckpt_dict["replay_buffer"]
    return model, replay_buffer


def sample_init(args, replay_buffer, bs, y=None):
    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
    if args.init_way == 'buffer':
        inds = torch.randint(0, buffer_size, (bs,))
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = torch.randn((bs, args.n_ch, args.im_sz, args.im_sz))
        choose_random = (torch.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        # pdb.set_trace()
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        # return samples.to(device), inds
    else:
        samples = torch.randn((bs, args.n_ch, args.im_sz, args.im_sz))
    
    return samples.to(args.device), inds

def sample_sgld(args, energy_f, replay_buffer, y=None):
    energy_f.eval()
    bs = args.batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_init(args, replay_buffer, bs=bs, y=y)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(args.n_steps):
        f_prime = torch.autograd.grad(energy_f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
        x_k.data += args.sgld_lr * f_prime + args.sgld_std * torch.randn_like(x_k)
    energy_f.train()
    final_samples = x_k.detach()
    # update replay buffer
    # if len(replay_buffer) > 0:
    #     replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples

# @torch.no_grad()
def TestFid(args, model, replay_buffer):
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * torch.randn_like(x)]
    )
    fid = FID().to(args.device)
    # model, replay_buffer = load_model(args.checkpoints)
    # model = model.to(args.device)
    test_dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    fake_features_list = []
    real_features_list = []
    for real_data, real_lable in tqdm(test_loader):
        # pdb.set_trace()
        real_data = real_data.to(args.device)
        real_data = (torch.clamp(real_data, -1, 1)+1)/2
        fake_data = sample_sgld(args, model, replay_buffer)
        fake_data = (torch.clamp(fake_data, -1, 1)+1)/2
        # fake_data = (fake_data-fake_data.min())/fake_data.max()
        # pdb.set_trace()
        real_feature = fid.compute_feats([{'images':real_data},]).detach().cpu()
        fake_feature = fid.compute_feats([{'images':fake_data},]).detach().cpu()
        real_features_list.append(real_feature)
        fake_features_list.append(fake_feature)
    
    real_features = torch.cat(real_features_list, dim=0)
    fake_features = torch.cat(fake_features_list, dim=0)

    fid_score= fid(real_features, fake_features)
    print('FID:', fid_score)
    return fid_score.item()

if __name__=='__main__':
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=30,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment/{}'.format(now))
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000)

    parser.add_argument('--mcmc_way', type=str, default='base', help='base, simplemonment, adam, lookahead, lookadam')
    parser.add_argument('--sp_norm', default=False, help='Spertral Norm')
    # parser.add_argument('--wandb_name', default=str, required=True)
    parser.add_argument('--noise', action='store_true', help='Add Noise')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--init_way', type=str, default='buffer')
    parser.add_argument('--n_ch', type=int, default=3)
    parser.add_argument('--im_sz', type=int, default=32)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    model, replay_buffer = load_model('/remote-home/yfsong/Diffusion/JEM-master/experiment/best_valid_ckpt.pt')
    model = model.to(args.device)
    fid = TestFid(args, model, replay_buffer)
    print(fid)