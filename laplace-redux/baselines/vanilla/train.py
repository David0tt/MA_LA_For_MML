import os, sys
# import os.path as o
# sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), '../..')))

print("path: ", os.getcwd())

import torch
import torch.nn.functional as F
from torch import optim
from utils import data_utils, test
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import math

from torch import nn

import matplotlib.pyplot as plt

from baselines.vanilla.models.lenet import LeNet
from baselines.vanilla.models.wrn import WideResNet


import inspect

print('inspect.getabsfile(data_utils): ', inspect.getabsfile(data_utils))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR10', 'FMNIST', 'ImageNet', 'HAM10000'])
parser.add_argument('--model', default=None, choices=['LeNet', 'WRN16-4', 'resnet18', 'resnet50', 'ViT_B_16', 'ViT_H_14'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--randseed', type=int, default=123)
parser.add_argument('--download', action='store_true', help='if True, downloads the datasets needed for training')
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Path: ", os.getcwd())
print("device: ", device)



if args.dataset == 'CIFAR100':
    num_classes = 100
elif args.dataset == 'ImageNet':
    num_classes = 1000
elif args.dataset == 'HAM10000':
    num_classes = 7
else:
    num_classes = 10

if args.model is None:
    if args.dataset == 'MNIST' or args.dataset == 'FMNIST':
        args.model = 'LeNet'
    elif args.dataset == 'CIFAR10':
        args.model = 'WRN16-4'
    elif args.dataset == 'ImageNet':
        args.model = 'resnet50' # 'resnet50', 'resnet152', 'resnet18'
    elif args.dataset == 'HAM10000':
        args.model = 'resnet50'

if args.model == 'LeNet':
    model = LeNet(num_classes)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    arch_name = 'lenet'
    dir_name = 'lenet_' + args.dataset.lower()
if args.model == 'WRN16-4':    
    model = WideResNet(16, 4, num_classes)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0, nesterov=True)
    arch_name = 'wrn_16-4'
    dir_name = 'wrn_16-4_' + args.dataset.lower()
if args.model == 'resnet50':
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    arch_name = 'resnet50'
    dir_name = 'resnet50_' + args.dataset.lower()
    BATCH_SIZE = 32

if args.model == 'resnet18':
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    arch_name = 'resnet18'
    dir_name = 'resnet18_' + args.dataset.lower()
    BATCH_SIZE=256

if args.model == 'ViT_H_14':
    from torchvision.models import vit_h_14, ViT_H_14_Weights
    model = vit_h_14(weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    arch_name = args.model
    dir_name = args.model + "_" + args.dataset.lower()
    BATCH_SIZE=16
    IMAGE_SIZE=518

if args.model == 'ViT_B_16':
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    model = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    arch_name = args.model
    dir_name = args.model + "_" + args.dataset.lower()
    BATCH_SIZE=6
    IMAGE_SIZE=384



print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')


# Just symlink your dataset folder into your home directory like so
# No need to change this code---this way it's more consistent
# data_path = os.path.expanduser('~/Datasets')
data_path = "./data"

if args.dataset == 'MNIST':
    train_loader, val_loader, test_loader = data_utils.get_mnist_loaders(data_path, device=device, download=args.download)
elif args.dataset == 'FMNIST':
    train_loader, val_loader, test_loader = data_utils.get_fmnist_loaders(data_path, device=device, download=args.download)
elif args.dataset == 'CIFAR10':
    train_loader, val_loader, test_loader = data_utils.get_cifar10_loaders(data_path, download=args.download)
elif args.dataset == "ImageNet":
    if args.download:
        raise ValueError("--download is not supported for ImageNet, the ImageNet dataset has to be downloaded manually")
    # setting num_workers smaller is important, otherwise I get a `Segmentation fault`
    # Not really sure why, but it has something to do with the multiprocessing library
    train_loader, val_loader, test_loader = data_utils.get_imagenet_loaders(data_path, train_batch_size=BATCH_SIZE, batch_size=BATCH_SIZE, num_workers=2)
elif args.dataset == 'HAM10000':
    if args.download:
        raise ValueError("--download is not supported for HAM10000, the dataset has to be downloaded manually")
    train_loader, val_loader, test_loader = data_utils.get_ham10000_loaders(data_path, train_batch_size=BATCH_SIZE, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)



# print()
# print(model)
# print()


model.to(device)
model.train()

loss_fn = nn.CrossEntropyLoss()

pbar = trange(args.epochs)

loss_list = []
iteration_list = []
accuracy_list = []

for epoch in pbar:
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # Testing the model
    if epoch % 1 == 0:
        total = 0
        correct = 0
    
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            conf, pred = torch.max(y_pred, 1)
            correct += (pred == y).sum()
            total += len(y)

        
        accuracy = correct * 100 / total
        loss_list.append(loss.item())
        iteration_list.append(epoch)
        accuracy_list.append(accuracy.item())
    
        pbar.set_description(
            f'[Epoch: {epoch+1}; acc: {accuracy:.1f}; loss: {loss:.3f}]'
        )


if not os.path.exists("./training_plots"):
    os.makedirs("./training_plots")

# Plotting
plt.plot(iteration_list, loss_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Loss")
plt.title("Loss")
plt.savefig(f"./training_plots/{arch_name}_{args.dataset.lower()}_{args.randseed}_loss.png")
# plt.show()

plt.plot(iteration_list, accuracy_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.savefig(f"./training_plots/{arch_name}_{args.dataset.lower()}_{args.randseed}_accuracy.png")
# plt.show()

path = f'./models/{dir_name}'

if not os.path.exists(path):
    os.makedirs(path)

save_name = f'{path}/{arch_name}_{args.dataset.lower()}_{args.randseed}'
torch.save(model.state_dict(), save_name)

## Try loading and testing
model.load_state_dict(torch.load(save_name))
model.eval()

print()















