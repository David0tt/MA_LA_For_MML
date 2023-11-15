import os, sys
# import os.path as o
# sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), '../..')))


### Notes: Settings to obtain good results on HAM10000
# batch_size as large as possible without crashing
# Large Image size
# Cosine annealing
# DataParallel training for faster training
# better data augmentation



print("path: ", os.getcwd())
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
print('Imported torch')
from utils import data_utils_skinlesions as data_utils
print('Imported utils')
from tqdm import tqdm, trange
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
print('Imported tqdm, numpy, argparse, tensorboard')

import itertools

import warnings



# Parse training arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--randseed', type=int, default=1254883)
parser.add_argument('--batch_size_per_gpu', type=int, default=16)
parser.add_argument('--num_workers_per_gpu', type=int, default=6)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_schedule', type=str, default='Constant', choices=['CosineAnnealingWarmRestarts', 'Constant', 'CosineAnnealing'])
parser.add_argument('--runname', type=str, default="")
parser.add_argument('--dataset', type=str, default='SkinLesions', choices=['SkinLesions', 'WILDS-camelyon17'])
parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'wide_resnet50'])
parser.add_argument('--dryrun', action='store_true', help='Dry-run using a small dataset only for debugging')
parser.add_argument('--data_root', type=str, default='./data',
                    help='root of dataset')
args = parser.parse_args()

# Print Config
print('args: ', args)

# Set seeds
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# Logging for debugging
for i in range(torch.cuda.device_count()):
    print(f"torch.cuda.mem_get_info({i}): ", torch.cuda.mem_get_info(i))
    print(f"torch.cuda.get_device_properties({i}).total_memory: ", torch.cuda.get_device_properties(i).total_memory)
    print(f"torch.cuda.memory_reserved({i}): ", torch.cuda.memory_reserved(i))
    print(f"torch.cuda.memory_allocated({i}): ", torch.cuda.memory_allocated(i))
    print(f"torch.cuda.memory_summary({i}): ", torch.cuda.memory_summary(i))




if args.dataset == 'SkinLesions':
    num_classes = 7
if args.dataset == 'WILDS-camelyon17':
    num_classes = 2



# Get model
if args.model == 'resnet50':
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
if args.model == 'wide_resnet50':
    from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
    model = wide_resnet50_2(weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)



# Use multiple devices, if available
model= nn.DataParallel(model)
model.to(device)
model.train()
print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# LR  scheduler
if args.lr_schedule == 'CosineAnnealingWarmRestarts':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                T_0=10, T_mult=2) # Using best settings from https://arxiv.org/abs/1608.03983

elif args.lr_schedule == 'CosineAnnealing':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.lr_schedule == 'Constant':
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)

# Dataset
num_gpus = torch.cuda.device_count()
num_workers = args.num_workers_per_gpu * num_gpus
batch_size = args.batch_size_per_gpu * num_gpus

if args.dataset == 'SkinLesions':
    train_loader, val_loader, test_loader = data_utils.get_ham10000_loaders(args.data_root, train_batch_size=batch_size, 
                                                                            batch_size=batch_size, image_size=args.image_size,
                                                                            num_workers=num_workers)
    class_names = train_loader.dataset.SKINLESIONS_CLASS_TO_IDX.keys()

if args.dataset == 'WILDS-camelyon17':
    try:
        from wilds_examples.configs.utils import populate_defaults
        from wilds_examples.models.initializer import initialize_model
        from wilds_examples.transforms import initialize_transform
    except ModuleNotFoundError as e:
        print("Files from Wilds examples not found. Download the examples folder from https://github.com/p-lambda/wilds in version 1.1.0 and copy them into /laplace-redux/wilds_examples. Then adapt all local imports by prepending `wilds_examples.` where necessary")
        print("ModuleNotFoundError: ", e)
    try:
        from wilds import get_dataset
        from wilds.common.data_loaders import get_train_loader, get_eval_loader
        from wilds.common.grouper import CombinatorialGrouper
    except ModuleNotFoundError as e:
        print('WILDS library/dependencies not found -- please install following https://github.com/p-lambda/wilds.')
        print("ModuleNotFoundError: ", e)

    from torchvision import transforms, datasets
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if args.image_size > 96:
        warnings.warn(f"args.image_size is {args.image_size}, which is larger than 96, which is the size of the image patches in camelyon17, so it does not make sense")


    class ProperDataLoader:
        """ This class defines an iterator that wraps a PyTorch DataLoader 
            to only return the first two of three elements of the data tuples.

            This is used to make the data loaders from the WILDS benchmark
            (which return (X, y, metadata) tuples, where metadata for example
            contains domain information) compatible with the uq.py script and
            with the laplace library (which both expect (X, y) tuples).
        """
        def __init__(self, data_loader):
            self.data_loader = data_loader
            self.dataset = self.data_loader.dataset

        def __iter__(self):
            self.data_iter = iter(self.data_loader)
            return self

        def __next__(self):
            X, y, _ = next(self.data_iter)
            return X, y

        def __len__(self):
            return len(self.data_loader)


    image_size = args.image_size

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tforms_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    transform_train = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.RandomBrightness(limit=0.2, p=0.75),
        # A.RandomContrast(limit=0.2, p=0.75),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=0.7),

        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(image_size, image_size),
        # A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        A.OneOf([A.CoarseDropout(max_holes=4, max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), 
                        min_holes=1, min_height=1, min_width=1, fill_value=0)], p=0.7),
        A.Normalize(),
        ToTensorV2(),
    ])
    tforms_train = lambda x: transform_train(image=np.array(x))['image']



    full_dataset = get_dataset('camelyon17', root_dir=args.data_root)

    train_data = full_dataset.get_subset('train', transform=tforms_train)
    train_loader = get_train_loader(loader='standard', dataset=train_data, batch_size=batch_size, num_workers=num_workers)


    val_data = full_dataset.get_subset('id_val', transform=tforms_test)
    val_loader = get_eval_loader(loader='standard', dataset=val_data, batch_size=batch_size, num_workers=num_workers)

    test_loader = [] # There is no in-distribution test-loader for camelyon17, and we don't need one for training

    train_loader = ProperDataLoader(train_loader)
    val_loader = ProperDataLoader(val_loader)



dataloaders = {'train': train_loader,
               'val': val_loader,
            #    'test': test_loader
               }
dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']} # , 'test']}


if args.dryrun:
    for k, v in dataloaders.items():
        dataloaders[k] = itertools.islice(v, 3)


# Loss
criterion = nn.CrossEntropyLoss()



# Tensorboard logging 
runname = f'{args.dataset}_{args.model}_{args.lr_schedule}_{args.runname}'
writer = SummaryWriter(comment=runname)

# Save configuration
with open(os.path.join(writer.log_dir, 'config.txt'), 'w') as f:
    print(args, file=f)


best_model_params_path = os.path.join(writer.log_dir, f'best_{args.randseed}.pt')
checkpoint_model_params_path = os.path.join(writer.log_dir, f'ckpt_{args.randseed}.pt')




# Training

torch.save(model.module.state_dict(), best_model_params_path)

pbar = trange(args.epochs) if not args.dryrun else trange(2)
best_acc = 0.0

for epoch in pbar:
    epoch_losses = {'train': np.nan, 'val': np.nan}
    epoch_accs = {'train': np.nan, 'val': np.nan}
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            scheduler.step()
            

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
        writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
        epoch_losses[phase] = epoch_loss
        epoch_accs[phase] = epoch_acc
        pbar.set_description(
            f'[Epoch: {epoch}; Acc/train: {epoch_accs["train"]:.2f}; loss/train: {epoch_losses["train"]:.3f}; Acc/val: {epoch_accs["val"]:.2f}; loss/val: {epoch_losses["val"]:.3f}]'
        )

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.module.state_dict(), best_model_params_path)
        if epoch % 10 == 0:
            torch.save(model.module.state_dict(), checkpoint_model_params_path)



writer.flush()



# load best model weights
model.module.load_state_dict(torch.load(best_model_params_path))
model.eval()



# Testing




