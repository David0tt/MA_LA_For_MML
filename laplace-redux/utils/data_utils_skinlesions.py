import os
from PIL import Image
import numpy as np


import torch
import torch.utils.data as data_utils
from torchvision import transforms, datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset

import pandas as pd



class HAM10000Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

        # -> Class order is: MEL, NV, BCC, AKIEC, BKL, DF, VASC
        # -> Class order of trained classifier is: akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'}
        self.LABELTRANSLATE = {0: 4, 1: 5, 2: 1, 3: 0, 4: 2, 5: 3, 6: 6}

        self.SKINLESIONS_CLASS_TO_IDX = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + '.jpg')
        # image = read_image(img_path)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        label = np.argmax(self.img_labels.iloc[idx, 1:])
        label = self.LABELTRANSLATE[label]
        return image, label


def get_ham10000_loaders(data_path, batch_size=128, train_batch_size=128, num_workers=4, image_size=512):
    print("num_workers: ", num_workers) # TODO remove

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tforms_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # tforms_train = transforms.Compose([
    #     transforms.RandomResizedCrop((image_size, image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(20),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),        
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # more elaborate augmentations, copied from https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412
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

    data_path = os.path.join(data_path, 'HAM10000')

    data_path_train = os.path.join(data_path, 'ISIC2018_Task3_Training_Input')
    data_path_val = os.path.join(data_path, 'ISIC2018_Task3_Validation_Input')
    data_path_test = os.path.join(data_path, 'ISIC2018_Task3_Test_Input')

    annotation_path_train = os.path.join(data_path, 'ISIC2018_Task3_Training_GroundTruth', 'ISIC2018_Task3_Training_GroundTruth.csv')
    annotation_path_val = os.path.join(data_path, 'ISIC2018_Task3_Validation_GroundTruth', 'ISIC2018_Task3_Validation_GroundTruth.csv')
    annotation_path_test = os.path.join(data_path, 'ISIC2018_Task3_Test_GroundTruth', 'ISIC2018_Task3_Test_GroundTruth.csv')
    

    train_set = HAM10000Dataset(annotation_path_train, data_path_train, transform=tforms_train)
    val_set = HAM10000Dataset(annotation_path_val, data_path_val, transform=tforms_test)
    test_set = HAM10000Dataset(annotation_path_test, data_path_test, transform=tforms_test)


    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=train_batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         pin_memory=True)
    val_loader = data_utils.DataLoader(val_set,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=True)
    test_loader = data_utils.DataLoader(test_set,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=True)

    return train_loader, val_loader, test_loader


def get_ham10000c_loaders(data_path, batch_size=128, train_batch_size=128, num_workers=4, image_size=512):
    # TODO we could follow the way the corupptions were generated in https://github.com/ZerojumpLine/Robust-Skin-Lesion-Classification/tree/main/skinlesiondatasets
    # which is for 30% of the HAM10000 set
    # alternatively we could generate corruptions for the ~1500 items HAM10000 test set that was released after the competition
    raise NotImplementedError
    # return train_loader, val_loader, test_loader


# SKINLESIONS_CLASS_TO_IDX = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

# def get_skinlesions_loaders(data_path, batch_size=128, train_batch_size=128, num_workers=4, image_size=256):
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ResizeTest = transforms.Resize(image_size)

#     transform_val = transforms.Compose([
#             ResizeTest,
#             transforms.ToTensor(),
#             normalize,
#         ])

#     tforms_train = transforms.Compose([
#         transforms.RandomResizedCrop((image_size, image_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(20),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),        
#         transforms.ToTensor(),
#         normalize,
#     ])

#     data_folder = os.path.join(data_path, 'SkinLesionDatasets')
#     train_dataset = datasets.ImageFolder(root=os.path.join(data_folder, 'HAMtrain'), transform=tforms_train)
#     val_dataset = datasets.ImageFolder(root=os.path.join(data_folder, 'HAMtest'), transform=transform_val)

#     val_dataset.class_to_idx = SKINLESIONS_CLASS_TO_IDX
#     train_dataset.class_to_idx = SKINLESIONS_CLASS_TO_IDX

#     train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True,
#             num_workers=num_workers, pin_memory=True)

#     val_loader = torch.utils.data.DataLoader(
#             val_dataset, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True)

#     # TODO From the ISIC website we can download a dedicated test-set that was released after the competition
#     data_path = os.path.join(data_path, 'HAM10000')
#     data_path_test = os.path.join(data_path, 'ISIC2018_Task3_Test_Input')
#     annotation_path_test = os.path.join(data_path, 'ISIC2018_Task3_Test_GroundTruth', 'ISIC2018_Task3_Test_GroundTruth.csv')
#     test_set = HAM10000Dataset(annotation_path_test, data_path_test, transform=transform_val)
#     in_test_loader = data_utils.DataLoader(test_set,
#                                        batch_size=batch_size,
#                                        num_workers=num_workers,
#                                        pin_memory=True)

#     return train_loader, val_loader, in_test_loader

def get_SkinLesions_ood_loader(ood_dataset, data_path='./data', batch_size=128, num_workers=4, image_size=512):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ResizeTest = transforms.Resize(image_size)
    ResizeTest = transforms.Resize((image_size, image_size))

    transform_val = transforms.Compose([
            ResizeTest,
            transforms.ToTensor(),
            normalize,
        ])

    datasetlist = ['BCN', 'D7P', 'MSK', 'PH2', 'SON', 'UDA', 'VIE']

    data_folder = os.path.join(data_path, 'SkinLesionDatasets')

    test_datasets = []
    for datasetn in datasetlist:
        test_dataset = datasets.ImageFolder(root=os.path.join(data_folder, datasetn), transform=transform_val)
        # val_dataset.class_to_idx = train_dataset.class_to_idx # TODO there could be an indexing error from not doing this
        test_dataset.class_to_idx = SKINLESIONS_CLASS_TO_IDX
        test_datasets.append(test_dataset)

    for d in test_datasets:
        image, label = d[0]
        print("image.shape: ", image.shape)
    
    test_dataset = data_utils.ConcatDataset(test_datasets)
    
    print("Got after test_dataset = data_utils.ConcatDataset(test_datasets)")

    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)


    return test_loader
