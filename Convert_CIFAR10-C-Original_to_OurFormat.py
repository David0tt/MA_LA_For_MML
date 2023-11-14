# prior to running this, download CIFAR10-C from https://zenodo.org/record/2535967#.YrMov5DMLX0
# un-tar and move the files into the CIFAR10_C_ORIGINAL_DIR

import os
print(os.getcwd())

import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

CIFAR10_DIR = "/mnt/qb/work/hennig/hmx148/MastersThesisCode/laplace-redux/data/cifar-10-batches-py"
CIFAR10_C_ORIGINAL_DIR = "/mnt/qb/work/hennig/hmx148/MastersThesisCode/laplace-redux/data/CIFAR-10-C-ORIGINAL/CIFAR-10-C"
CIFAR10_C_TARGET_DIR = "/mnt/qb/work/hennig/hmx148/MastersThesisCode/laplace-redux/data/CIFAR-10-C"

if not os.path.exists(CIFAR10_C_TARGET_DIR):
    os.makedirs(CIFAR10_C_TARGET_DIR)

corruption_files = os.listdir(CIFAR10_C_ORIGINAL_DIR)
corruption_files.remove("labels.npy")

# load original CIFAR10 data
dat = unpickle(os.path.join(CIFAR10_DIR, "test_batch"))

labels_orig = np.asarray(dat[b"labels"], dtype="uint8")
data_orig = dat[b"data"]


labels_corrupted = np.load(os.path.join(CIFAR10_C_ORIGINAL_DIR, "labels.npy"))
assert np.alltrue(labels_orig == labels_corrupted[:10000])
assert np.alltrue(labels_orig == labels_corrupted[10000:20000])
assert np.alltrue(labels_orig == labels_corrupted[20000:30000])
assert np.alltrue(labels_orig == labels_corrupted[30000:40000])
assert np.alltrue(labels_orig == labels_corrupted[40000:50000])

# load corrupted CIFAR data
cifar10_c_data = []
for f in corruption_files:
    cifar10_c_data.append(np.load(os.path.join(CIFAR10_C_ORIGINAL_DIR, f)))

# CIFAR10_c0.npy is just the original CIFAR data repeated multiple times to match the labels
data_corruption_0 = np.concatenate([data_orig for _ in range(len(corruption_files))])

labels = np.concatenate([labels_orig for _ in range(len(corruption_files))])

assert data_corruption_0.shape[0] == labels.shape[0]

red = data_corruption_0[:, 0:1024].reshape(190000, 32, 32)
green = data_corruption_0[:, 1024:2048].reshape(190000, 32, 32)
blue = data_corruption_0[:, 2048:3072].reshape(190000, 32, 32)

data_corruption_0 = np.stack([red, green, blue], axis=3)

np.save(os.path.join(CIFAR10_C_TARGET_DIR, "CIFAR10_c0.npy"), data_corruption_0)
np.save(os.path.join(CIFAR10_C_TARGET_DIR, "CIFAR10_c_labels.npy"), labels)

for corruption in range(1, 6):
    idx = (corruption - 1) * 10000
    corrupted_data = np.concatenate([d[idx:idx+10000] for d in cifar10_c_data])
    assert corrupted_data.shape[0] == labels.shape[0]
    np.save(os.path.join(CIFAR10_C_TARGET_DIR, "CIFAR10_c" + str(corruption) + ".npy"), corrupted_data)



## Manual Checking
# import matplotlib.pyplot as plt

# dat = np.load(os.path.join(CIFAR10_C_TARGET_DIR, "CIFAR10_c1.npy"))
# plt.imshow(dat[3])


# dat = np.load(os.path.join(CIFAR10_C_TARGET_DIR, "CIFAR10_c0.npy"))
# plt.imshow(dat[3])