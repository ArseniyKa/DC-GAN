import torch.nn as nn
import numpy as np

# custom weights initialization called on netG and netD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def read_raw(ct_path, volume_size=30, dtype=np.uint8):

    pixels = np.fromfile(ct_path, dtype=dtype)
    stack_size = int(pixels.shape[0] / volume_size ** 2)
    raw_data_cube = pixels.reshape((stack_size, 1, volume_size, volume_size))

    return raw_data_cube


def generateBatches(batch_size, num_train):
    shuffled_indices = np.arange(num_train)
    np.random.shuffle(shuffled_indices)
    sections = np.arange(batch_size, num_train, batch_size)

    # print("sections shape is\n",sections)
    batches_indices = np.array_split(shuffled_indices, sections)
    return batches_indices
