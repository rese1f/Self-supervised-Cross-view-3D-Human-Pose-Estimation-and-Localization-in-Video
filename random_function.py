import os
import numpy as np
import torch


def random_data_source(input_path, min, max) -> list:
    """
    @brief create random data source from certain range
    @return a file list
    """
    files = os.listdir(input_path)
    random_data_source_list = \
        [files[i] for i in np.random.randint(0, len(files), size=int((np.random.rand() * (max - min) + min)))]
    return random_data_source_list


def random_data_timeline(raw, rate=0.1):
    """
    preprocess raw data into the new data.
    """
    frame = raw.shape[0]
    adjust = int(0.1 * np.random.randint(frame))
    data = raw[adjust:, :]

    return data


def random_translate(raw, translate_distance):
    """
    Translate tensor
    """
    translate = torch.zeros(raw.size())
    random_x, random_y = torch.rand(1), torch.rand(1)
    translate[:, :, 0], translate[:, :, 1] = random_x, random_y

    return raw + translate_distance * translate


def random_rotate(raw):
    """
    Rotate tensor
    """
    angle = torch.deg2rad(360 * torch.rand(1))
    r = torch.zeros(raw.size())
    r[:, :, 2] = 1
    temp = raw[0, 0, :]
    raw = raw - temp

    r2 = torch.zeros(raw.size())
    r2[:, :, 1] = -1

    raw2 = torch.zeros(raw.size())
    raw2[:, :, 1] = raw[:, :, 0]
    raw2[:, :, 2] = raw[:, :, 1]

    dot = torch.zeros(raw.size())
    dot_sum = (raw * r).sum(dim=2)
    dot[:, :, 0] = dot_sum
    dot[:, :, 1] = dot_sum
    dot[:, :, 2] = dot_sum

    raw = torch.cos(angle) * raw + (1 - torch.cos(angle)) * dot * r + torch.sin(angle) * (r2 * raw2) + temp

    return raw
