import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import time
from tqdm.contrib.itertools import product
from lfxai.utils.datasets import ECG5000

from pathlib import Path


def old_loop(save_dir, dataset_size, disable_tqdm):
    attribution = torch.zeros((dataset_size, dataset_size))
    for test_idx, train_idx in product(
            range(dataset_size), range(dataset_size), leave=False, disable=disable_tqdm
        ):
            train_grad = torch.load(
                save_dir
                / f"train_checkpoint0_grad{train_idx}.pt"
            )
            test_grad = torch.load(
                save_dir
                / f"test_checkpoint0_grad{test_idx}.pt"
            )
            attribution[test_idx, train_idx] += torch.dot(
                train_grad.flatten(), test_grad.flatten()
            )
    return attribution

def unravelled(save_dir, dataset_size, disable_tqdm):
    attribution = torch.zeros((dataset_size, dataset_size))

    for test_idx in tqdm(range(dataset_size), disable=disable_tqdm):
        test_grad = torch.load(
                save_dir
                / f"test_checkpoint0_grad{test_idx}.pt"
            )
        for train_idx in tqdm(range(dataset_size), disable=disable_tqdm):
            train_grad = torch.load(
                save_dir
                / f"train_checkpoint0_grad{train_idx}.pt"
            )
            attribution[test_idx, train_idx] += torch.dot(
                train_grad.flatten(), test_grad.flatten()
            )
    return attribution

if __name__ == "__main__":
    save_dir = Path.cwd() / "results/ecg5000/consistency_examples/tracin_grads/"

    start = time.time()
    old_loop(save_dir, 1000, disable_tqdm=False)
    print(f"Naive takes {time.time() - start}")

    start = time.time()
    attr1 = old_loop(save_dir, 1000, disable_tqdm=True)
    print(f"No TDQM takes {time.time() - start}")

    start = time.time()
    attr2 = unravelled(save_dir, 1000, disable_tqdm=False)
    print(f"Unravelled with TDQM takes {time.time() - start}")

    assert torch.allclose(attr1, attr2), "Unravelled function is different from original."

    start = time.time()
    unravelled(save_dir, 1000, disable_tqdm=True)
    print(f"Unravelled NO TDQM takes {time.time() - start}")





