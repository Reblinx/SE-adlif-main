import math
import os
from typing import Optional

import torch
import torch.utils.data

# from tonic import DiskCachedDataset
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl
import urllib3
from datasets.utils.pad_tensors import PadTensors
import numpy as np

import scipy.io
import torch
import hydra
from pathlib import Path
import requests

def convert_dataset_wtime(mat_data):
    X = mat_data["x"]
    Y = mat_data["y"]
    t = mat_data["t"]
    Y = np.argmax(Y[:, :, :], axis=-1)
    d1, d2 = t.shape

    # dt = np.zeros((size(t[:, 1]), size(t[1, :])))
    dt = np.zeros((d1, d2))
    for trace in range(d1):
        dt[trace, 0] = 1
        dt[trace, 1:] = t[trace, 1:] - t[trace, :-1]

    return dt, X, Y


class ECGLDM(pl.LightningDataModule):
    files_to_url = {
        'QTDB/QTDB_test.mat': 'https://raw.githubusercontent.com/byin-cwi/Efficient-spiking-networks/main/data/QTDB_test.mat',
        'QTDB/QTDB_train.mat': 'https://raw.githubusercontent.com/byin-cwi/Efficient-spiking-networks/main/data/QTDB_train.mat'
    }
    url = 'https://api.github.com/repos/byin-cwi/Efficient-spiking-networks/contents/data'
    
    @staticmethod
    def download_if_not_exist(base_path:Path, files_to_url: dict[str, str]):
        files_names = list(files_to_url.keys())
        if not (base_path / str(files_names[0])).exists():
            for k, v in files_to_url.items():
                response = requests.get(v)
                # Check if the request was successful
                if response.status_code == 200:
                    file_path = base_path / k
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with file_path.open('wb') as file:
                        file.write(response.content)
                else:
                    raise requests.exceptions.ConnectionError
        
            
    def __init__(
        self,
        data_path: str,
        cache_root: str = None,
        valid_fraction=0.1,
        random_seed=42,
        batch_size=256,
        num_workers=0,
        burn_in_time=0,
        num_classes: int = 6,  # for hydra
    ) -> None:
        super().__init__()
        if not os.path.isabs(data_path):
            cwd = hydra.utils.get_original_cwd()
            data_path = os.path.abspath(os.path.join(cwd, data_path))
        self.data_path = Path(data_path)

        self.cache_root = data_path if cache_root is None else cache_root
        self.valid_fraction = valid_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.burn_in_time = burn_in_time

        self.collate_fn = PadTensors(batch_first=True)
        self.generator = torch.Generator().manual_seed(random_seed)
        
        self.data_path.mkdir(exist_ok=True, parents=True)
        self.download_if_not_exist(self.data_path, self.files_to_url)

        
    def prepare_data(self):
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist"
        assert os.path.exists(
            os.path.join(self.data_path, "QTDB/QTDB_train.mat")
        ), f"Data path {self.data_path} does not contain QTDB/QTDB_train.mat"
        assert os.path.exists(
            os.path.join(self.data_path, "QTDB/QTDB_test.mat")
        ), f"Data path {self.data_path} does not contain QTDB/QTDB_test.mat"
        

    def setup(self, stage: Optional[str] = None) -> None:
        train_mat = scipy.io.loadmat(
            os.path.join(self.data_path, "QTDB/QTDB_train.mat")
        )
        test_mat = scipy.io.loadmat(os.path.join(self.data_path, "QTDB/QTDB_test.mat"))
        train_dt, train_x, train_y = convert_dataset_wtime(train_mat)
        test_dt, test_x, test_y = convert_dataset_wtime(test_mat)
        self.train_val_dataset_ = QTDB_ECGWrapper(
            ds=TensorDataset(
                torch.from_numpy(train_x * 1.0), torch.from_numpy(train_y)
            ),
            burn_in_time=self.burn_in_time,
        )
        self.data_test = QTDB_ECGWrapper(
            ds=TensorDataset(torch.from_numpy(test_x * 1.0), torch.from_numpy(test_y)),
            burn_in_time=self.burn_in_time,
        )
        valid_len = math.floor(len(self.train_val_dataset_) * self.valid_fraction)
        self.data_train, self.data_val = torch.utils.data.random_split(
            self.train_val_dataset_,
            [len(self.train_val_dataset_) - valid_len, valid_len],
            generator=self.generator,
        )
        

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            pin_memory=True,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


# CODE FROM Bojian Yin, Federico Corradi, and Sander M. Bohté. “Accurate and Efficient Time-Domain Classifica-
# tion with Adaptive Spiking Recurrent Neural Networks”. In: Nature Machine Intelligence 3.10 (Oct.
# 2021), pp. 905–913.

class QTDB_ECGWrapper(Dataset):
    def __init__(self, ds, burn_in_time) -> None:
        self.burn_in_time = burn_in_time
        self.ds = ds

    def __getitem__(self, index):
        x, y = self.ds[index]
        block_idx = np.arange(start=1, stop=len(y) + 1)
        block_idx[:self.burn_in_time] = 0

        return (
            x.float(),
            y,
            torch.from_numpy(block_idx),
        )

    def __len__(self):
        return len(self.ds)
