import pathlib
import random

import lightning as pl
import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import itertools
from torch.utils.data import Dataset, DataLoader

from model.lane_keeping.dave.dave_model import Dave2
from model.segmentation.unet.unet_model import SegmentationUnet
from utils.conf import ACCELERATOR, DEVICE, DEFAULT_DEVICE, CHECKPOINT_DIR, PROJECT_DIR

def random_flip(x, y):
    if random.random() > 0.5:
        return torchvision.transforms.functional.hflip(x), torchvision.transforms.functional.hflip(y)
    else:
        return x, y

class SegmentationDataset(Dataset):

    def __init__(self, dataset_dir: str, split: str = "train"):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.metadata = pd.read_csv(self.dataset_dir.joinpath('log.csv'))
        self.split = split
        if self.split == "train":
            self.metadata = self.metadata[10: int(len(self.metadata) * 0.9)]
        elif self.split == "val":
            self.metadata = self.metadata[int(len(self.metadata) * 0.9):int(len(self.metadata) * 0.95)]
        elif self.split == "test":
            self.metadata = self.metadata[int(len(self.metadata) * 0.95):]
        self.x_transform = torchvision.transforms.Compose([
            torchvision.transforms.AugMix(),
            torchvision.transforms.ToTensor(),
        ])
        self.y_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = Image.open(self.dataset_dir.joinpath("image", self.metadata['image_filename'].values[idx]))
        segmentation = Image.open(self.dataset_dir.joinpath("segmentation", self.metadata['segmentation_filename'].values[idx]))
        segmentation = np.array(segmentation)
        segmentation = segmentation[:,:,2:] == 255
        if self.split == "train":
            return random_flip(self.x_transform(image), self.y_transform(segmentation).to(torch.float))
        else:
            return self.x_transform(image), self.y_transform(segmentation).to(torch.float)


if __name__ == '__main__':

    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # Run parameters
    input_shape = (3, 160, 320)
    max_epochs = 2000
    accelerator = ACCELERATOR
    devices = [DEVICE]

    # train_dataset = []
    # val_dataset = []
    # for track, daytime, weather in itertools.product(
    #         ["lake", "jungle", "mountain"],
    #         ["day", "daynight"],
    #         ["sunny",] #"snowy", "rainy", "foggy"],
    # ):
    #     print("cp ", PROJECT_DIR.joinpath(f"udacity_dataset/{track}_{weather}_{daytime}", "log.csv"), PROJECT_DIR.joinpath(f"udacity_dataset/inpainting_{track}_{weather}_{daytime}", "log.csv"))
    #     print("cp -Tr ", PROJECT_DIR.joinpath(f"udacity_dataset/{track}_{weather}_{daytime}", "segmentation/"),
    #           PROJECT_DIR.joinpath(f"udacity_dataset/inpainting_{track}_{weather}_{daytime}", "segmentation/"))
    for track, daytime, weather in itertools.product(
            ["lake", "jungle", "mountain"],
            ["day", "daynight"],
            ["sunny"],
    ):
        # TODO: fix path names
        train_dataset.append(
            SegmentationDataset(dataset_dir=PROJECT_DIR.joinpath(f"udacity_dataset/{track}_{weather}_{daytime}"), split="train")
        )
        train_dataset.append(
            SegmentationDataset(dataset_dir=PROJECT_DIR.joinpath(f"udacity_dataset/inpainting_{track}_{weather}_{daytime}"), split="train")
        )
        val_dataset.append(
            SegmentationDataset(dataset_dir=PROJECT_DIR.joinpath(f"udacity_dataset/{track}_{weather}_{daytime}"), split="val")
        )
        val_dataset.append(
            SegmentationDataset(dataset_dir=PROJECT_DIR.joinpath(f"udacity_dataset/inpainting_{track}_{weather}_{daytime}"), split="val")
        )

    train_loader = DataLoader(
        torch.utils.data.ConcatDataset(train_dataset),
        batch_size=32,
        shuffle=True,
        prefetch_factor=2,
        num_workers=8,
    )
    val_loader = DataLoader(
        torch.utils.data.ConcatDataset(val_dataset),
        batch_size=16,
        prefetch_factor=2,
        num_workers=8,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR.joinpath("segmentation", "unet"),
        filename="segmentation_unet_{epoch}_{step}_{val_mIoU}_{val_loss}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=-1,
        verbose=True,
    )
    earlystopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, earlystopping_callback],
        devices=devices,
    )
    model_params = {
        'hidden_dims': [64, 128, 256],
        'input_shape': (3, 160, 320),
        'num_groups': 32,
        'in_channels': 3,
        'out_channels': 1,
        'learning_rate': 1e-5,
    }

    seg_model = SegmentationUnet(**model_params)
    trainer.fit(
        seg_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path=CHECKPOINT_DIR.joinpath("segmentation", "unet", "segmentation_unet_epoch=92_step=94023_val_mIoU=0.9812600612640381_val_loss=0.019919371232390404.ckpt"),
    )
