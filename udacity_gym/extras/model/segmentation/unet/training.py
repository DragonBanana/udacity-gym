import pathlib
import lightning as pl
import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import itertools
from torch.utils.data import Dataset, DataLoader

from udacity_gym.extras.model.segmentation.unet.unet_model import SegmentationUnet
from utils.conf import ACCELERATOR, DEVICE, CHECKPOINT_DIR


class SegmentationDataset(Dataset):

    def __init__(self, dataset_dir: str, split: str = "train"):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.metadata = pd.read_csv(self.dataset_dir.joinpath('log.csv'))
        self.split = split
        if self.split == "train":
            self.metadata = self.metadata[10: int(len(self.metadata) * 0.9)]
        elif self.split == "val":
            self.metadata = self.metadata[int(len(self.metadata) * 0.9):int(len(self.metadata) * 0.95)]
        else:
            self.metadata = self.metadata[int(len(self.metadata) * 0.95):]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = Image.open(self.dataset_dir.joinpath("image", self.metadata['image_filename'].values[idx]))
        segmentation = Image.open(self.dataset_dir.joinpath("segmentation", self.metadata['segmentation_filename'].values[idx]))
        segmentation = np.array(segmentation)
        segmentation = segmentation[:,:,2:] == 255
        return self.transform(image), self.transform(segmentation).to(torch.float)


if __name__ == '__main__':

    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # Run parameters
    input_shape = (3, 160, 320)
    max_epochs = 2000
    accelerator = ACCELERATOR
    devices = [DEVICE]

    train_dataset = []
    val_dataset = []
    for track, daytime, weather in itertools.product(
            ["lake", "jungle", "mountain"],
            ["day", "daynight"],
            ["sunny"],
    ):
        train_dataset.append(
            SegmentationDataset(dataset_dir=f"../../../udacity_dataset/{track}_{weather}_{daytime}", split="train")
        )
        val_dataset.append(
            SegmentationDataset(dataset_dir=f"../../../udacity_dataset/{track}_{weather}_{daytime}", split="val")
        )

    train_loader = DataLoader(
        torch.utils.data.ConcatDataset(train_dataset),
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.ConcatDataset(val_dataset),
        batch_size=16,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="segmentation_unet",
        monitor="val/loss",
        save_top_k=1,
        verbose=True,
    )
    earlystopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=20)
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, earlystopping_callback],
        devices=devices,
    )
    model_params = {
        'hidden_dims': [32, 64, 128, 256],
        'input_shape': (3, 160, 320),
        'num_groups': 32,
        'in_channels': 3,
        'out_channels': 1,
        'learning_rate': 1e-5,
    }

    driving_model = SegmentationUnet(**model_params)
    trainer.fit(
        driving_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
