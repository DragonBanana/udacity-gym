import pathlib
import random
import lightning as pl
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import Dataset, DataLoader

from udacity_gym.extras.model.lane_keeping.chauffeur.chauffeur_model import Chauffeur
from utils.conf import ACCELERATOR, DEVICE, DEFAULT_DEVICE, CHECKPOINT_DIR, PROJECT_DIR

pl.seed_everything(42)
torch.set_float32_matmul_precision('high')


def random_flip(x, y):
    if random.random() > 0.5:
        return torchvision.transforms.functional.hflip(x), -y
    return x, y


class DrivingDataset(Dataset):

    def __init__(self, dataset_dir: str, split: str = "train", transform=None):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.metadata = pd.read_csv(self.dataset_dir.joinpath('log.csv'))
        self.split = split
        if self.split == "train":
            self.metadata = self.metadata[10: int(len(self.metadata) * 0.9)]
        else:
            self.metadata = self.metadata[int(len(self.metadata) * 0.9):]
        if transform == None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.AugMix(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = Image.open(self.dataset_dir.joinpath("image", self.metadata['image_filename'].values[idx]))
        steering = self.metadata['predicted_steering_angle'].values[idx]
        steering = torch.tensor([steering], dtype=torch.float32)
        if self.split == "train":
            image, steering = random_flip(image, steering)
        return self.transform(image), steering


if __name__ == '__main__':
    # Run parameters
    input_shape = (3, 160, 320)
    max_epochs = 2000
    accelerator = ACCELERATOR
    devices = [DEVICE]
    dataset_paths = [
        'udacity_dataset_lake',
        'udacity_dataset_lake_8_8_1',
        'udacity_dataset_lake_12_8_1',
        'udacity_dataset_lake_12_12_1',
    ]

    train_dataset = torch.utils.data.ConcatDataset([
        DrivingDataset(dataset_dir=PROJECT_DIR.joinpath(dataset, "lake_sunny_day"), split="train")
        for dataset in dataset_paths
    ])
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        prefetch_factor=4,
        num_workers=16,
    )

    val_dataset = torch.utils.data.ConcatDataset([
        DrivingDataset(dataset_dir=PROJECT_DIR.joinpath(dataset, "lake_sunny_day"), split="val", transform=torchvision.transforms.ToTensor())
        for dataset in dataset_paths
    ])
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True,
        prefetch_factor=2,
        num_workers=8,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR.joinpath("lane_keeping", "chauffeur"),
        filename="chauffeur",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
    )
    earlystopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=20)
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, earlystopping_callback],
        devices=devices,
    )

    driving_model = Chauffeur()
    trainer.fit(
        driving_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path="/media/banana/data/models/udacity-gym/lane_keeping/dave2/dave2-v3.ckpt",
    )
