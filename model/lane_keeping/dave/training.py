import pathlib
import lightning as pl
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import Dataset, DataLoader

from model.lane_keeping.dave.dave_model import Dave2

pl.seed_everything(42)
torch.set_float32_matmul_precision('high')


class DrivingDataset(Dataset):

    def __init__(self, dataset_dir: str, split: str = "train"):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.metadata = pd.read_csv(self.dataset_dir.joinpath('log.csv'))
        self.split = split
        if self.split == "train":
            self.metadata = self.metadata[10: int(len(self.metadata) * 0.9)]
        else:
            self.metadata = self.metadata[int(len(self.metadata) * 0.9):]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = Image.open(self.dataset_dir.joinpath("image", self.metadata['image_filename'].values[idx]))
        steering = self.metadata['predicted_steering_angle'].values[idx]
        steering = torch.tensor([steering], dtype=torch.float32)
        return self.transform(image), steering


if __name__ == '__main__':
    # Run parameters
    input_shape = (3, 160, 320)
    max_epochs = 2000
    accelerator = "cpu"
    # devices = [0]

    train_dataset = DrivingDataset(dataset_dir="../../../dataset", split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_dataset = DrivingDataset(dataset_dir="../../../dataset", split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True
    )

    checkpoint_callback = ModelCheckpoint(
        # dirpath=checkpoint_path.parent,
        filename="dave2.ckpt",
        monitor="val/loss",
        save_top_k=1,
        verbose=True,
    )
    earlystopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=20)
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, earlystopping_callback],
        # devices=devices,
    )

    driving_model = Dave2()
    trainer.fit(
        driving_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
