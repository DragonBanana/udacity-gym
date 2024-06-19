import lightning as pl
import numpy as np
import torch
import itertools
from torch.utils.data import DataLoader
import torchmetrics
from udacity_gym.extras.model.segmentation.unet.training import SegmentationDataset
from udacity_gym.extras.model.segmentation.unet.unet_model import SegmentationUnet
from utils.conf import ACCELERATOR, DEVICE, DEFAULT_DEVICE


if __name__ == '__main__':

    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # Run parameters
    input_shape = (3, 160, 320)
    max_epochs = 2000
    accelerator = ACCELERATOR
    devices = [DEVICE]
    checkpoint_name = "/home/banana/projects/udacity-gym/model/segmentation/unet/lightning_logs/version_9/checkpoints/segmentation_unet.ckpt.ckpt" # TODO: replace way to compute path

    metric = torchmetrics.classification.BinaryJaccardIndex().to(DEFAULT_DEVICE)
    metric_values = []

    train_dataset = []
    val_dataset = []
    for track, daytime, weather in itertools.product(
            ["lake", "jungle", "mountain"],
            ["day", "daynight"],
            ["sunny", "rainy", "snowy", "foggy"],
    ):

        driving_model = SegmentationUnet.load_from_checkpoint(checkpoint_name, map_location=DEFAULT_DEVICE)

        dataset = SegmentationDataset(dataset_dir=f"../../../udacity_dataset/{track}_{weather}_{daytime}", split="val")

        loader = DataLoader(
            dataset,
            batch_size=16,
        )

        with torch.no_grad():
            for batch in loader:
                img, true = batch
                pred = driving_model(img.to(DEFAULT_DEVICE))
                miou = metric(pred, true.to(DEFAULT_DEVICE))
                metric_values.append(miou.item())

        print(f"mIOU computed on dataset {track}-{weather}-{daytime}: {np.array(metric_values).mean()}")




