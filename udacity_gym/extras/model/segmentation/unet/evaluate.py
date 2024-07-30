import pathlib
import lightning as pl
import numpy as np
import pandas as pd
import torch
import torchvision.transforms
import tqdm
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import itertools
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from model.segmentation.unet.training import SegmentationDataset
from model.lane_keeping.dave.dave_model import Dave2
from model.segmentation.unet.unet_model import SegmentationUnet
from utils.conf import ACCELERATOR, DEVICE, DEFAULT_DEVICE, CHECKPOINT_DIR

if __name__ == '__main__':

    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # Run parameters
    input_shape = (3, 160, 320)
    max_epochs = 2000
    accelerator = ACCELERATOR
    devices = [DEVICE]
    # TODO: fix checkpoint path
    # checkpoint_name = CHECKPOINT_DIR.joinpath("segmentation", "unet", "segmentation_unet_step=178947_val/mIoU=0.7033489346504211_val/loss=0.0048842825926840305.ckpt")
    # checkpoint_name = CHECKPOINT_DIR.joinpath("segmentation", "unet", "segmentation_unet_step=4044_val/mIoU=0.6062168478965759_val/loss=0.01580829918384552.ckpt")
    # checkpoint_name = CHECKPOINT_DIR.joinpath("segmentation", "unet", "segmentation_unet_step=14154_val/mIoU=0.6437448859214783_val/loss=0.00950705911964178.ckpt")
    # checkpoint_name = CHECKPOINT_DIR.joinpath("segmentation", "unet", "segmentation_unet_epoch=68_step=69759_val_mIoU=0.9794721603393555_val_loss=0.022616790607571602.ckpt")
    # checkpoint_name = CHECKPOINT_DIR.joinpath("segmentation", "unet", "segmentation_unet_epoch=84_step=85935_val_mIoU=0.9816617369651794_val_loss=0.019545618444681168.ckpt")
    checkpoint_name = CHECKPOINT_DIR.joinpath("segmentation", "unet", "segmentation_unet_epoch=142_step=289146_val_mIoU=0.9765028953552246_val_loss=0.0236128531396389.ckpt")

    miou_metric = torchmetrics.classification.BinaryJaccardIndex().to(DEFAULT_DEVICE)
    prc_metric = torchmetrics.classification.BinaryPrecision().to(DEFAULT_DEVICE)
    rec_metric = torchmetrics.classification.BinaryRecall().to(DEFAULT_DEVICE)
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(DEFAULT_DEVICE)
    cm_metric = torchmetrics.classification.BinaryConfusionMatrix().to(DEFAULT_DEVICE)

    metric_values = {
        'miou': [],
        'acc': [],
        'rec': [],
        'prc': [],
        'TP': [],
        'TN': [],
        'FP': [],
        'FN': [],
    }

    train_dataset = []
    val_dataset = []
    for track, daytime, weather in itertools.product(
            ["lake", "jungle", "mountain"],
            ["day", "daynight"],
            ["sunny", "rainy", "snowy", "foggy"],
    ):

        driving_model = SegmentationUnet.load_from_checkpoint(checkpoint_name, map_location=DEFAULT_DEVICE)

        dataset = SegmentationDataset(dataset_dir=f"../../../udacity_dataset/inpainting_{track}_{weather}_{daytime}", split="test")

        loader = DataLoader(
            dataset,
            batch_size=16,
            prefetch_factor=2,
            num_workers=4,
        )
        i = 0
        stored_segmentation_dir = dataset.dataset_dir.joinpath("computed_segmentation")
        stored_segmentation_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm.tqdm(loader):
                img, true = batch
                true = true.to(DEFAULT_DEVICE)
                img = img.to(DEFAULT_DEVICE)
                pred = driving_model(img)
                for img in pred:
                    torchvision.utils.save_image(img, stored_segmentation_dir.joinpath(dataset.metadata['segmentation_filename'].values[i]))
                    i = i + 1
                miou = miou_metric(pred, true)
                acc = acc_metric(pred, true)
                prc = prc_metric(pred, true)
                rec = rec_metric(pred, true)
                cm = cm_metric(pred, true)
                metric_values['miou'].append(miou.detach().cpu())
                metric_values['acc'].append(acc.detach().cpu())
                metric_values['rec'].append(rec.detach().cpu())
                metric_values['prc'].append(prc.detach().cpu())
                metric_values['TP'].append(cm[1][1].detach().cpu())
                metric_values['TN'].append(cm[0][0].detach().cpu())
                metric_values['FN'].append(cm[1][0].detach().cpu())
                metric_values['FP'].append(cm[0][1].detach().cpu())

        print(f"mIoU computed on dataset {track}-{weather}-{daytime}: {np.array(metric_values['miou']).mean()}")
        print(f"acc computed on dataset {track}-{weather}-{daytime}: {np.array(metric_values['acc']).mean()}")
        print(f"rec computed on dataset {track}-{weather}-{daytime}: {np.array(metric_values['rec']).mean()}")
        print(f"prc computed on dataset {track}-{weather}-{daytime}: {np.array(metric_values['prc']).mean()}")
        print(f"TP computed on dataset {track}-{weather}-{daytime}: {np.array(metric_values['TP']).mean()}")
        print(f"TN computed on dataset {track}-{weather}-{daytime}: {np.array(metric_values['TN']).mean()}")
        print(f"FN computed on dataset {track}-{weather}-{daytime}: {np.array(metric_values['FN']).mean()}")
        print(f"FP computed on dataset {track}-{weather}-{daytime}: {np.array(metric_values['FP']).mean()}")