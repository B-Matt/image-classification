import os
import sys
import torch
import wandb
import sklearn
import pathlib
import argparse
import torchvision

import numpy as np
import albumentations as A
import segmentation_models_pytorch.utils.meter as meter

from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from utils.dataset import DatasetType, ImageDataset
from utils.early_stopping import YOLOEarlyStopping

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Trainer:
    def __init__(self, args, model) -> None:
        assert model is not None

        self.args = args
        self.model = model
        self.device = torch.device('cuda:1')
        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=np.ceil(len(self.train_loader.dataset) / self.args.batch_size) * self.args.epochs,
            eta_min=self.args.lr_min
        )
        self.early_stopping = YOLOEarlyStopping(patience=30)

        self.metrics = [
            sklearn.metrics.f1_score,
            sklearn.metrics.accuracy_score,
            sklearn.metrics.precision_score,
            sklearn.metrics.recall_score,
        ]
        self.loss_meter = meter.AverageValueMeter()
        self.metrics_meters = { metric.__name__: meter.AverageValueMeter() for metric in self.metrics }

    def get_augmentations(self):
        self.train_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=800, interpolation=1),
                A.PadIfNeeded(min_height=800, min_width=800, border_mode=0, value=(0, 0, 0), p=1.0),

                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.5),
                A.CoarseDropout(
                    max_holes=6, max_height=12, max_width=12, min_holes=1, p=0.5
                ),
                A.ShiftScaleRotate(shift_limit=0.09, rotate_limit=0, p=0.2),
                A.OneOf(
                    [
                        A.GridDistortion(distort_limit=0.1, p=0.5),
                        A.OpticalDistortion(distort_limit=0.08, shift_limit=0.4, p=0.5),
                    ],
                    p=0.6
                ),
                A.Perspective(scale=(0.02, 0.07), p=0.5),

                # Color transforms
                A.ColorJitter(
                    brightness=0, contrast=0, saturation=0.12, hue=0.01, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.05, 0.20), contrast_limit=(-0.05, 0.20), p=0.6
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 20.0), p=0.5),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.02, 0.09), p=0.5),
                    ],
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(5, 7), p=0.39),
                ToTensorV2()
            ]
        )

        self.val_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=self.args.patch_size, interpolation=1),
                A.PadIfNeeded(min_height=self.args.patch_size, min_width=self.args.patch_size, border_mode=0, value=(0,0,0), p=1.0),
                ToTensorV2(),
            ],
        )

    def get_loaders(self):
        self.train_dataset = ImageDataset(
            data_dir=r'data',
            img_dir=r'imgs',
            type=DatasetType.TRAIN,
            patch_size=self.args.patch_size,
            transform=self.train_transforms
        )
        # self.val_dataset = ImageDataset(
        #     data_dir=r'data',
        #     img_dir=r'imgs',
        #     type=DatasetType.VALIDATION,
        #     patch_size=self.args.patch_size,
        #     transform=self.val_transforms
        # )

        # # Get Loaders
        # self.train_loader = DataLoader(
        #     self.train_dataset,
        #     num_workers=self.args.workers,
        #     batch_size=self.args.batch_size,
        #     pin_memory=self.args.pin_memory,
        #     shuffle=True,
        #     drop_last=True,
        #     persistent_workers=True
        # )
        # self.val_loader = DataLoader(
        #     self.val_dataset,
        #     batch_size=self.args.batch_size,
        #     num_workers=self.args.workers,
        #     pin_memory=self.args.pin_memory,
        #     shuffle=False,
        #     drop_last=False,
        #     persistent_workers=True
        # )

    def train(self):
        torch.backends.cudnn.benchmark = True
        self.model.train()

        log.info(f'''[TRAINING]:
            Model:           {self.args.model}
            Encoder:         {self.args.encoder}
            Resolution:      {self.args.patch_size}x{self.args.patch_size}
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batch_size}
            Learning rate:   {self.args.lr}
            Min. Learning rate:   {self.args.lr_min}
            Training size:   {int(len(self.train_dataset))}
            Validation size: {int(len(self.val_dataset))}
            Checkpoints:     {self.args.save_checkpoints}
            Device:          {self.device.type}
            Mixed Precision: {self.args.use_amp}
        ''')

        wandb_log = wandb.init(project='benchmark-article', entity='firebot031')
        wandb_log.config.update(
            dict(
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                learning_rate=self.args.lr,
                save_checkpoint=self.args.save_checkpoints,
                patch_size=self.args.patch_size,
                amp=self.args.use_amp,
                weight_decay=self.args.weight_decay,
                adam_epsilon=self.args.adam_eps,
                encoder=self.args.encoder,
                model=self.args.model,
            )
        )

        self.run_name = wandb.run.name if wandb.run.name is not None else f'{self.args.model}-{self.args.encoder}-{self.args.batch_size}-{self.args.patch_size}'
        save_path = pathlib.Path('checkpoints/benchmark-article', self.run_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        criterion = torch.nn.CrossEntropyLoss().to(device=self.device)

        global_step = 0
        last_best_score = float('inf')
        masks_pred = []

        for epoch in range(self.start_epoch, self.args.epochs):
            val_loss = 0.0
            progress_bar = tqdm(total=int(len(self.train_dataset)), desc=f'Epoch {epoch + 1}/{self.args.epochs}', unit='img', position=0)

            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)

                # Get Batch Of Images
                x = batch['image'].to(self.device, non_blocking=True)
                y_true = batch['label'].to(self.device, non_blocking=True)

                # Predict
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    y_pred = self.model(x)
                    loss = criterion(masks_pred, y_true)

                # Scaler & Scheduler Update
                grad_scaler.scale(loss).backward()
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.scheduler.step()

                # Statistics
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_true, y_pred, average="micro").cpu().detach()
                    self.metrics_meters[metric_fn.__name__].add(metric_value)

                metrics_logs = {
                    k: v.mean for k, v in self.metrics_meters.items()
                }
                self.loss_meter.add(loss.item())
                print(metrics_logs)

                # Show batch progress to terminal
                progress_bar.update(x.shape[0])
                global_step += 1

            # Update Progress Bar
            progress_bar.set_postfix(**{'Loss': self.loss_meter.mean})
            progress_bar.close()

        # Push average training metrics
        wandb_log.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr-min', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--adam-eps', nargs='+', type=float, default=1e-3, help='Adam epsilon')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay that is used for AdamW')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add_argument('--patch-size', type=int, default=800, help='Patch size')
    parser.add_argument('--pin-memory', type=bool, default=True, help='Use pin memory for DataLoader?')
    parser.add_argument('--eval-step', type=int, default=1, help='Run evaluation every # step')
    parser.add_argument('--load-model', action='store_true', help='Load model from directories?')
    parser.add_argument('--save-checkpoints', action='store_true', help='Save checkpoints after every epoch?')
    parser.add_argument('--use-amp', action='store_true', help='Use Pytorch Automatic Mixed Precision?')
    args = parser.parse_args()

    model = torchvision.models.efficientnet_v2_s()
    trainer = Trainer(args, model)
    torch.cuda.empty_cache()

    # try:
    #     trainer.train()
    # except KeyboardInterrupt:
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)

    # End Training
    logging.info('[TRAINING]: Training finished!')
    torch.cuda.empty_cache()

    model = None
    trainer = None
