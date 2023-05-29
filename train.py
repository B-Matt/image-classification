import os
import sys
import torch
import wandb
import sklearn
import pathlib
import datetime
import argparse
import torchvision

import numpy as np
import albumentations as A

from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from utils.meter import AverageValueMeter
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
        self.start_epoch = 0
        self.check_best_cooldown = 0

        self.get_augmentations()
        self.get_loaders()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        self.early_stopping = YOLOEarlyStopping(patience=30)

        self.metrics = [
            sklearn.metrics.f1_score,
            sklearn.metrics.accuracy_score,
            sklearn.metrics.precision_score,
            sklearn.metrics.recall_score,
        ]
        self.loss_meter = AverageValueMeter()
        self.train_metrics_meters = { metric.__name__: AverageValueMeter() for metric in self.metrics }
        self.val_metrics_meters = { metric.__name__: AverageValueMeter() for metric in self.metrics }

    def get_augmentations(self):
        self.train_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=self.args.patch_size, interpolation=1),
                A.PadIfNeeded(min_height=self.args.patch_size, min_width=self.args.patch_size, border_mode=0, value=(0, 0, 0), p=1.0),

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
                A.PadIfNeeded(min_height=self.args.patch_size, min_width=self.args.patch_size, border_mode=0, value=(0, 0, 0), p=1.0),
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
        self.val_dataset = ImageDataset(
            data_dir=r'data',
            img_dir=r'imgs',
            type=DatasetType.VALIDATION,
            patch_size=self.args.patch_size,
            transform=self.val_transforms
        )

        # Get Loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.args.workers,
            batch_size=self.args.batch_size,
            pin_memory=self.args.pin_memory,
            shuffle=True,
            drop_last=True,
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            shuffle=False,
            drop_last=False,
            persistent_workers=True
        )

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not self.args.save_checkpoints:
            return

        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module

        state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch
        }

        if is_best is False:
            # log.info('[SAVING MODEL]: Model checkpoint saved!')
            torch.save(state, pathlib.Path('checkpoints/benchmark-article', self.run_name, 'checkpoint.pth.tar'))

        if is_best:
            log.info('[SAVING MODEL]: Saving checkpoint of best model!')
            torch.save(state, pathlib.Path('checkpoints/benchmark-article', self.run_name, 'best-checkpoint.pth.tar'))

    def load_checkpoint(self, path: pathlib.Path):
        log.info('[LOADING MODEL]: Started loading model checkpoint!')
        best_path = pathlib.Path(path, 'best-checkpoint.pth.tar')

        if best_path.is_file():
            path = best_path
        else:
            path = pathlib.Path(path, 'checkpoint.pth.tar')

        if not path.is_file():
            return

        state_dict = torch.load(path)
        self.start_epoch = state_dict['epoch']
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.optimizer.name = state_dict['optimizer_name']
        log.info(
            f"[LOADING MODEL]: Loaded model with stats: epoch ({state_dict['epoch']}), time ({state_dict['time']})")

    @torch.inference_mode()
    def validate(self, epoch, wandb_log):
        self.model.eval()

        metrics = [
            sklearn.metrics.f1_score,
            sklearn.metrics.accuracy_score,
            sklearn.metrics.precision_score,
            sklearn.metrics.recall_score,
        ]
        loss_meter = AverageValueMeter()
        criterion = torch.nn.BCEWithLogitsLoss().to(device=self.device)
        criterion = criterion.to(device=self.device)

        with torch.no_grad():
            for (images, target) in tqdm(self.val_loader, total=len(self.val_loader), desc='Validation', position=1, unit='batch', leave=False):
                # Get Batch Of Images
                x = images.to(self.device, non_blocking=True)
                y_true = target.to(self.device, non_blocking=True).float().view(-1, 1)

                y_pred = self.model(x)
                loss = criterion(y_pred, y_true)

                y_pred = torch.sigmoid(y_pred) >= 0.5
                y_true = y_true == 1.0

                for metric_fn in metrics:
                    if metric_fn.__name__ == 'accuracy_score':
                        metric_value = metric_fn(y_true.cpu().detach(), y_pred.cpu().detach())
                    else:
                        metric_value = metric_fn(y_true.cpu().detach(), y_pred.cpu().detach(), average='micro', zero_division=0)
                    self.val_metrics_meters[metric_fn.__name__].add(metric_value)

        metrics_logs = {
            k: v.mean for k, v in self.val_metrics_meters.items()
        }
        loss_meter.add(loss.item())

        wandb_log.log({
            'Loss [validation]': loss_meter.mean,
            'F1-Score [validation]': metrics_logs['f1_score'],
            'Accuracy Score [validation]': metrics_logs['accuracy_score'],
            'Precision Score [validation]': metrics_logs['precision_score'],
            'Recall Score [validation]': metrics_logs['recall_score'],
        }, step=epoch)

        self.model.train()
        return loss_meter.mean

    def train(self):
        torch.backends.cudnn.benchmark = True
        self.model.train()

        log.info(f'''[TRAINING]:
            Resolution:      {self.args.patch_size}x{self.args.patch_size}
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batch_size}
            Learning rate:   {self.args.lr}
            Min. Learning rate:   {self.args.lr_min}
            Training size:   {int(len(self.train_dataset))}
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
            )
        )

        self.run_name = wandb.run.name if wandb.run.name is not None else f'{self.args.batch_size}-{self.args.patch_size}'
        save_path = pathlib.Path('checkpoints/benchmark-article', self.run_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        criterion = torch.nn.BCEWithLogitsLoss().to(device=self.device)

        global_step = 0
        last_best_score = float('inf')

        for epoch in range(self.start_epoch, self.args.epochs):
            val_loss = 0.0
            progress_bar = tqdm(total=int(len(self.train_dataset)), desc=f'Epoch {epoch + 1}/{self.args.epochs}', unit='img', position=0)

            for i, (images, target) in enumerate(self.train_loader):
                # Get Batch Of Images
                x = images.to(self.device, non_blocking=True)
                y_true = target.to(self.device, non_blocking=True).float().view(-1, 1)

                # Predict
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    y_pred = self.model(x)
                    loss = criterion(y_pred, y_true)

                # Scaler & Scheduler Update
                
                grad_scaler.scale(loss).backward()
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Metrics
                y_pred = torch.sigmoid(y_pred) >= 0.5
                y_true = y_true == 1.0

                for metric_fn in self.metrics:
                    if metric_fn.__name__ == 'accuracy_score':
                        metric_value = metric_fn(y_true.cpu().detach(), y_pred.cpu().detach())
                    else:
                        metric_value = metric_fn(y_true.cpu().detach(), y_pred.cpu().detach(), average='micro', zero_division=0)
                    self.train_metrics_meters[metric_fn.__name__].add(metric_value)

                # Show batch progress to terminal
                progress_bar.update(x.shape[0])
                global_step += 1

                # Evaluation of training
                eval_step = (int(len(self.train_dataset)) // (self.args.eval_step * self.args.batch_size))
                if eval_step > 0 and global_step % eval_step == 0:
                    val_loss = self.validate(epoch, wandb_log)
                    self.early_stopping(epoch, val_loss)

                    if epoch >= self.check_best_cooldown and val_loss < last_best_score:
                        self.save_checkpoint(epoch, True)
                        last_best_score = val_loss

                    # Mean all metrics
                    metrics_logs = {
                        k: v.mean for k, v in self.train_metrics_meters.items()
                    }
                    self.loss_meter.add(loss.item())

            # Update Scheduler
            self.scheduler.step()

            # Saving last model
            if self.args.save_checkpoints:
                self.save_checkpoint(epoch, False)

            # Early Stopping
            if self.early_stopping.early_stop:
                self.save_checkpoint(epoch, False)
                log.info(f'[TRAINING]: Early stopping training at epoch {epoch}!')
                break

            # Update WANDB
            try:
                wandb_log.log({
                    'Images [training]': {
                        'Image': wandb.Image(
                            x[:3].cpu(),
                            caption=f'Ground Truth: {y_true.tolist()[:3]}, Prediction: {y_pred.tolist()[:3]}'
                        ),
                    },
                    'Epoch': epoch,
                    'Learning Rate': self.optimizer.param_groups[0]['lr'],
                    'Loss [training]': self.loss_meter.mean,
                    'F1-Score [training]': metrics_logs['f1_score'],
                    'Accuracy Score [training]': metrics_logs['accuracy_score'],
                    'Precision Score [training]': metrics_logs['precision_score'],
                    'Recall Score [training]': metrics_logs['recall_score'],
                }, step=epoch)
            except Exception as e:
                print(e)

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
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=12, help='Number of DataLoader workers')
    parser.add_argument('--patch-size', type=int, default=128, help='Patch size')
    parser.add_argument('--pin-memory', type=bool, default=True, help='Use pin memory for DataLoader?')
    parser.add_argument('--eval-step', type=int, default=1, help='Run evaluation every # step')
    parser.add_argument('--load-model', action='store_true', help='Load model from directories?')
    parser.add_argument('--save-checkpoints', action='store_true', help='Save checkpoints after every epoch?')
    parser.add_argument('--use-amp', action='store_true', help='Use Pytorch Automatic Mixed Precision?')
    args = parser.parse_args()

    model = torchvision.models.efficientnet_v2_l(num_classes=1,)
    trainer = Trainer(args, model)
    torch.cuda.empty_cache()

    try:
        trainer.train()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    # End Training
    logging.info('[TRAINING]: Training finished!')
    torch.cuda.empty_cache()

    model = None
    trainer = None
