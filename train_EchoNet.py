import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

from SpikingUMamba import create_umamba_bot_2d

warnings.filterwarnings('ignore')


class Config:
    train_image_dir = "./TRAIN/images"
    train_mask_dir = "./TRAIN/masks"
    valid_image_dir = "./VAL/images"
    valid_mask_dir = "./VAL/masks"

    input_channels = 1
    num_classes = 2
    image_size = (112, 112)
    base_features = 32
    max_features = 320
    fusion_type = 'bidirectional'

    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5

    patience = 20
    min_delta = 1e-4

    ce_weight = 1.0
    dice_weight = 1.0

    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    save_dir = "./checkpoints"
    log_dir = "./logs"


class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, C, H, W] logits
        target: [B, H, W] class indices
        """

        pred_foreground = torch.sigmoid(pred[:, 1, :, :])
        target_foreground = (target == 1).float()

        pred_flat = pred_foreground.reshape(-1)
        target_flat = target_foreground.reshape(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


class EchoNetDataset(Dataset):

    def __init__(self, image_dir: str, mask_dir: str, image_size: Tuple = (112, 112)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        assert len(self.image_files) == len(self.mask_files), "The number of images and masks does not match"

        self.data_info = []
        for img_f, mask_f in zip(self.image_files, self.mask_files):

            parts = img_f.replace('.png', '').split('_')
            if len(parts) >= 3:
                file_name = '_'.join(parts[:-2])
                phase = parts[-2]
                frame = parts[-1]
            else:
                file_name = parts[0]
                phase = 'ED'
                frame = '0000'

            self.data_info.append({
                'image_file': img_f,
                'mask_file': mask_f,
                'file_name': file_name,
                'phase': phase,
                'frame': frame
            })

        print(f"Load the dataset: {image_dir}")
        print(f"Number of images: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        img_path = os.path.join(self.image_dir, info['image_file'])
        image = Image.open(img_path).convert('L')
        image = np.array(image, dtype=np.float32)

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        mask_path = os.path.join(self.mask_dir, info['mask_file'])
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int64)

        mask = (mask > 0).astype(np.int64)

        if image.shape[:2] != self.image_size:
            image = np.array(Image.fromarray(image).resize(self.image_size[::-1], Image.BILINEAR))
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(self.image_size[::-1], Image.NEAREST))

        image = image[np.newaxis, :, :]  # [1, H, W]


        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask, info


class DiceScore:

    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:

        if pred.dim() == 4:
            pred = pred.argmax(dim=1)  # [B, H, W]

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        pred_foreground = (pred == 1)
        target_foreground = (target == 1)

        intersection = np.logical_and(pred_foreground, target_foreground).sum()
        union = pred_foreground.sum() + target_foreground.sum()

        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = 2.0 * intersection / union

        return dice


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    pred_foreground = (pred == 1)
    target_foreground = (target == 1)

    intersection = np.logical_and(pred_foreground, target_foreground).sum()
    union = np.logical_or(pred_foreground, target_foreground).sum()

    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = intersection / union

    return iou


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )

        self.criterion = CombinedLoss(
            ce_weight=config.ce_weight,
            dice_weight=config.dice_weight
        )

        self.dice_metric = DiceScore()

        self.scaler = GradScaler()

        self.best_dice = 0.0
        self.best_epoch = -1
        self.patience_counter = 0

        self.setup_logging()

        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def setup_logging(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config.log_dir, f"training_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train_one_epoch(self, epoch):

        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs} [Train]')

        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.config.device)
            masks = masks.to(self.config.device)

            self.optimizer.zero_grad()

            with autocast():
                output = self.model(images)
                loss = self.criterion(output, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):

        self.model.eval()

        total_dice = 0
        total_iou = 0
        num_batches = 0

        pbar = tqdm(self.valid_loader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs} [Valid]')

        for images, masks, _ in pbar:
            images = images.to(self.config.device)
            masks = masks.to(self.config.device)


            output = self.model(images)


            dice = self.dice_metric(output, masks)
            iou = compute_iou(output, masks)

            total_dice += dice
            total_iou += iou
            num_batches += 1

            pbar.set_postfix({'dice': f'{dice:.4f}', 'iou': f'{iou:.4f}'})

        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches

        return avg_dice, avg_iou

    def save_checkpoint(self, epoch, dice, is_best=False):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_dice': self.best_dice,
            'dice': dice,
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('__')}
        }


        latest_path = os.path.join(self.config.save_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)


        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Save the best model，Dice: {dice:.4f}")

    def train(self):

        self.logger.info("=" * 70)
        self.logger.info("Start training EchoNet-Dynamic split model")
        self.logger.info("=" * 70)
        self.logger.info(f"device: {self.config.device}")
        self.logger.info(f"Size of the training dataset: {len(self.train_loader.dataset)}")
        self.logger.info(f"Size of the validation dataset: {len(self.valid_loader.dataset)}")
        self.logger.info(f"num_classes: {self.config.num_classes} (Two classes)")
        self.logger.info(f"fusion_type: {self.config.fusion_type}")
        self.logger.info(f"Input size: {self.config.image_size}")
        self.logger.info(f"batch_size: {self.config.batch_size}")
        self.logger.info(f"learning_rate: {self.config.learning_rate}")
        self.logger.info(f"early stopping patience: {self.config.patience}")
        self.logger.info("=" * 70)

        train_losses = []
        valid_dices = []

        for epoch in range(self.config.num_epochs):

            train_loss = self.train_one_epoch(epoch)
            train_losses.append(train_loss)


            avg_dice, avg_iou = self.validate(epoch)
            valid_dices.append(avg_dice)


            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]


            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Dice: {avg_dice:.4f} | "
                f"IoU: {avg_iou:.4f} | "
                f"LR: {current_lr:.2e}"
            )


            is_best = avg_dice > self.best_dice + self.config.min_delta
            if is_best:
                self.best_dice = avg_dice
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, avg_dice, is_best=True)
            else:
                self.patience_counter += 1


            self.save_checkpoint(epoch, avg_dice, is_best=False)


            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping trigger！{self.config.patience}epochs without any improvement")
                self.logger.info(f"Best Dice: {self.best_dice:.4f} (Epoch {self.best_epoch + 1})")
                break

        self.logger.info("=" * 70)
        self.logger.info(f"Training completed！Best average Dice: {self.best_dice:.4f}")
        self.logger.info("=" * 70)


        history = {
            'train_losses': train_losses,
            'valid_dices': valid_dices,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch
        }
        history_path = os.path.join(self.config.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)

        return history



def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    config = Config()


    set_seed(config.seed)

    print("=" * 70)
    print("EchoNet-Dynamic split training ")
    print("=" * 70)
    print(f"device: {config.device}")
    print(f"fusion_type: {config.fusion_type}")
    print(f"Input size: {config.image_size}")
    print(f"num_classes: {config.num_classes} (Two classes)")


    if not os.path.exists(config.train_image_dir):
        print(f"Error: The path of the training images does not exist - {config.train_image_dir}")
        return
    if not os.path.exists(config.train_mask_dir):
        print(f"Error: The path of the training masks does not exist. - {config.train_mask_dir}")
        return
    if not os.path.exists(config.valid_image_dir):
        print(f"Error: The path of the validation images does not exist - {config.valid_image_dir}")
        return
    if not os.path.exists(config.valid_mask_dir):
        print(f"Error: The path of the validation masks does not exist - {config.valid_mask_dir}")
        return


    print("\nLoad dataset...")
    train_dataset = EchoNetDataset(
        image_dir=config.train_image_dir,
        mask_dir=config.train_mask_dir,
        image_size=config.image_size
    )

    valid_dataset = EchoNetDataset(
        image_dir=config.valid_image_dir,
        mask_dir=config.valid_mask_dir,
        image_size=config.image_size
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print(f"Number of training dataset samples: {len(train_dataset)}")
    print(f"Training dataset batch: {len(train_loader)}")
    print(f"Number of validation dataset samples: {len(valid_dataset)}")
    print(f"Validation dataset batch: {len(valid_loader)}")


    ed_count = sum(1 for info in train_dataset.data_info if info['phase'] == 'ED')
    es_count = sum(1 for info in train_dataset.data_info if info['phase'] == 'ES')
    print(f"training dataset - ED: {ed_count}, ES: {es_count}")


    print("\nCreate model...")
    model = create_umamba_bot_2d(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        image_size=config.image_size,
        deep_supervision=False,
        base_features=config.base_features,
        max_features=config.max_features
    )


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params:,}")
    print(f"The number of trainable parameters: {trainable_params:,}")


    model = model.to(config.device)


    trainer = Trainer(model, train_loader, valid_loader, config)


    history = trainer.train()

    print(f"\nTraining completed！")
    print(f"The best model is saved in: {config.save_dir}/best_model.pth")
    print(f"The training log is saved in: {config.log_dir}")


if __name__ == "__main__":
    main()