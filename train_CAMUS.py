import os
import sys
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

    train_image_dir = "./train/images"
    train_mask_dir = "./train/masks"
    valid_image_dir = "./valid/images"
    valid_mask_dir = "./valid/masks"


    CLASS_MAPPING = {
        0: 0,
        29: 1,
        76: 2,
        149: 3,
    }


    input_channels = 1
    num_classes = 4
    image_size = (224, 224)
    base_features = 32
    max_features = 320
    fusion_type = 'bidirectional'


    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5


    patience = 15
    min_delta = 1e-4


    ce_weight = 1.0
    dice_weight = 1.0


    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    save_dir = "./checkpoints"
    log_dir = "./logs"



class DiceLoss(nn.Module):


    def __init__(self, num_classes: int, ignore_index: int = 0, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, C, H, W] logits
        target: [B, H, W] class indices
        """

        pred_softmax = F.softmax(pred, dim=1)


        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        dice_loss = 0
        for c in range(1, self.num_classes):
            pred_c = pred_softmax[:, c]
            target_c = target_one_hot[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice)

        return dice_loss / (self.num_classes - 1)


class CombinedLoss(nn.Module):

    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice



class CAMUSDataset(Dataset):

    def __init__(self, image_dir: str, mask_dir: str, class_mapping: Dict,
                 image_size: Tuple = (224, 224)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_mapping = class_mapping
        self.image_size = image_size


        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])


        assert len(self.image_files) == len(self.mask_files), "The number of images and masks does not match"

        print(f"Load the dataset: {image_dir}")
        print(f"Number of images: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        remapped = np.zeros_like(mask, dtype=np.int64)
        for original, new in self.class_mapping.items():
            remapped[mask == original] = new
        return remapped

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        image = np.array(image, dtype=np.float32)


        image = (image - image.min()) / (image.max() - image.min() + 1e-8)


        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int64)


        mask = self._remap_mask(mask)


        if image.shape[:2] != self.image_size:
            image = np.array(Image.fromarray(image).resize(self.image_size[::-1], Image.BILINEAR))
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(self.image_size[::-1], Image.NEAREST))


        image = image[np.newaxis, :, :]  # [1, H, W]


        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask



class DiceScore:

    def __init__(self, num_classes: int, ignore_index: int = 0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, List[float]]:
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)  # [B, H, W]

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        dice_scores = []
        for c in range(1, self.num_classes):
            pred_c = (pred == c)
            target_c = (target == c)

            intersection = np.logical_and(pred_c, target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union == 0:
                dice = 1.0 if intersection == 0 else 0.0
            else:
                dice = 2.0 * intersection / union

            dice_scores.append(dice)

        mean_dice = np.mean(dice_scores)
        return mean_dice, dice_scores


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:

    pred = pred.argmax(dim=1) if pred.dim() == 4 else pred
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    ious = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        ious.append(iou)

    return np.mean(ious)



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
            num_classes=config.num_classes,
            ce_weight=config.ce_weight,
            dice_weight=config.dice_weight
        )


        self.dice_metric = DiceScore(config.num_classes)


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

        for batch_idx, (images, masks) in enumerate(pbar):
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
        all_dice_scores = []
        num_batches = 0

        pbar = tqdm(self.valid_loader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs} [Valid]')

        for images, masks in pbar:
            images = images.to(self.config.device)
            masks = masks.to(self.config.device)


            output = self.model(images)


            mean_dice, dice_scores = self.dice_metric(output, masks)
            mean_iou = compute_iou(output, masks, self.config.num_classes)

            total_dice += mean_dice
            total_iou += mean_iou
            all_dice_scores.append(dice_scores)
            num_batches += 1

            pbar.set_postfix({'dice': f'{mean_dice:.4f}', 'iou': f'{mean_iou:.4f}'})

        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches


        all_dice_scores = np.array(all_dice_scores)
        class_dices = all_dice_scores.mean(axis=0)

        return avg_dice, avg_iou, class_dices

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
            self.logger.info(f"Save the best model，average Dice: {dice:.4f}")

    def load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_dice = checkpoint['best_dice']
        self.best_epoch = checkpoint['epoch']
        self.logger.info(f"Load checkpoint，best Dice: {self.best_dice:.4f}")
        return checkpoint['epoch']

    def train(self):

        self.logger.info("=" * 70)
        self.logger.info("start training")
        self.logger.info("=" * 70)
        self.logger.info(f"device: {self.config.device}")
        self.logger.info(f"Size of the training dataset: {len(self.train_loader.dataset)}")
        self.logger.info(f"Size of the validation dataset: {len(self.valid_loader.dataset)}")
        self.logger.info(f"num_classes: {self.config.num_classes}")
        self.logger.info(f"fusion_type: {self.config.fusion_type}")
        self.logger.info(f"batch_size: {self.config.batch_size}")
        self.logger.info(f"learning_rate: {self.config.learning_rate}")
        self.logger.info(f"early stopping patience: {self.config.patience}")
        self.logger.info("=" * 70)

        train_losses = []
        valid_dices = []

        for epoch in range(self.config.num_epochs):

            train_loss = self.train_one_epoch(epoch)
            train_losses.append(train_loss)


            avg_dice, avg_iou, class_dices = self.validate(epoch)
            valid_dices.append(avg_dice)


            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]


            class_names = ['LV Cavity', 'LV Myocardium', 'LA Cavity']
            class_dice_str = ', '.join([f'{name}: {dice:.4f}' for name, dice in zip(class_names, class_dices)])

            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Average Dice: {avg_dice:.4f} | "
                f"mIoU: {avg_iou:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            self.logger.info(f"  Various classes Dice - {class_dice_str}")


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
                self.logger.info(f"Best average Dice: {self.best_dice:.4f} (Epoch {self.best_epoch + 1})")
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
    print("CAMUS split training")
    print("=" * 70)
    print(f"device: {config.device}")
    print(f"fusion_type: {config.fusion_type}")
    print(f"image_size: {config.image_size}")
    print(f"num_classes: {config.num_classes}")


    print("\nLoad dataset...")
    train_dataset = CAMUSDataset(
        image_dir=config.train_image_dir,
        mask_dir=config.train_mask_dir,
        class_mapping=config.CLASS_MAPPING,
        image_size=config.image_size
    )

    valid_dataset = CAMUSDataset(
        image_dir=config.valid_image_dir,
        mask_dir=config.valid_mask_dir,
        class_mapping=config.CLASS_MAPPING,
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