"""
Trainer — full training loop with AMP, grad accumulation, cosine LR, TensorBoard.
"""
import os
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.edge_detection import build_edge_gt
from .losses import CombinedLoss
from ..evaluation.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion,
                 optimizer, scheduler, cfg, device):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.cfg          = cfg
        self.device       = device

        train_cfg = cfg.get("training", {})
        out_cfg   = cfg.get("output", {})
        ds_cfg    = cfg.get("dataset", {})

        self.epochs       = train_cfg.get("epochs", 100)
        self.use_amp      = train_cfg.get("amp", True) and device.type == "cuda"
        self.accum_steps  = train_cfg.get("accumulation_steps", 1)
        self.grad_clip    = train_cfg.get("gradient_clip", 1.0)
        self.log_every    = out_cfg.get("log_every", 50)
        self.save_every   = out_cfg.get("save_every", 10)
        self.num_classes  = ds_cfg.get("num_classes", 3)
        self.ignore_index = ds_cfg.get("ignore_index", 255)

        self.ckpt_dir = Path(out_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.log_dir  = Path(out_cfg.get("log_dir", "outputs/logs"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.scaler  = GradScaler("cuda", enabled=self.use_amp)
        self.writer  = SummaryWriter(str(self.log_dir)) if out_cfg.get("tensorboard", True) else None
        self.metrics = SegmentationMetrics(num_classes=self.num_classes, ignore_index=self.ignore_index)

        self.best_miou   = 0.0
        self.global_step = 0

    def train(self):
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_losses = self._train_epoch(epoch)
            val_metrics  = self._validate(epoch)
            elapsed = time.time() - t0

            miou = val_metrics.get("miou", 0.0)
            logger.info(
                f"Epoch {epoch:03d}/{self.epochs}  "
                f"loss={train_losses['total']:.4f}  "
                f"val_mIoU={miou:.4f}  ({elapsed:.1f}s)"
            )

            if self.writer:
                for k, v in train_losses.items():
                    self.writer.add_scalar(f"train/{k}_loss", v, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                self.writer.add_scalar("lr", self._get_lr(), epoch)

            if miou > self.best_miou:
                self.best_miou = miou
                self._save_checkpoint(epoch, "best", val_metrics)
                logger.info(f"  New best mIoU: {miou:.4f}")

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch:03d}", val_metrics)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(miou)
                else:
                    self.scheduler.step()

        if self.writer:
            self.writer.close()
        logger.info(f"Training complete. Best val mIoU: {self.best_miou:.4f}")

    def _train_epoch(self, epoch):
        self.model.train()
        running = {k: 0.0 for k in ["total", "seg", "dice", "focal", "edge"]}
        n = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)

        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            masks  = batch["mask"].long().to(self.device, non_blocking=True)
            edge_gt = build_edge_gt(masks, dilation=1)

            with autocast("cuda", enabled=self.use_amp):
                outputs   = self.model(images)
                loss_dict = self.criterion(
                    seg_logits  = outputs["seg"],
                    seg_targets = masks,
                    edge_logits = outputs["edge"],
                    edge_gt     = edge_gt,
                )
                loss = loss_dict["total"] / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for k, v in loss_dict.items():
                running[k] += v.item()
            n += 1
            self.global_step += 1

            if step % self.log_every == 0:
                pbar.set_postfix(loss=f"{loss_dict['total'].item():.4f}",
                                 dice=f"{loss_dict['dice'].item():.4f}")

        return {k: v / n for k, v in running.items()}

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        self.metrics.reset()

        for batch in tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            masks  = batch["mask"].long().to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)

            preds = outputs["seg"].argmax(dim=1)
            self.metrics.update(preds, masks)

        return self.metrics.compute()

    def _save_checkpoint(self, epoch, tag, metrics):
        path = self.ckpt_dir / f"{tag}.pth"
        torch.save({
            "epoch": epoch, "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics, "best_miou": self.best_miou,
        }, str(path))

    def _get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


def build_optimizer(model, cfg):
    opt_cfg = cfg.get("training", {}).get("optimizer", {})
    name = opt_cfg.get("name", "adamw").lower()
    lr   = opt_cfg.get("lr", 1e-4)
    wd   = opt_cfg.get("weight_decay", 1e-4)

    enc_params, other_params = model.get_params()
    param_groups = [
        {"params": enc_params,   "lr": lr * 0.1},
        {"params": other_params, "lr": lr},
    ]

    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(param_groups)
    elif name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=wd, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    sch_cfg = cfg.get("training", {}).get("scheduler", {})
    name    = sch_cfg.get("name", "cosine")
    epochs  = cfg.get("training", {}).get("epochs", 100)
    warmup  = sch_cfg.get("warmup_epochs", 5)
    min_lr  = sch_cfg.get("min_lr", 1e-6)

    if name == "cosine":
        main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup, 1), eta_min=min_lr)
        if warmup > 0:
            warm_sch = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warm_sch, main_sch], milestones=[warmup])
        return main_sch
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == "poly":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: (1 - e/epochs)**0.9)
    raise ValueError(f"Unknown scheduler: {name}")