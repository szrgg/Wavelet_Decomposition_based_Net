

import logging
import os

import torch
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils import DiceLoss, one_hot_targets


def _mean_foreground_dice(predictions, labels, num_classes):
    """Return the mean per-sample, per-foreground-class Dice score."""
    scores = []
    for prediction, label in zip(predictions, labels):
        for class_index in range(1, num_classes):
            pred_mask = prediction == class_index
            label_mask = label == class_index
            denominator = pred_mask.sum().item() + label_mask.sum().item()
            if denominator == 0:
                scores.append(1.0)
            else:
                intersection = torch.logical_and(pred_mask, label_mask).sum().item()
                scores.append(2.0 * intersection / denominator)
    return sum(scores) / len(scores) if scores else 0.0


def trainer_synapse(args, model, train_loader, val_loader, snapshot_path):

    if args.patience < 1 or args.max_epochs < 1:
        raise ValueError("patience and max_epochs must be positive integers")
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Training and validation loaders must both contain batches")
    os.makedirs(snapshot_path, exist_ok=True)
    logging.info("Snapshot path: %s", snapshot_path)

    device = next(model.parameters()).device
    bce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )
    writer = SummaryWriter(os.path.join(snapshot_path, "log"))

    # Save the first validated model even when its foreground Dice is zero.
    best_dice = float("-inf")
    best_loss_for_stopping = float("inf")
    loss_at_best_dice = float("inf")
    best_epoch = 0
    dice_no_improvement_epochs = 0
    loss_no_improvement_epochs = 0
    iteration = 0

    try:
        for epoch in range(args.max_epochs):
            model.train()
            train_loss_sum = 0.0
            logging.info("Epoch %d/%d", epoch + 1, args.max_epochs)

            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                outputs = model(images)
                loss_bce = bce_loss(outputs, one_hot_targets(labels, outputs))
                loss_dice = dice_loss(outputs, labels, softmax=True)
                loss = 0.3 * loss_bce + 0.7 * loss_dice

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                iteration += 1
                if iteration % 20 == 0:
                    writer.add_scalar("train/loss", loss.item(), iteration)
                    logging.info(
                        "Iteration %d, Loss: %.4f, LR: %.6f",
                        iteration,
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )

            average_train_loss = train_loss_sum / len(train_loader)
            writer.add_scalar("train/epoch_loss", average_train_loss, epoch)
            logging.info(
                "Epoch %d Average Training Loss: %.4f",
                epoch + 1,
                average_train_loss,
            )
            scheduler.step()

            model.eval()
            validation_loss_sum = 0.0
            validation_dice_sum = 0.0
            validation_sample_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)
                    outputs = model(images)
                    batch_size = labels.shape[0]

                    loss_bce = bce_loss(outputs, one_hot_targets(labels, outputs))
                    loss_dice = dice_loss(outputs, labels, softmax=True)
                    validation_loss_sum += (
                        0.3 * loss_bce.item() + 0.7 * loss_dice.item()
                    ) * batch_size

                    predictions = torch.argmax(outputs, dim=1)
                    validation_dice_sum += _mean_foreground_dice(
                        predictions, labels, args.num_classes
                    ) * batch_size
                    validation_sample_count += batch_size

            average_validation_loss = validation_loss_sum / validation_sample_count
            average_validation_dice = validation_dice_sum / validation_sample_count
            writer.add_scalar("val/loss", average_validation_loss, epoch)
            writer.add_scalar("val/dice", average_validation_dice, epoch)
            logging.info(
                "Epoch %d Validation Loss: %.4f, Dice: %.4f",
                epoch + 1,
                average_validation_loss,
                average_validation_dice,
            )

            # Independent counters: improving one metric cannot reset the other.
            if average_validation_dice > best_dice:
                best_dice = average_validation_dice
                loss_at_best_dice = average_validation_loss
                best_epoch = epoch + 1
                dice_no_improvement_epochs = 0
                checkpoint = os.path.join(snapshot_path, "best_model.pth")
                torch.save(model.state_dict(), checkpoint)
                logging.info(
                    "Saved best-Dice model to %s (epoch %d, Dice %.4f, loss %.4f)",
                    checkpoint, best_epoch, best_dice, loss_at_best_dice,
                )
            else:
                dice_no_improvement_epochs += 1

            if average_validation_loss < best_loss_for_stopping:
                best_loss_for_stopping = average_validation_loss
                loss_no_improvement_epochs = 0
            else:
                loss_no_improvement_epochs += 1

            logging.info(
                "Validation non-improvement counters - Dice: %d/%d, loss: %d/%d.",
                dice_no_improvement_epochs, args.patience,
                loss_no_improvement_epochs, args.patience,
            )
            stop_for_dice = dice_no_improvement_epochs >= args.patience
            stop_for_loss = loss_no_improvement_epochs >= args.patience
            if stop_for_dice or stop_for_loss:
                stalled_metrics = []
                if stop_for_dice:
                    stalled_metrics.append("validation Dice")
                if stop_for_loss:
                    stalled_metrics.append("validation loss")
                logging.info(
                    "Early stopping at epoch %d: %s failed to improve for %d "
                    "consecutive validation epochs (OR rule).",
                    epoch + 1, " and ".join(stalled_metrics), args.patience,
                )
                break
    finally:
        writer.close()

    logging.info(
        "Training finished. Best validation Dice %.4f at epoch %d, loss %.4f.",
        best_dice, best_epoch, loss_at_best_dice,
    )
    return best_dice, loss_at_best_dice
