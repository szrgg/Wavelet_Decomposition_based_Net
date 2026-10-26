

import argparse
import logging
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets.dataset_synapse import RandomGenerator, Synapse_dataset
from trainer import trainer_synapse
from wavelet_decomposition_net import Wavelet_Decomposition_Net


MODEL_NAME = "Wavelet_Decomposition_Net"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train Wavelet_Decomposition_Net with cross-validation."
    )
    parser.add_argument("--root_path", type=str, default="./data/Midbrain/all_npz")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Midbrain")
    parser.add_argument("--dataset", type=str, default="Midbrain")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=400)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help=(
            "Stop a fold when validation Dice fails to improve OR validation "
            "loss fails to decrease for this many consecutive epochs. "
            "Each metric has an independent counter."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic CUDNN operations (slower but reproducible).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Cross-validation folds to train, for example: --folds 1 2 3 4 5",
    )
    return parser


def create_loader(dataset, args, *, shuffle, drop_last=False):
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last,
    }
    if args.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **loader_options)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args=None):
    args = build_parser().parse_args(args)
    seed_everything(args.seed)

    cudnn.deterministic = args.deterministic
    cudnn.benchmark = not args.deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    invalid_folds = [fold for fold in args.folds if fold not in range(1, 6)]
    if invalid_folds:
        raise ValueError(f"Fold numbers must be between 1 and 5: {invalid_folds}")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Using device: %s", device)

    fold_results = []
    model_root = os.path.join("model", MODEL_NAME)

    for fold in args.folds:
        logging.info("==== Fold %d Training Start ====", fold)

        train_dataset = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split=f"fold_{fold}/train",
            transform=RandomGenerator(
                output_size=(args.img_size, args.img_size), augment=True
            ),
        )
        val_dataset = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split=f"fold_{fold}/val",
            transform=RandomGenerator(
                output_size=(args.img_size, args.img_size), augment=False
            ),
        )
        train_loader = create_loader(
            train_dataset, args, shuffle=True, drop_last=True
        )
        val_loader = create_loader(val_dataset, args, shuffle=False)

        logging.info(
            "Fold %d - Train samples: %d, Val samples: %d",
            fold,
            len(train_dataset),
            len(val_dataset),
        )

        experiment_name = f"{MODEL_NAME}_{args.dataset}_fold{fold}_{args.img_size}"
        snapshot_path = os.path.join(model_root, experiment_name)
        os.makedirs(snapshot_path, exist_ok=True)

        file_handler = logging.FileHandler(
            os.path.join(snapshot_path, "training.log"), encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        try:
            model = Wavelet_Decomposition_Net(
                img_ch=args.in_channels,
                output_ch=args.num_classes,
            ).to(device)
            best_dice, loss_at_best_dice = trainer_synapse(
                args, model, train_loader, val_loader, snapshot_path
            )
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()

        fold_results.append((fold, best_dice, loss_at_best_dice))
        logging.info(
            "Fold %d Completed - Best Dice: %.4f, Loss at Best Dice: %.4f",
            fold,
            best_dice,
            loss_at_best_dice,
        )

    mean_dice = float(np.mean([result[1] for result in fold_results]))
    mean_loss_at_best_dice = float(np.mean([result[2] for result in fold_results]))
    logging.info("==== Cross-Validation Completed ====")
    logging.info(
        "Mean Best Dice: %.4f, Mean Loss at Best Dice: %.4f",
        mean_dice, mean_loss_at_best_dice,
    )

    os.makedirs(model_root, exist_ok=True)
    results_file = os.path.join(model_root, "crossval_results.txt")
    with open(results_file, "w", encoding="utf-8") as file:
        for fold, best_dice, loss_at_best_dice in fold_results:
            file.write(
                f"Fold {fold}: Best Dice {best_dice:.4f}, "
                f"Loss at Best Dice {loss_at_best_dice:.4f}\n"
            )
        file.write(f"Mean Best Dice: {mean_dice:.4f}\n")
        file.write(f"Mean Loss at Best Dice: {mean_loss_at_best_dice:.4f}\n")
    logging.info("Cross-validation results saved to %s", results_file)


if __name__ == "__main__":
    main()
