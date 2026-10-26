
import argparse
import logging
import os
import random
import sys
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from wavelet_decomposition_net import Wavelet_Decomposition_Net


MODEL_NAME = "Wavelet_Decomposition_Net"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate one fold of Wavelet_Decomposition_Net."
    )
    parser.add_argument(
        "--volume_path",
        type=str,
        default="./data/Midbrain/all_npz",
        help="Directory containing all NPZ samples from every fold.",
    )
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Midbrain")
    parser.add_argument("--dataset", type=str, default="Midbrain")
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 6),
        default=1,
        help="Fold to evaluate. The test list is fold_<fold>/test.txt.",
    )
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=400)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Checkpoint path. By default, use the best-Dice model for --fold.",
    )
    parser.add_argument("--test_save_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--non_deterministic",
        action="store_true",
        help="Enable faster non-deterministic CUDNN operations.",
    )
    return parser


def setup_logging(log_folder):
    os.makedirs(log_folder, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(
        os.path.join(log_folder, "test.txt"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def create_test_loader(args):
    dataset = Synapse_dataset(
        base_dir=args.volume_path,
        split=f"fold_{args.fold}/test",
        list_dir=args.list_dir,
    )
    options = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        options["persistent_workers"] = True
    return dataset, DataLoader(dataset, **options)


def inference(args, model, device):
    os.makedirs(args.test_save_dir, exist_ok=True)
    test_dataset, test_loader = create_test_loader(args)
    logging.info("%d samples to test", len(test_dataset))

    model.eval()
    metric_sum = np.zeros((args.num_classes - 1, 4), dtype=float)
    details = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            case_name = batch["case_name"][0]

            with autocast(enabled=device.type == "cuda"):
                metrics = test_single_volume(
                    image,
                    label,
                    model,
                    classes=args.num_classes,
                    patch_size=(args.img_size, args.img_size),
                    test_save_path=args.test_save_dir,
                    case=case_name,
                )

            metrics_array = np.asarray(metrics, dtype=float)
            metric_sum += metrics_array
            details.append((case_name, metrics))
            mean_metrics = metrics_array.mean(axis=0)
            logging.info(
                "Case %s | Dice: %.4f, HD95: %.4f, IoU: %.4f, HD: %.4f",
                case_name,
                *mean_metrics,
            )

    if not details:
        raise RuntimeError("The test list is empty; no samples were evaluated.")

    class_metrics = metric_sum / len(details)
    for class_index, metrics in enumerate(class_metrics, start=1):
        logging.info(
            "Class %d | Mean Dice: %.4f, Mean HD95: %.4f, "
            "Mean IoU: %.4f, Mean HD: %.4f",
            class_index,
            *metrics,
        )

    overall = class_metrics.mean(axis=0)
    logging.info(
        "Overall | Mean Dice: %.4f, HD95: %.4f, IoU: %.4f, HD: %.4f",
        *overall,
    )
    return class_metrics, overall, details


def save_results(log_folder, class_metrics, overall, details):
    with open(os.path.join(log_folder, "detailed.txt"), "w", encoding="utf-8") as file:
        for case_name, metrics in details:
            mean_metrics = np.asarray(metrics, dtype=float).mean(axis=0)
            file.write(
                f"{case_name} | Dice {mean_metrics[0]:.4f}, "
                f"HD95 {mean_metrics[1]:.4f}, IoU {mean_metrics[2]:.4f}, "
                f"HD {mean_metrics[3]:.4f}\n"
            )

    with open(os.path.join(log_folder, "summary.txt"), "w", encoding="utf-8") as file:
        for class_index, metrics in enumerate(class_metrics, start=1):
            file.write(
                f"Class {class_index} | Dice {metrics[0]:.4f}, "
                f"HD95 {metrics[1]:.4f}, IoU {metrics[2]:.4f}, "
                f"HD {metrics[3]:.4f}\n"
            )
        file.write(
            f"Overall | Dice {overall[0]:.4f}, HD95 {overall[1]:.4f}, "
            f"IoU {overall[2]:.4f}, HD {overall[3]:.4f}\n"
        )


def main(args=None):
    args = build_parser().parse_args(args)
    if args.weights is None:
        experiment_name = (
            f"{MODEL_NAME}_{args.dataset}_fold{args.fold}_{args.img_size}"
        )
        args.weights = os.path.join(
            "model", MODEL_NAME, experiment_name, "best_model.pth"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.test_save_dir is None:
        args.test_save_dir = os.path.join(
            "predictions", MODEL_NAME, f"fold_{args.fold}", timestamp
        )
    if args.log_dir is None:
        args.log_dir = os.path.join(
            "test_log", f"{MODEL_NAME}_fold{args.fold}_{timestamp}"
        )

    setup_logging(args.log_dir)
    logging.info("Arguments: %s", args)
    logging.info("Evaluating fold %d with fold_%d/test.txt", args.fold, args.fold)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    deterministic = not args.non_deterministic
    cudnn.deterministic = deterministic
    cudnn.benchmark = not deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model = Wavelet_Decomposition_Net(
        img_ch=args.in_channels,
        output_ch=args.num_classes,
    ).to(device)

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    logging.info("Loaded weights from %s", args.weights)

    class_metrics, overall, details = inference(args, model, device)
    save_results(args.log_dir, class_metrics, overall, details)
    logging.info(
        "Finished evaluation. Logs: %s | Predictions: %s",
        args.log_dir,
        args.test_save_dir,
    )


if __name__ == "__main__":
    main()
