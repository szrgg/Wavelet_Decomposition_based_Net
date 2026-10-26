# Wavelet Decomposition-Based Network for Midbrain Segmentation

This repository contains the PyTorch implementation of
`Wavelet_Decomposition_Net`, a wavelet-enhanced encoder-decoder network for
2D midbrain segmentation.

The current model uses:

- two conventional max-pooling stages in the shallow encoder;
- two Wavelet Fusion Down-sampling Modules (WFDMs) in the deeper encoder;
- four Dynamic Separable Attention Gates (DSAGs) for encoder-decoder skip
  connections;
- adaptive fusion of the original and DSAG-filtered features at the shallowest
  skip connection.

The default input is a three-channel `400 x 400` image. The model returns raw
segmentation logits with shape `[batch_size, num_classes, 400, 400]`.

## Repository structure

```text
Wavelet_Decomposition_based_Net/
├── wavelet_decomposition_net.py   # Network and all modules used by the model
├── requirements.txt              # Runtime dependencies
├── .gitignore                    # Excludes private data, caches and outputs
├── train.py                       # Five-fold training entry point
├── trainer.py                     # Optimization, validation and early stopping
├── test.py                        # Evaluation of one fold at a time
├── make_npz.py                    # Image/mask to NPZ conversion utility
├── utils.py                       # Losses, metrics and test-time utilities
├── datasets/
│   └── dataset_synapse.py         # Dataset loader and augmentation
├── data/
│   └── Midbrain/
│       ├── images/                # Original RGB images
│       ├── labels/                # Original segmentation masks
│       └── all_npz/               # All converted NPZ samples
└── lists/
    └── lists_Midbrain/
        ├── fold_1/
        │   ├── train.txt
        │   ├── val.txt
        │   └── test.txt
        ├── fold_2/
        ├── fold_3/
        ├── fold_4/
        └── fold_5/
```

## Environment requirements

Recommended environment:

- Python 3.10 or 3.11
- PyTorch 2.0 or later
- CUDA-capable GPU with a PyTorch-compatible CUDA runtime (recommended)
- Windows or Linux

Required Python packages:

```text
torch
numpy
scipy
Pillow>=9.1
MedPy
tqdm
tensorboard
```

Example installation with Conda:

```bash
conda create -n wavelet-midbrain python=3.10 -y
conda activate wavelet-midbrain

# Install the appropriate PyTorch build for the local CPU/CUDA environment.
pip install torch

pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching the CUDA environment on
the machine. A GPU is strongly recommended because the model and `400 x 400`
inputs require substantial memory.

`requirements.txt` lists runtime dependencies with version ranges rather than
an exact environment lockfile.

## Preparing NPZ data

The public copy contains empty data directories only. No original images,
masks or NPZ samples are included; provide your own paired data below.

Place the original images and masks under `data/Midbrain`. Their filename
stems must match. For example:

```text
data/
└── Midbrain/
    ├── images/
    │   ├── patient001_img001.png
    │   └── patient001_img002.png
    └── labels/
        ├── patient001_img001.png
        └── patient001_img002.png
```

Convert the paired files with:

```bash
python make_npz.py \
  --image-dir ./data/Midbrain/images \
  --label-dir ./data/Midbrain/labels \
  --output-dir ./data/Midbrain/all_npz
```

On Windows PowerShell, the same command can be written on one line:

```powershell
python make_npz.py --image-dir ./data/Midbrain/images --label-dir ./data/Midbrain/labels --output-dir ./data/Midbrain/all_npz
```

Each generated file contains:

```text
image: uint8 array with shape (400, 400, 3)
label: uint8 array with shape (400, 400), values in {0, 1}
```

The converter changes images to RGB, resizes images with bilinear
interpolation, and resizes masks with nearest-neighbor interpolation. Existing
NPZ files are not overwritten unless `--overwrite` is specified.

The current dataset loader converts the stored image to `float32` but does not
apply automatic intensity normalization. If normalization is added, apply the
same rule consistently during training, validation and testing.

## Patient-level five-fold lists

All NPZ files remain in the same `data/Midbrain/all_npz` directory. Dataset
membership is controlled by text files under `lists/lists_Midbrain`.

Every line contains an NPZ filename without the `.npz` extension:

```text
patient001_img001
patient001_img002
patient005_img001
```

For every fold, fill in:

```text
lists/lists_Midbrain/fold_<k>/train.txt
lists/lists_Midbrain/fold_<k>/val.txt
lists/lists_Midbrain/fold_<k>/test.txt
```

The split must be performed at the patient level:

- all images from one patient must remain in exactly one subset within a fold;
- the test subset should contain approximately 20% of all images;
- the remaining patients should be divided into training and validation sets
  with an approximate image ratio of 7:1;
- the resulting train/validation/test proportions should be approximately
  70%/10%/20%;
- each patient-level subset serves as the independent test subset exactly once
  across the five folds.

The committed list files are placeholders and must be populated before
training or testing.

## Training

Train all five folds with the default paper settings:

```bash
python train.py
```

Training and validation use the same combined loss:

```text
loss = 0.3 * BCEWithLogitsLoss + 0.7 * DiceLoss
```

On-the-fly augmentation is enabled only for the training set:

- horizontal flipping with probability 0.5;
- an in-plane rotation with probability 0.2, using an integer angle sampled
  uniformly from -20 to 19 degrees and retaining the original image size;
- translation with probability 0.2, with independent row and column offsets
  sampled uniformly from -10 to 10 pixels (inclusive), using reflection padding.

The three transforms are independently triggered and applied to images and
masks together. Validation uses resizing and tensor conversion only, without
random augmentation.

Important defaults:

```text
input size:           400 x 400
input channels:       3
number of classes:    2
batch size:           16
maximum epochs:       100 per fold
initial learning rate: 1e-4
minimum learning rate: 1e-6
learning-rate policy: cosine annealing
early-stopping patience: 20 validation epochs
random seed:          1234
```

Early stopping uses independent counters for validation Dice and validation
loss. Training stops if Dice fails to exceed its best value for 20 consecutive
validation epochs **or** loss fails to fall below its lowest value for 20
consecutive validation epochs. Use `--patience` to change this threshold.

Train selected folds only:

```bash
python train.py --folds 1
python train.py --folds 2 3
```

Override common settings when needed:

```bash
python train.py --batch_size 8 --num_workers 4 --deterministic
```

The default batch size follows the experimental configuration but may require
substantial GPU memory. Reduce it if the available device runs out of memory.

Checkpoints and TensorBoard logs are written to:

```text
model/Wavelet_Decomposition_Net/
└── Wavelet_Decomposition_Net_Midbrain_fold<k>_400/
    ├── best_model.pth
    ├── training.log
    └── log/
```

`best_model.pth` contains the weights from the epoch with the highest validation
Dice. The corresponding validation loss is recorded in `crossval_results.txt`.

To inspect TensorBoard logs:

```bash
tensorboard --logdir ./model/Wavelet_Decomposition_Net
```

## Testing

Evaluate one fold with its corresponding independent test list:

```bash
python test.py --fold 1
```

The default checkpoint is selected automatically from the requested fold:

```text
model/Wavelet_Decomposition_Net/
Wavelet_Decomposition_Net_Midbrain_fold1_400/best_model.pth
```

Evaluate the other folds separately:

```bash
python test.py --fold 2
python test.py --fold 3
python test.py --fold 4
python test.py --fold 5
```

A custom checkpoint can be supplied with:

```bash
python test.py --fold 1 --weights ./path/to/checkpoint.pth
```

The evaluation reports Dice, HD95, IoU and Hausdorff distance. Predictions and
logs are stored separately for each fold:

```text
predictions/Wavelet_Decomposition_Net/fold_<k>/<timestamp>/
test_log/Wavelet_Decomposition_Net_fold<k>_<timestamp>/
```

Final cross-validation results can be calculated from the five fold-level test
summaries and reported as the mean plus or minus standard deviation.

## Reproducibility notes

- Keep the patient-level list files fixed for all compared methods.
- Record the random seed used to generate the patient-level splits.
- Do not use test patients for model selection or early stopping.
- Use only the validation subset to select the checkpoint for each fold.
- Report final metrics from the independent test subset of each fold.
- Training defaults to seed `1234`; pass `--deterministic` to request
  deterministic cuDNN operations.

## Quick model check

The following command verifies the output shape without starting training:

```bash
python -c "import torch; from wavelet_decomposition_net import Wavelet_Decomposition_Net; torch.set_grad_enabled(False); model=Wavelet_Decomposition_Net(3, 2).eval(); print(model(torch.randn(1, 3, 400, 400)).shape)"
```

Expected output:

```text
torch.Size([1, 2, 400, 400])
```
