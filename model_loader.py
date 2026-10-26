
import torch

from wavelet_decomposition_net import Wavelet_Decomposition_Net


def load_model(args):
    """Create Wavelet_Decomposition_Net on the available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Wavelet_Decomposition_Net(
        img_ch=args.in_channels,
        output_ch=args.num_classes,
    ).to(device)
