import torch
import os
import requests
from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
LABEL2NAME = {"vits": "Small", "vitb": "Base", "vitl": "Large", "vitg": "Giant"}


def get_model_url(model):
    return f"https://huggingface.co/depth-anything/Depth-Anything-V2-{LABEL2NAME[model]}/resolve/main/depth_anything_v2_{model}.pth?download=true"


def load(MODEL):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if MODEL not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model type: {MODEL}. Choose from {list(MODEL_CONFIGS.keys())}")
    checkpoint_path = f'checkpoints/depth_anything_v2_{MODEL}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Downloading {MODEL} model from Hugging Face...")
        r = requests.get(get_model_url(MODEL))
        with open(checkpoint_path, 'wb') as f:
            f.write(r.content)
    depth_anything = DepthAnythingV2(**MODEL_CONFIGS[MODEL])
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return depth_anything.to(DEVICE).eval()