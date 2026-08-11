"""Wrapper that disables cuDNN before importing train module.
Used by accelerate launch to ensure cuDNN is disabled in all subprocesses.
"""
import torch
torch.backends.cudnn.enabled = False

from src.image_cond.train import main
main()
