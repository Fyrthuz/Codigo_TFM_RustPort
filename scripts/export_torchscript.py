#!/usr/bin/env python3
"""Trace models to TorchScript for the Rust numeric parity tests (Fase 3).

Run from the ORIGINAL repo's virtualenv:

    ../Codigo_TFM/.venv/bin/python scripts/export_torchscript.py

Outputs:
    unet_model.pt        (traced UNet, input: (1,3,256,256))
    universeg.pt         (traced UniVerSeg, inputs: image + support_images + support_labels)
    universeg_context.pt (support tensors used for tracing, to reuse in the Rust test)
"""
import os
import sys

import torch

from universeg import universeg


def export_unet(path_out: str) -> None:
    from src.models.unet import UNet

    model = UNet(in_channels=3, out_channels=1, init_features=32).cpu().eval()
    pth = os.path.join(os.path.dirname(os.path.abspath(path_out)), "unet_model.pth")
    state = torch.load(pth, map_location="cpu")
    model.load_state_dict(state)
    example = torch.randn(1, 3, 256, 256)
    traced = torch.jit.trace(model, example)
    traced.save(path_out)
    print(f"UNet traced -> {path_out}")


def export_universeg(path_out: str, context_out: str) -> None:
    model = universeg(pretrained=True).cpu().eval()
    support_images = torch.randn(1, 4, 1, 128, 128)
    support_labels = (torch.rand(1, 4, 1, 128, 128) > 0.5).float()
    example = torch.randn(1, 1, 128, 128)
    traced = torch.jit.trace(model, (example, support_images, support_labels))
    traced.save(path_out)
    torch.save({"support_images": support_images, "support_labels": support_labels}, context_out)
    print(f"UniVerSeg traced -> {path_out}")


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    export_unet(os.path.normpath(os.path.join(base, "..", "unet_model.pt")))
    export_universeg(
        os.path.normpath(os.path.join(base, "..", "universeg.pt")),
        os.path.normpath(os.path.join(base, "..", "universeg_context.pt")),
    )


if __name__ == "__main__":
    sys.exit(main())
