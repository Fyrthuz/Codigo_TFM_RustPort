#!/usr/bin/env python3
"""Export model weights to safetensors for the Rust port (no third-party deps).

Writes the safetensors format manually (header JSON + raw little-endian data),
so it works without the `safetensors` pip package.

Run from the ORIGINAL repo's virtualenv (has torch + universeg):

    ../Codigo_TFM/.venv/bin/python scripts/export_safetensors.py

Outputs:
    unet_model.safetensors
    universeg.safetensors
"""
import json
import os
import struct
import sys

import numpy as np
import torch

DTYPE_MAP = {
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float16: "F16",
    torch.int64: "I64",
    torch.int32: "I32",
}


def _save_file(tensors, path: str) -> None:
    header = {}
    data = bytearray()
    for name, t in tensors.items():
        t = t.contiguous()
        np_arr = t.cpu().numpy()
        if np_arr.dtype == np.float64:
            np_arr = np_arr.astype(np.float32)
        raw = np_arr.tobytes()
        start = len(data)
        data += raw
        end = len(data)
        header[name] = {
            "dtype": "F32",
            "shape": list(np_arr.shape),
            "data_offsets": [start, end],
        }
    header["__metadata__"] = {}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(bytes(data))
    print(f"{len(tensors)} tensors -> {path}")


def export_unet(path_in: str, path_out: str) -> None:
    state = torch.load(path_in, map_location="cpu")
    state = {k: v for k, v in state.items() if v.dim() != 0}
    _save_file(state, path_out)


def export_universeg(path_out: str) -> None:
    from universeg import universeg

    model = universeg(pretrained=True).cpu().eval()
    _save_file(model.state_dict(), path_out)


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    unet_in = os.path.normpath(os.path.join(base, "..", "unet_model.pth"))
    unet_out = os.path.normpath(os.path.join(base, "..", "unet_model.safetensors"))
    universeg_out = os.path.normpath(os.path.join(base, "..", "universeg.safetensors"))

    export_unet(unet_in, unet_out)
    export_universeg(universeg_out)


if __name__ == "__main__":
    sys.exit(main())
