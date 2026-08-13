#!/usr/bin/env python3
"""Fail-fast environment check for the released baseline GDN workflow."""

from __future__ import annotations

import sys
from importlib import metadata

EXPECTED = {
    "causal-conv1d": "1.5.3.post1",
    "flash-attn": "2.8.3.post1",
    "transformers": "4.53.3",
    "flash-linear-attention": "0.5.2",
}
EXPECTED_TORCH_RELEASE = "2.7.1"
EXPECTED_CUDA = "12.6"


def main() -> None:
    errors: list[str] = []
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        errors.append(f"Python 3.10-3.12 is required; found {sys.version.split()[0]}")

    versions: dict[str, str] = {}
    for package, expected in EXPECTED.items():
        try:
            found = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed (expected {expected})")
            continue
        versions[package] = found
        if found != expected:
            errors.append(f"{package}=={expected} is required; found {found}")

    try:
        import torch
    except ImportError:
        errors.append("torch is not installed")
    else:
        versions["torch"] = torch.__version__
        versions["cuda_runtime"] = str(torch.version.cuda)
        if torch.__version__.split("+")[0] != EXPECTED_TORCH_RELEASE:
            errors.append(
                f"torch=={EXPECTED_TORCH_RELEASE} is required; found {torch.__version__}"
            )
        if torch.version.cuda != EXPECTED_CUDA:
            errors.append(f"CUDA runtime {EXPECTED_CUDA} is required; found {torch.version.cuda}")
        if not torch.cuda.is_available():
            errors.append("CUDA is not available; released GDN inference requires an NVIDIA GPU")
        else:
            versions["gpu"] = torch.cuda.get_device_name(0)

    if errors:
        print("Environment check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)

    print("Environment check passed:")
    for key, value in versions.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
