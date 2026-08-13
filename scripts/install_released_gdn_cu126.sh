#!/usr/bin/env bash
set -euo pipefail

# Reproducible A800/CUDA 12.6 installation. Run from the repository root after
# activating a fresh Python 3.10-3.12 environment.
python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install \
  torch==2.7.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126

# causal-conv1d has no PyTorch 2.10 wheel and its source build requires nvcc.
# Use its official PyTorch 2.7/CUDA 12 wheel and retry transient GitHub failures.
python_tag="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
case "$python_tag" in
  cp310|cp311|cp312) ;;
  *) echo "The released-GDN installer supports Python 3.10-3.12." >&2; exit 1 ;;
esac
wheel_name="causal_conv1d-1.5.3.post1+cu12torch2.7cxx11abiTRUE-${python_tag}-${python_tag}-linux_x86_64.whl"
case "$python_tag" in
  cp310) wheel_sha256="3a60ede12aa2bcd0e0cd435956bb65a9d85260381c9d99ea4c45551e3174b894" ;;
  cp311) wheel_sha256="e487480275bbf39ea0c67b29f00614cde34078a2e6bf5b026c29d6316871ff33" ;;
  cp312) wheel_sha256="4f02ed4b9fe16b99589e110a4f684b01c8f97e6925807aad95151846c20d6340" ;;
esac
wheel_dir="$(mktemp -d)"
trap 'rm -rf "$wheel_dir"' EXIT
wheel_url="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/${wheel_name/+/%2B}"
if ! curl --fail --location --retry 2 --retry-all-errors --speed-limit 131072 \
  --speed-time 30 "$wheel_url" --output "$wheel_dir/$wheel_name"; then
  echo "Direct GitHub download is slow or unavailable; trying the ghfast mirror." >&2
  curl --fail --location --retry 5 --retry-all-errors \
    "https://ghfast.top/$wheel_url" --output "$wheel_dir/$wheel_name"
fi
printf '%s  %s\n' "$wheel_sha256" "$wheel_dir/$wheel_name" | sha256sum --check --status
python -m pip install "$wheel_dir/$wheel_name"

flash_wheel_name="flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-${python_tag}-${python_tag}-linux_x86_64.whl"
case "$python_tag" in
  cp310) flash_wheel_sha256="52e290486e8bf00fd48e9993251b492e6500450ee79e91197671719f60229446" ;;
  cp311) flash_wheel_sha256="a37b7740067ab5f5e257ad2921265fb9f4227f3907ceda4a9bd1237270c7f308" ;;
  cp312) flash_wheel_sha256="f87164bb919f5597cb94f7485196be6ac7842b71d7bf8e9f8e9133b9045ab5ea" ;;
esac
flash_wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/${flash_wheel_name/+/%2B}"
if ! curl --fail --location --retry 2 --retry-all-errors --speed-limit 131072 \
  --speed-time 30 "$flash_wheel_url" --output "$wheel_dir/$flash_wheel_name"; then
  echo "Direct GitHub download is slow or unavailable; trying the ghfast mirror." >&2
  curl --fail --location --retry 5 --retry-all-errors \
    "https://ghfast.top/$flash_wheel_url" --output "$wheel_dir/$flash_wheel_name"
fi
printf '%s  %s\n' "$flash_wheel_sha256" "$wheel_dir/$flash_wheel_name" | sha256sum --check --status
python -m pip install "$wheel_dir/$flash_wheel_name"

python -m pip install transformers==4.53.3
python -m pip install flash-linear-attention==0.5.2 --no-build-isolation

python -m pip install -e ".[released-gdn]"
python scripts/check_environment.py
