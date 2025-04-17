#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <trt_output_dir>"
  exit 1
fi

TRT_DIR="$1"

echo "Building grid-sample3d TRT plugin..."
/build_grid_sample3d_plugin.sh

sed -i '37c\
        ctypes.CDLL("/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so", mode=ctypes.RTLD_GLOBAL)
' /FasterLivePortrait/scripts/onnx2trt.py

python3 /FasterLivePortrait/scripts/onnx2trt.py
