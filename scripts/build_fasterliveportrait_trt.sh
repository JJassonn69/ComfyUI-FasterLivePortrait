#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <trt_output_dir>"
  exit 1
fi

TRT_DIR="$1"

echo "Building grid-sample3d TRT plugin..."
/build_grid_sample3d_plugin.sh

# By default the container’s linker searches in /usr/lib/x86_64-linux-gnu (and any dirs in /etc/ld.so.conf.d/).
# Dropping the .so.10.6.0 files there makes them discoverable by dlopen().
# Copy all of the core TensorRT runtime libraries:
cp /opt/TensorRT-10.9.0.34/lib/libnvinfer* \
         /usr/lib/x86_64-linux-gnu/

# Copy helper libs like the ONNX parser:
cp /opt/TensorRT-10.9.0.34/lib/libnvonnxparser* \
         /usr/lib/x86_64-linux-gnu/

# Ensures #include <NvInfer.h> (and others) resolve without needing -I/opt/TensorRT-10.9.0.34/include.
cp /opt/TensorRT-10.9.0.34/include/NvInfer* \
         /usr/include/

ln -s "$(which python3)" /usr/local/bin/python
cd /FasterLivePortrait/
git checkout vbrealtime_upgrade
sed -i 's|python scripts/onnx2trt.py|python /FasterLivePortrait/scripts/onnx2trt.py|g' /FasterLivePortrait/scripts/all_onnx2trt.sh
sed -i '37c\
        ctypes.CDLL("/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so", mode=ctypes.RTLD_GLOBAL)
' /FasterLivePortrait/scripts/onnx2trt.py
chmod +x /FasterLivePortrait/scripts/all_onnx2trt.sh
/FasterLivePortrait/scripts/all_onnx2trt.sh
mv /FasterLivePortrait/checkpoints/*.trt "$TRT_DIR"
mv /grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so "$TRT_DIR"