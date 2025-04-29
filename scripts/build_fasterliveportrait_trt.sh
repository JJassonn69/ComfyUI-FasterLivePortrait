#!/usr/bin/env bash
set -e

# Usage: ./build_fasterliveportrait_trt.sh <input_dir> <onnx_dir> <trt_output_dir>
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <input_dir> <onnx_models_dir> <trt_output_dir>"
  exit 1
fi

INPUT_DIR="$1"
ONNX_DIR="$2"
TRT_OUTPUT_DIR="$3"

# Set TensorRT paths
export TensorRT_ROOT=/opt/TensorRT-10.9.0.34/targets/x86_64-linux-gnu
export LD_LIBRARY_PATH=$TensorRT_ROOT/lib:$LD_LIBRARY_PATH

# Setup directories
PLUGIN_DIR="$INPUT_DIR/grid-sample3d-trt-plugin"
FLP_DIR="$INPUT_DIR/FasterLivePortrait"

echo "🔵 Cloning required repositories..."
git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git "$PLUGIN_DIR"
git clone https://github.com/varshith15/FasterLivePortrait.git "$FLP_DIR"

# Build grid-sample3d plugin
echo "🔵 Building grid-sample3d TensorRT plugin..."
rm -rf "$PLUGIN_DIR/build" && mkdir -p "$PLUGIN_DIR/build"
cd "$PLUGIN_DIR/build"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTensorRT_ROOT="$TensorRT_ROOT" \
  -DCMAKE_PREFIX_PATH="$TensorRT_ROOT" \
  -DCMAKE_LIBRARY_PATH="$TensorRT_ROOT/lib" \
  -DCMAKE_CXX_FLAGS="-I$TensorRT_ROOT/include" \
  -DCMAKE_CUDA_FLAGS="-I$TensorRT_ROOT/include" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$TensorRT_ROOT/lib"

make -j"$(nproc)"

echo "✅ Grid-sample3d plugin built:"
ls -lh libgrid_sample_3d_plugin.so

# Copy TensorRT libraries to standard system locations
echo "🔵 Copying TensorRT libraries to system paths..."
cp /opt/TensorRT-10.9.0.34/lib/libnvinfer* /usr/lib/x86_64-linux-gnu/
cp /opt/TensorRT-10.9.0.34/lib/libnvonnxparser* /usr/lib/x86_64-linux-gnu/
cp /opt/TensorRT-10.9.0.34/include/NvInfer* /usr/include/

# Ensure python symlink (for Docker environments missing it)
ln -sf "$(which python3)" /usr/local/bin/python

# Prepare FasterLivePortrait repo
echo "🔵 Preparing FasterLivePortrait..."
cd "$FLP_DIR"
git checkout vbrealtime_upgrade

# Patch paths in all_onnx2trt.sh and onnx2trt.py
sed -i "s|python scripts/onnx2trt.py|python $FLP_DIR/scripts/onnx2trt.py|g" "$FLP_DIR/scripts/all_onnx2trt.sh"
sed -i "37c\
        ctypes.CDLL(\"$PLUGIN_DIR/build/libgrid_sample_3d_plugin.so\", mode=ctypes.RTLD_GLOBAL)
" "$FLP_DIR/scripts/onnx2trt.py"

# Update all_onnx2trt.sh to use your ONNX directory
sed -i "s|./checkpoints/liveportrait_onnx|$ONNX_DIR|g" "$FLP_DIR/scripts/all_onnx2trt.sh"

# Make the script executable
chmod +x "$FLP_DIR/scripts/all_onnx2trt.sh"

# Run ONNX to TRT conversion
echo "🔵 Running ONNX -> TensorRT conversion..."
"$FLP_DIR/scripts/all_onnx2trt.sh"

# Move output files
echo "🔵 Moving outputs to $TRT_OUTPUT_DIR..."
mkdir -p "$TRT_OUTPUT_DIR"
mv "$FLP_DIR/checkpoints/" "$TRT_OUTPUT_DIR"
mv "$PLUGIN_DIR/build/libgrid_sample_3d_plugin.so" "$TRT_OUTPUT_DIR"

echo "🎉 All done!"
