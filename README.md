# ComfyUI-FasterLivePortrait

## Building TensorRT Engines

This repository includes scripts to build TensorRT engines optimized for the FasterLivePortrait pipeline inside ComfyUI.

---

## 1. Build the Docker Image

```bash
docker build -t trt-builder -f docker/Dockerfile.base .
```

---

## 2. Prepare the Models Directory

```bash
cd ComfyUI  # Navigate to your ComfyUI installation

# Ensure the liveportrait_onnx models are downloaded here
mkdir -p "$(pwd)/models/liveportrait_onnx"
```

---

## 3. Run the TensorRT Build Script

```bash
docker run --rm --gpus all \
  -v "$(pwd)/models":/workspace/ComfyUI/models \
  trt-builder \
  bash /workspace/ComfyUI/custom_nodes/ComfyUI-FasterLivePortrait/scripts/build_fasterliveportrait_trt.sh \
    /workspace/ComfyUI/custom_nodes/ComfyUI-FasterLivePortrait/assets \
    /workspace/ComfyUI/models/liveportrait_onnx \
    /workspace/ComfyUI/models/liveportrait_onnx
```

---

## Outputs

- TensorRT engine files (`.engine`) will be generated into `/workspace/ComfyUI/models/liveportrait_onnx/`.
- The compiled TensorRT plugin library (`libgrid_sample_3d_plugin.so`) will also be copied into `/workspace/ComfyUI/models/liveportrait_onnx/`.

## Notes
- You **must** have the ONNX models pre-downloaded into `/models/liveportrait_onnx/` before running the build script.
- The Docker image automatically includes TensorRT 10.9.0.34 built against CUDA 12.8.
