# ComfyUI-FasterLivePortrait

## Building TensorRT engines

```
docker build -t trt-builder .

docker run --rm --gpus all \
  -v "$(pwd)/models":/opt/models_out \
  trt-builder \
  /build_fasterliveportrait_trt.sh /opt/models_out
```