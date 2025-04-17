FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TRT_VERSION=10.6.0.26
ARG CUDA_VERSION=12.6 # supports 12.0->12.6

ENV TensorRT_ROOT=/opt/TensorRT-${TRT_VERSION}
ENV LD_LIBRARY_PATH=${TensorRT_ROOT}/lib:${LD_LIBRARY_PATH}
ENV PATH=${TensorRT_ROOT}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-distutils \
        python-is-python3 \
        wget \
        ca-certificates \
        git \
        cmake \
        build-essential \
        libprotobuf-dev \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py

# Make python3 and pip3 point at 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.11 1

RUN pip3 install --no-cache-dir numpy onnx onnxruntime huggingface_hub

WORKDIR /opt
RUN wget --progress=dot:giga \
    https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.6.0/tars/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz \
 && tar -xzf TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz \
 && rm TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz

RUN echo "${TensorRT_ROOT}/lib" > /etc/ld.so.conf.d/tensorrt.conf \
&& ldconfig

RUN git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git /grid-sample3d-trt-plugin
RUN git clone https://github.com/varshith15/FasterLivePortrait.git /FasterLivePortrait

WORKDIR /FasterLivePortrait
RUN huggingface-cli download KwaiVGI/LivePortrait \
  --local-dir ./checkpoints \
  --exclude "*.git*" "README.md" "docs"
RUN pip install --no-cache-dir -r requirements.txt

RUN pip uninstall tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs
RUN pip install tensorrt==10.6.0 tensorrt-cu12==10.6.0 tensorrt-cu12-bindings==10.6.0 tensorrt-cu12-libs==10.6.0

COPY scripts/build_grid_sample3d_plugin.sh /build_grid_sample3d_plugin.sh
COPY scripts/build_fasterliveportrait_trt.sh /build_fasterliveportrait_trt.sh

RUN chmod +x /build_fasterliveportrait_trt.sh
RUN chmod +x /build_grid_sample3d_plugin.sh

WORKDIR /workspace

CMD ["/bin/bash"]
